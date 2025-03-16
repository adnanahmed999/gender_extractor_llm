import json
import math
import re
from urllib.parse import urlparse, parse_qs
import pandas as pd
import streamlit as st
from google import genai
from googleapiclient.discovery import build

st.title("💭♂️♀️Gender Extractor")

YOUTUBE_API_KEY = st.secrets['YOUTUBE_API_KEY']
GEMINI_API_KEY = st.secrets['GEMINI_API_KEY']

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

def get_corrected_csv_name(pre_correction_csv_name):
    # Correct the csv name for file system
    invalid_chars = r'[\/:*?"<>|]'
    csv_name = re.sub(invalid_chars, '_', pre_correction_csv_name)
    csv_name = csv_name.strip()
    return csv_name

def get_csv_name(video_id):
    with st.spinner(text='Getting video details', show_time=True):
        try:
            request = youtube.videos().list(
                part="snippet",
                id=video_id
            )
            response = request.execute()

            video_title = response["items"][0]["snippet"]["title"]
            video_channel = response["items"][0]["snippet"]["channelTitle"]
            pre_correction_csv_name = f'''{video_channel} - {video_title} - {video_id}.csv'''
            csv_name = get_corrected_csv_name(pre_correction_csv_name)

        except Exception as e:
            st.write(f"An error occurred: {e}")
            exit()

    st.write('✅Video details')
    return csv_name

def get_youtube_comments(video_id):
    with st.spinner(text='Getting video comments', show_time=True):
        comments_data = []
        next_page_token = None

        while True:
            request = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,  # Max limit per request
                textFormat="plainText",
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response.get("items", []):
                # Extract top-level comment
                top_comment = item["snippet"]["topLevelComment"]["snippet"]
                top_comment_author = top_comment["authorDisplayName"]
                top_comment_text = top_comment["textDisplay"]
                top_comment_published = top_comment["publishedAt"]

                comments_data.append([top_comment_author, top_comment_text, top_comment_published, "Top-Level Comment"])

                # Extract replies if available
                if "replies" in item:
                    for reply in item["replies"]["comments"]:
                        reply_author = reply["snippet"]["authorDisplayName"]
                        reply_text = reply["snippet"]["textDisplay"]
                        reply_published = reply["snippet"]["publishedAt"]

                        comments_data.append([reply_author, reply_text, reply_published, "Reply"])

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break  # No more pages left

    st.write('✅Video comments')
    return pd.DataFrame(comments_data, columns=["Author", "Comment", "Published At", "Type"])

def get_first_n_distinct_authors(comments_df, n):
    authors = comments_df['Author'].dropna().unique()
    cleaned_authors = [author[1:] if author.startswith('@') else author for author in authors]
    if len(cleaned_authors) >= 15000:
        cleaned_authors = cleaned_authors[:15000]
    return cleaned_authors

def process_in_chunks(unknown_gender_users, chunk_size=500):
  num_chunks = math.ceil(len(unknown_gender_users) / chunk_size)
  for i in range(num_chunks):
    start_index = i * chunk_size
    end_index = min((i + 1) * chunk_size, len(unknown_gender_users))
    yield unknown_gender_users[start_index:end_index]

def get_video_id(video_link):
    with st.spinner("Extracting video id"):
        parsed_url = urlparse(video_link)

        # Check if the URL is a valid YouTube link containing a query string with 'v' key
        if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
            # Extract the query parameters
            query_params = parse_qs(parsed_url.query)

            # Return the video ID if the 'v' key exists
            if 'v' in query_params:
                video_id = query_params['v'][0]
                # st.write('Video Id: ', video_id)
                return video_id

    return ''

def get_gender_df(authors):
    with st.spinner(text='Extracting gender', show_time=True):
        system_prompt = (
            "You are expert in classifying youtube usernames as male, female or unknown. Given a list of usernames, for each and every username, classify its gender as M for male, F for female or U for unknown.\n"
            "Provide a json. Keep username as key and gender as value.\n"
            "Format of the json:\n"
            "{\n"
            "    'username': gender,\n"
            "}"
        )

        unknown_gender_users = authors

        df_gender = pd.DataFrame(columns=['username', 'gender'])

        ITERATION_COUNT = 5

        for i in range(ITERATION_COUNT):
            if len(unknown_gender_users) == 0:
                break

            st.write(f"🔄Iteration {i + 1}/{ITERATION_COUNT}. 👥Users: {len(unknown_gender_users)}")
            unknown_gender_users_new = []

            for unknown_gender_users_chunk in process_in_chunks(unknown_gender_users):

                response = client.models.generate_content(
                    model="gemini-2.0-pro-exp-02-05",
                    contents=f'''
                          {system_prompt}\n
                          {unknown_gender_users_chunk}
                        '''
                )

                response_text = response.candidates[0].content.parts[0].text

                json_start = response_text.find('{')
                if '}' not in response_text:
                    last_comma_index = response_text.rfind(',')
                    response_text = response_text[:last_comma_index] + "}"
                json_end = response_text.rfind('}') + 1

                json_string = response_text[json_start:json_end]
                json_output = json.loads(json_string)

                for username, gender in json_output.items():
                    if gender == 'M' or gender == 'F':
                        df_gender = pd.concat([df_gender, pd.DataFrame({'username': [username], 'gender': [gender]})],
                                              ignore_index=True)
                    elif gender == 'U':
                        unknown_gender_users_new.append(username)

            unknown_gender_users = unknown_gender_users_new

        for username in unknown_gender_users:
            df_gender = pd.concat([df_gender, pd.DataFrame({'username': [username], 'gender': ['U']})], ignore_index=True)

    st.write('✅Extracted gender')

    return df_gender

def download_csv(df, csv_name):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=csv_name,
        mime='text/csv'
    )

def run(video_link):
    video_id = get_video_id(video_link)
    csv_name = get_csv_name(video_id)
    comments_df = get_youtube_comments(video_id)
    authors = get_first_n_distinct_authors(comments_df, 15000)
    df = get_gender_df(authors)
    download_csv(df, csv_name)

video_link = st.text_input("Enter youtube link:")
if st.button("submit"):
    # Display the entered text when the button is clicked
    run(video_link)