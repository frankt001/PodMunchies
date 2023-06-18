import os
import re
import requests
import time
import xml.etree.ElementTree as ET
import openai
from pydub import AudioSegment
from tqdm import tqdm
from dotenv import load_dotenv
import yt_dlp
from pytube import YouTube
import time

load_dotenv()

def with_retries(func, max_retries=5, *args, **kwargs):
    for i in range(max_retries):
        try:
            return func(*args, **kwargs)  # Call your function
        except openai.error.RateLimitError as e:
            print(f"Rate limit error: {str(e)}, retrying in {2 ** i} seconds...")
            time.sleep(2 ** i)  # Exponential backoff
    raise Exception("Failed after max retries")

def get_youtube_video_title(youtube_url):
    yt = YouTube(youtube_url)
    return yt.title

# download audio from YouTube
def download_youtube_audio(youtube_url, save_directory, episode_title=None, download=True):
    if not download:
        return None, episode_title

    print(f"Downloading episode: {episode_title}")

    local_filename = os.path.join(save_directory, f"{episode_title}")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': local_filename,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    print(f"Download complete: {local_filename}")
    return local_filename, episode_title

#Transcribe audio segments using Gambit Engine
def transcribe_audio_segments(file_path: str):
    headers = {'Authorization': 'ge_ec58094b1ad2ad.d547f6621a84c617e23c0f90dc988a41656f08042bd28a2fcaa76f5540054361'}
    url = 'https://api.gambitengine.com'
    r = requests.post(f'{url}/v1/scribe/transcriptions', headers=headers)
    r.raise_for_status()
    transcription = r.json()['transcription']
    upload_request = r.json()['upload_request']
    r = requests.post(upload_request['url'], data=upload_request['fields'], files={'file': open(file_path, 'rb')})

    for x in range(10):
        r = requests.get(f'{url}/v1/scribe/transcriptions/{transcription["transcription_id"]}', headers=headers)
        r.raise_for_status()
        transcription = r.json()
        if transcription['transcribed'] == True:
            break
        time.sleep(5)

    return [transcription['text']]

#Save transcripts to file
def save_transcripts(transcripts, save_directory, filename):
    transcript_path = os.path.join(save_directory, filename)

    with open(transcript_path, "w") as f:
        for transcript in transcripts:
            f.write(transcript)
            f.write("\n\n")

def split_text_into_chunks(text, max_tokens=4090):
    chunks = []
    while len(text) > 0:
        tokens = text[:max_tokens]
        last_space = tokens.rfind(' ')
        if last_space == -1:
            last_space = max_tokens
        chunks.append(text[:last_space].strip())
        text = text[last_space:].strip()
    print(f"Text split into {len(chunks)} chunks.")
    return chunks

def summarize_chunk(chunk):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text.    "},
            {"role": "user", "content": f"Summarize the transcript into bullet points. The bullet points should be informative detailed description of key points, topics and insights that the guests talked about.Use only content from the transcript. Do not add any additional information: {chunk}"}
        ],
        temperature=0.5,
    )
    summary = response.choices[0].message.content.strip()
    return summary

def summarize_large_text(transcripts):
    max_tokens = 4090  # Reserve tokens for instructions and conversation context
    summaries = []
    
    for text in transcripts:
        chunks = split_text_into_chunks(text, max_tokens)
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i+1}/{len(chunks)}")
            summary = summarize_chunk(chunk)
            summaries.append(summary)
    
    print("Summaries aggregated.")
    return ' '.join(summaries)


def generate_final_summary(summary_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Your name is PodMunchie Sensei. Imagine you have just completed recording an amazing podcast episode, and as the host of the podcast, you are now looking to create a captivating first-person narrative summary of your episode to share with your audience on social media. Weave the key takeaways and highlights of your episode into an engaging, informative, and succinct summary. Your goal is to entice your audience to listen to the full episode. Think about how you can turn the most important parts of your episode into a compelling and informative story that captures the essence of your podcast. Remember, the goal of your narrative summary is to get your audience excited about your podcast and make them eager to tune in."
            },
            {
                "role": "user",
                "content": f"So, let your creativity shine and craft a first-person narrative summary that is as captivating as your episode itself! Please Also follow the rules below: - Give more content, no fluff, and no need for buzz words. - Ensure to pack the summary with solid content and information that the listeners can learn from. - Say greeting first. - Mention in the begining this is a summary - The summary should be around the topics. - The length of the summary should be at least 800 words: {summary_text}"
            },
        ],
        temperature=0.6,
    )

    return response.choices[0].message.content.strip()

def show_note_summary(summarized_bullet_points):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes text."
            },
            {
                "role": "user",
                "content": f"Summarize the transcript in a clear and concise manner that. Chapters should be meaningful length and not too short. To format your markdown file, follow this structure: # [Descriptive Title]  <overview of the video> - Use bullet points to provide a detailed description of key points and insights. Title for the topic - Use bullet points to provide a detailed description of key points and insights. Repeat the above structure as necessary, and use subheadings to organize your notes. Formatting Tips: * Do not make the chapters too short, ensure that each section has at least 3-5 bullet points * Use bullet points to describe important steps and insights, being as comprehensive as possible. Summary Tips: * Use only content from the transcript. Do not add any additional information. * Make a new line after and before each bullet point {summarized_bullet_points}"
            },
        ],
        temperature=0.6,
    )

    return response.choices[0].message.content.strip()

def text_to_speech(text, output_file):
    CHUNK_SIZE = 1024
    url = f"https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": os.getenv('XI_API_KEY')  
    }

    data = {
        "text": text,
        "voice_settings": {
            "stability": 0.75,
            "similarity_boost": 0.75
        }
    }

    response = requests.post(url, json=data, headers=headers)

    with open(output_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)

    print(f"Text-to-speech saved to: {output_file}")



if __name__ == "__main__":
    openai.api_key = os.environ["OPENAI_API_KEY"]

    youtube_video_url = os.environ["YOUTUBE_VIDEO_URL"]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_directory = os.path.join(current_dir, "podsum_rss")
    os.makedirs(save_directory, exist_ok=True)


   # Fetch the video title and use it as the filename
    episode_title = get_youtube_video_title(youtube_video_url)
    # Replace characters in the title that are not valid in filenames
    episode_title = re.sub(r'[\\/*?:"<>|]', "", episode_title)

    print(episode_title)

    #Donwload the episode
    mp3_path = os.path.join(save_directory, f"{episode_title}.mp3")
    if not os.path.exists(mp3_path):
        download_youtube_audio(youtube_video_url, save_directory, episode_title, download=True)
    else:
        print(f"MP3 file already exists: {mp3_path}")
    
    local_filename = mp3_path
    print(local_filename)
    
    # Define the file paths
    mp3_path = os.path.join(save_directory, f"{episode_title}.mp3")
    transcript_path = os.path.join(save_directory, f"transcript_{episode_title}.txt")
    topics_path = os.path.join(save_directory, f"topics_{episode_title}.txt")
    final_summary_path = os.path.join(save_directory, f"final_summary_{episode_title}.txt")
    show_note_path = os.path.join(save_directory, f"show_note_{episode_title}.txt")

    # Transcribe the audio segments
    if not os.path.exists(transcript_path):
        transcripts = transcribe_audio_segments(local_filename)
        # Save the transcripts
        save_transcripts(transcripts, save_directory, f"transcript_{episode_title}.txt")
        print("Transcripts saved.")
    else:
        print(f"Transcript file already exists: {transcript_path}")
        with open(transcript_path, "r") as f:
            transcripts = f.read().split("\n\n")

    # Concatenate the transcripts
    full_transcript = "\n\n".join(transcripts)

    # Summarize the transcript into bullet points
    if not os.path.exists(topics_path):
        summarized_bullet_points = summarize_large_text(full_transcript)
        print("Bullet Points Created.")

        # Save the summarized text
        save_transcripts([summarized_bullet_points], save_directory, f"topics_{episode_title}.txt")
        print("Bullet Points saved.")
    else:
        print(f"Bullet Points file already exists: {topics_path}")

    # Summarize the bullet points into final first person summary
    if not os.path.exists(final_summary_path):
        with open(topics_path, "r") as f:
            summary_text = f.read()
        final_summary = generate_final_summary(summary_text)
        print("Final summary generated.")

        with open(final_summary_path, "w") as outfile:
            outfile.write(final_summary)
        print("Final summary saved.")
    else:
        print(f"Final summary file already exists: {final_summary_path}")


    # Read the final summary text
    with open(final_summary_path, 'r') as f:
        final_summary_text = f.read()
        
    # Convert the final summary text to speech and save it as an MP3 file
    tts_output_path = os.path.join(save_directory, f"tts_final_summary_{episode_title}.mp3")
    print("Text-to-speech conversion in progress...")
    text_to_speech(final_summary_text, tts_output_path)
