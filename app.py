import streamlit as st
import os
import openai
import re
from dotenv import load_dotenv
from youtube_downloader import download_youtube_audio, split_audio, transcribe_audio_segments, save_transcripts, get_youtube_video_title, summarize_large_text, generate_final_summary, text_to_speech
import imageio_ffmpeg as ffmpeg

print(ffmpeg.get_ffmpeg_version())
print(ffmpeg.get_ffmpeg_exe())

# Load environment variables
load_dotenv()

def app():

    # Set up Streamlit layout
    st.set_page_config(page_title="Podcast Summary Generator", layout="wide")
    st.title("Podcast Summary Generator")

    # Set up side bar
    st.sidebar.title("Controls")
    youtube_video_url = st.sidebar.text_input("Enter YouTube Video URL")
    transcribe_button = st.sidebar.button("Generate Summary")

    if transcribe_button:
        if youtube_video_url != "":

            openai.api_key = os.environ["OPENAI_API_KEY"]

            # Fetch the video title and use it as the filename
            episode_title = get_youtube_video_title(youtube_video_url)
            # Replace characters in the title that are not valid in filenames
            episode_title = re.sub(r'[\\/*?:"<>|]', "", episode_title)

            # Current Directory
            current_dir = os.getcwd()
            save_directory = os.path.join(current_dir, "youtube_summary_streamlit")
            os.makedirs(save_directory, exist_ok=True)

            # Download the episode
            mp3_path = os.path.join(save_directory, f"{episode_title}.mp3")
            download_youtube_audio(youtube_video_url, save_directory, episode_title)

            # Split the episode into 20-minute segments
            segments = split_audio(mp3_path)

            # Transcribe the audio segments
            transcripts = transcribe_audio_segments(segments)

            # Save the transcripts
            save_transcripts(transcripts, save_directory, f"transcript_{episode_title}.txt")

            # Concatenate the transcripts
            full_transcript = "\n\n".join(transcripts)

            # Summarize the transcript into bullet points
            summarized_bullet_points = summarize_large_text(full_transcript)

            # Save the summarized text
            save_transcripts([summarized_bullet_points], save_directory, f"topics_{episode_title}.txt")

            # Summarize the bullet points into final first person summary
            final_summary = generate_final_summary(summarized_bullet_points)

            # Save the final summary
            with open(os.path.join(save_directory, f"final_summary_{episode_title}.txt"), "w") as outfile:
                outfile.write(final_summary)

            # Convert the final summary text to speech and save it as an MP3 file
            tts_output_path = os.path.join(save_directory, f"tts_final_summary_{episode_title}.mp3")
            text_to_speech(final_summary, tts_output_path)

            # Display the final summary
            st.header("Final Summary")
            st.write(final_summary)
            st.audio(tts_output_path)

        else:
            st.sidebar.warning("Please enter a YouTube Video URL.")
    

if __name__ == "__main__":
    app()
