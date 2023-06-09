import streamlit as st
import os
import openai
import re
from dotenv import load_dotenv
from youtube_downloader import download_youtube_audio, transcribe_audio_segments, save_transcripts, get_youtube_video_title, summarize_large_text, generate_final_summary, text_to_speech, show_note_summary, with_retries
import static_ffmpeg
# ffmpeg installed on first call to add_paths(), threadsafe.
static_ffmpeg.add_paths()  

# Load environment variables
load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

def app():

    # Set up Streamlit layout
    st.set_page_config(page_title="Podcast Summary Generator", layout="wide")
    st.title("Podcast Summary Generator")

    # Set up side bar
    st.sidebar.title("Welcom to PodMunchies")
    youtube_video_url = st.sidebar.text_input("Enter YouTube Video URL")
    transcribe_button = st.sidebar.button("Generate Summary")

    if transcribe_button:
        if youtube_video_url != "":
            
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

            # Transcribe the audio
            full_transcript = transcribe_audio_segments(mp3_path)

            # Save the transcripts
            save_transcripts(full_transcript, save_directory, f"transcript_{episode_title}.txt")


            # Summarize the transcript into bullet points
            summarized_bullet_points = with_retries(summarize_large_text, 5, full_transcript)

            # Save the summarized text
            save_transcripts([summarized_bullet_points], save_directory, f"topics_{episode_title}.txt")

            # Summarize the bullet points into final first person summary
            final_summary = generate_final_summary(summarized_bullet_points)

            # Save the final summary
            with open(os.path.join(save_directory, f"final_summary_{episode_title}.txt"), "w") as outfile:
                outfile.write(final_summary)

            # Summarize the bullet points into final first person summary
            show_note = show_note_summary(summarized_bullet_points)

            # Save the Show Note
            with open(os.path.join(save_directory, f"show_note_{episode_title}.txt"), "w") as outfile:
                outfile.write(show_note)

            # Convert the final summary text to speech and save it as an MP3 file
            tts_output_path = os.path.join(save_directory, f"tts_final_summary_{episode_title}.mp3")
            text_to_speech(final_summary, tts_output_path)

            # Display the final summary
            st.header("Final Summary")
            st.write(show_note)
            st.audio(tts_output_path)

        else:
            st.sidebar.warning("Please enter a YouTube Video URL:")
    

if __name__ == "__main__":
    app()
