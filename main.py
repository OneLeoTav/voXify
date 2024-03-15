#0. Libraries and dependencies
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

import matplotlib.pyplot as plt

import torch
import io

import torchaudio
from torchaudio.transforms import Resample
import pyaudio

import asyncio
import ffmpeg
import time

import base64
from typing import Optional

from utils.utilities import (
    load_speech_to_text_model,
    fetch_audio_from_YT,
    generate_video_transcript,
    extract_file_name,
    click_button,
    generate_pdf,
    generate_word_transcript,
    plot_soundwave,
    load_speech_transcriptor,
    upsample_audio,
    create_plot_dataframe,
    start_recording,
    stop_recording,
)

def main():

    async def send():
        while st.session_state['run']:
            try:
                data = stream.read(FRAMES_PER_BUFFER)
                frames.append(data)
                if len(frames) >= (RATE * RECORD_SECONDS) / FRAMES_PER_BUFFER:
                    # Return frames when a chunk is ready
                    result_frames = frames.copy()
                    frames.clear()
                    return result_frames

                await asyncio.sleep(0.01)

            except Exception as e:
                print(e)
                break

        return None

    async def receive(frames, accumulated_waveform=None, total_elapsed_time=0.0, idx=0, chunk_size: Optional[int] = None):
        # chunk_size = 4096
        if chunk_size:
            for chunk_start in range(0, len(frames), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(frames))
                chunk = frames[chunk_start:chunk_end]
                output_path = upsample_audio(chunk)
        else:
        # chunk_frame
            output_path = upsample_audio(frames)
            result = speech_transcriptor(output_path)['text']
            waveform, sample_rate = torchaudio.load(output_path, normalize=True)

            # Initialize or accumulate the entire soundwave
        if accumulated_waveform is None:
            accumulated_waveform = waveform
        else:
            accumulated_waveform = torch.cat([accumulated_waveform, waveform], dim=1)
        
        current_elapsed_time = torch.arange(0, accumulated_waveform.size(1)) / sample_rate

        # Update the existing plot
        idx+=1 #idx is a trick to force @st.cache_data to update at each pass trhough the loop
        df = create_plot_dataframe(total_elapsed_time+current_elapsed_time, accumulated_waveform, idx)
        
        soundwave = plot_soundwave(df)
        chart_placeholder.pyplot(soundwave, use_container_width=False)
        # chart_placeholder.line_chart(df, x="Seconds", y="Amplitude")

        # Print the result
        speech_placeholder.write(result)
        st.session_state['transcript'] += result

        # Update total elapsed time
        total_elapsed_time += waveform.size(1) / sample_rate
        return accumulated_waveform, total_elapsed_time, idx

    async def send_receive():
        accumulated_waveform = None
        total_elapsed_time = 0.0
        idx = 0
        try:
            while st.session_state['run']:
                # Process the received frames
                frames = await send()
                if frames:
                        accumulated_waveform, total_elapsed_time, idx = await receive(
                            frames,
                            accumulated_waveform,
                            total_elapsed_time,
                            idx,
                        )

        except asyncio.CancelledError:
            pass

    async def live_speech_recognition():
        try:
            await asyncio.gather(send_receive())  
        except Exception as e:
            print(f"Error: {e}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('./layout/styles.css') as f:
        css = f.read()

    st.markdown(
        f"""<style>{css}</style>""",
        unsafe_allow_html=True
    )

    with open('./layout/index.html') as f:
        html = f.read()

    st.markdown(
        f"""<body>{html}</body>""",
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    col1, col2, col3 = st.columns(3)

    # Three buttons aligned horizontally
    with col1:
        # Button for handling YouTube video
        youtube_button = st.button(
            "Transcribe YT Video 🎬",
            key='youtube_button',
            on_click=click_button,
            args=[1],
        )

    with col2:
        # Button for uploading an audio file
        audio_file_button = st.button(
            "Upload Audio File 📤",
            key='audio_file_button',
            on_click=click_button,
            args=[2],
        )

    with col3:
        speech_recording_button = st.button(
            "Record Speech 💬",
            key='speech_recording_button',
            on_click=click_button,
            args=[3],
        )

    if 'clicked' not in st.session_state:
        st.session_state.clicked = {
            1: False,
            2: False,
            3: False,
        }

    if 'transcript' not in st.session_state:
        st.session_state['transcript'] = ""
        st.session_state['on_screen_transcript'] = ""
        st.session_state['run'] = False

    # Create a container below the buttons
    display_interface_container = st.container()

    ### Handling YouTube URLs
    if st.session_state.clicked.get(1, False):

        with display_interface_container:

            youtube_url = st.text_input(
                'Enter the YouTube video URL',
                key='youtube_url',
            )

            if youtube_url:
                with st.spinner('Generating transcript...'):
                    audio_stream, audio_name = fetch_audio_from_YT(str(youtube_url))
                    # Check whether the URL is valid
                    if audio_stream is not None:
                        speech_to_text_model = load_speech_to_text_model(device=device)
                        transcript = generate_video_transcript(audio_stream, speech_to_text_model, device)

                        # Enables the user to play the original audio file
                        st.audio(audio_stream)

                        # Displays transcript
                        with st.expander("Your Youtube video transcript below"):
                            on_screen_transcript = st.write(transcript)
                    
                        download_option = st.radio(
                                            "Set your download preference 👇",
                                            ["PDF", "Word"],
                                            key="radio_layout",
                                            horizontal=True,
                                        )

                        # If URL is not valid or failed to fecth the audio file
                        if download_option == "PDF":
                            with st.spinner("Generating PDF..."):
                                pdf = generate_pdf(transcript, audio_name)
                                pdf_bytes = pdf.output(dest='S').encode('latin1')

                                download_button_html = f"""
                                    <a download="{audio_name}_transcript.pdf" href="data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}" 
                                    style="text-decoration:none; color:white; padding: 10px 20px; background-color: #008CBA; 
                                    border-radius: 5px; cursor: pointer; text-align: center; display: inline-block;">
                                    Download PDF 💾
                                    </a>
                                    """
                                                            
                        else:
                            with st.spinner("Generating Word..."):
                                doc = generate_word_transcript(transcript, audio_name)
                                doc_bytes = io.BytesIO()
                                doc.save(doc_bytes)

                                # Create a download link for Word
                                download_button_html = f"""
                                <a download="{audio_name}_transcript.docx" href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{base64.b64encode(doc_bytes.getvalue()).decode()}" 
                                style="text-decoration:none; color:white; padding: 10px 20px; background-color: #008CBA; 
                                border-radius: 5px; cursor: pointer; text-align: center; display: inline-block;">
                                Download Word 💾
                                </a>
                                """
                        
                        st.markdown(download_button_html, unsafe_allow_html=True)
                    else:
                        st.warning(
                        """Failed to fetch. 
                            Please try again or enter another URL.""",
                            icon="⚠️"
                        )

    ### Handling audio files
    elif st.session_state.clicked.get(2, False):

        with display_interface_container:
            audio_file = st.file_uploader(
            "Please upload an audio file",
            type=["wav", "mp3", "m4a", "mp4"]
            )

            if audio_file:
                # audio_bytes = audio_file.read()
                # audio = whisper.load_audio(audio_bytes)
                speech_to_text_model = load_speech_to_text_model(device=device)

                #If a single file (i.e. accept_multiple_files=False),
                #the filename can be retrieved by using the '.name' attribute on the returned UploadedFile object
                #Otherwise throws: 'TypeError: expected np.ndarray (got UplodedFile)'

                transcript = generate_video_transcript(audio_file.name, speech_to_text_model, device)

                # Enables the user to play the original audio file
                st.audio(audio_file)

                # Displays transcript
                with st.expander("Show transcript"):
                    on_screen_transcript = st.write(transcript)
                
                download_option = st.radio(
                    "Set your download preference 👇",
                    ["PDF", "Word"],
                    key="radio_layout",
                    horizontal=True,
                )
                
                audio_file_name = extract_file_name(audio_file.name)
                if download_option == "PDF":
                    with st.spinner("Generating PDF..."):
                        pdf = generate_pdf(transcript, audio_file_name)
                        pdf_bytes = pdf.output(dest='S').encode('latin1')


                        download_button_html = f"""
                            <a download="{audio_file_name}_transcript.pdf" href="data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}" 
                            style="text-decoration:none; color:white; padding: 10px 20px; background-color: #008CBA; 
                            border-radius: 5px; cursor: pointer; text-align: center; display: inline-block;">
                            Download PDF 💾
                            </a>
                            """
                                                    
                else:
                    with st.spinner("Generating Word..."):

                        doc = generate_word_transcript(transcript, audio_file_name)
                        doc_bytes = io.BytesIO()
                        doc.save(doc_bytes)

                        # Create a download link for Word
                        download_button_html = f"""
                        <a download="{audio_file_name}_transcript.docx" href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{base64.b64encode(doc_bytes.getvalue()).decode()}" 
                        style="text-decoration:none; color:white; padding: 10px 20px; background-color: #008CBA; 
                        border-radius: 5px; cursor: pointer; text-align: center; display: inline-block;">
                        Download Word 💾
                        </a>
                        """
                st.markdown(download_button_html, unsafe_allow_html=True)


    elif st.session_state.clicked.get(3, False):
        speech_transcriptor = load_speech_transcriptor()

        FRAMES_PER_BUFFER = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 48000
        RECORD_SECONDS = 3 
        frames = []
        p = pyaudio.PyAudio()

        # starts recording
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER
        )

        st.text("")
        st.text("")
        recording_buttons_container = st.container()
        st.divider()
        chart_placeholder = st.empty()
        st.text("")
        st.text("")
        speech_placeholder = st.container()
        message_placeholder = st.empty()

        with recording_buttons_container:
            col1_record, col2_record = st.columns(2)

            with col1_record:
                # Button to START recording
                start_recording_button = st.button(
                    "Start Recording 🎤",
                    key='start_recording_button',
                    on_click=start_recording,
                )

            with col2_record:
                # Button to STOP recording
                stop_recording_button = st.button(
                    "Stop Recording ⛔️",
                    key='stop_recording_button',
                    on_click=stop_recording,
                )

        if start_recording_button:
            with st.spinner("Recording . . ."):
                asyncio.run(live_speech_recognition())

        if stop_recording_button:
            message_placeholder.write("Recording has been stopped")

        with st.expander("Your speech transcript below 👇"):
            on_screen_transcript = st.session_state['on_screen_transcript']
            st.write(on_screen_transcript)

        download_option = st.radio(
            "Set your download preference 👇",
            ["PDF", "Word"],
            key="radio_layout",
            horizontal=True,
        )

        file_name = 'Live Speech'

        if download_option == "PDF":
            with st.spinner("Generating PDF..."):
                pdf = generate_pdf(on_screen_transcript, file_name)
                pdf_bytes = pdf.output(dest='S').encode('latin1')

                download_button_html = f"""
                    <a download="{file_name}_transcript.pdf" href="data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}" 
                    style="text-decoration:none; color:white; padding: 10px 20px; background-color: #008CBA; 
                    border-radius: 5px; cursor: pointer; text-align: center; display: inline-block;">
                    Download PDF 💾
                    </a>
                    """
        else:
            with st.spinner("Generating Word..."):
                doc = generate_word_transcript(on_screen_transcript, file_name)
                doc_bytes = io.BytesIO()
                doc.save(doc_bytes)

                # Create a download link for Word
                download_button_html = f"""
                <a download="{file_name}_transcript.docx" href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{base64.b64encode(doc_bytes.getvalue()).decode()}" 
                style="text-decoration:none; color:white; padding: 10px 20px; background-color: #008CBA; 
                border-radius: 5px; cursor: pointer; text-align: center; display: inline-block;">
                Download Word 💾
                </a>
                """
        st.markdown(download_button_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()