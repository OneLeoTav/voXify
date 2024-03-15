# voXify

<img src="media/voxify_homepage_gif.gif" alt="Homepage GIF" width="550" height="320">

## Introduction
Welcome to the **voXify** repository, a web application powered by Streamlit. **voXify** provides a comprehensive suite of Speech-to-Text functionalities, facilitating the generation of transcripts from diverse audio sources.allowing users to obtain transcripts from various audio sources. For tasks encompassing the extraction of text from a YouTube video, uploading an audio file, or capturing live speech, **voXify**  offers a seamless solution to generate the corresponding transcript and visualize it onto the plateform. Your transcript can finally be conveniently downloaded in either PDF or Word format.

## Main Features
**YouTube Transcription 🎬**\
**voXify** offers the ability to convert any YouTube video into text, leveraging `pytube` library as well as [Whisper](https://github.com/openai/whisper), OpenAI's open-source speech-to-text model.

<img src="media/voXify_YT_gif.gif" alt="YT GIF" width="300" height="160">


**Audio File Transcription 📤**\
Transcribe audio files of various formats (WAV, MP3, M4A, MP4, MPEG4) into text by simply uploading them through the file uploader widget provided for this purpose.

<img src="media/voXifY_Audio_gif.gif" alt="Audio GIF" width="300" height="160">

**Real-time Microphone Transcription 💬**\
Obtain live transcriptions directly from your microphone. **voXify** captures speech in real-time and provides a simultaneous display of both the transcript and the relate soundwave.

**Flexible Output Formats 🚀**\
Download transcripts in the desired format, choosing between either Word document or PDF through custom buttons, depending on user preference. This feature is available for all three functionalities mentioned hereinabove.

**Note ⚠️**
- As voXify provides an diverse spectrum of speech-to-text fucntionnalities, the occurrence of bugs is deemed to be frequent in the near future. Hence, with a view to ensuring the robustness of the application, tests are yet to be implemented (WIP ⚙️).
- The real-time text-to-speech feature leverages the `asyncio` library to facilitate the sequential handling of temporary audio files generated from microphone recordings and their subsequent processing through the speech-to-text pipeline. However, this approach turns out to be relatively slow and error-prone. Indeed, transcription accuracy tends to be (way) higher when running in a Jupyter Notebook environment. I infer that this is because the Streamlit application is tasked with continuously rendering both the current transcript and waveform, all while managing concurrent processes in the background. **To mitigate this issue, future commits may involve exploring multi-threading as an alternative to asynchronous programming.**
- Furthermore, the current real-time text-to-speech implementation does not generate a good looking transcript (i.e., only capital letters, no punctuation, etc.). To improve the user experience, potential strategies include **upgrading to a higher-performance model or implementing a layer of formatting to enhance readability**.
- Additionally, I am exploring the possibility of integrating a translation to further enhance the application's exhaustiveness.

## Overview of the Repository
- `layout/`: This directory contains the HTML template file (`index.html`) and the CSS styling file (`styles.css`) used for the design of the user interface.
- `.streamlit/`: Comprises the custom `config.toml` file that sets tailored configuration options to optimize the functioning of the application.
- `utils/`: This directory houses essential Python functions and classes that are utilized throughout the application for various tasks such as data processing, interaction with APIs, or other utility functions.
- `main.py`: This Python script serves as the main entry point for the application, containing the core functionality and orchestrating the overall layout and behavior of the application. It handles routing, data processing, user interactions, and integration with external services or libraries.

## Installation
First and foremost, ensure you have Python >= 3.8 installed, then:
1. Clone the project repository: `git clone https://github.com/OneLeoTav/voXify.git`
2. Navigate to the root of the directory: `cd voXify/`
3. Install the required Python packages: `pip install -r requirements.txt`
4. Run the Streamlit application: `streamlit run main.py`

**Important Notice**: To enable live speech-to-text functionality, it is necessary to locally have the <i>ffmpeg.exe</i> file located at the root of the directory. You can retrieve the aforementioned file [here](https://ffmpeg.org/download.html).
