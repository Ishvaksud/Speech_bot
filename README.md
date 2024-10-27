# Speech_bot

## Introduction 

This project is a speech bot powered by LLaMA 3.1 and whisper to get appropriate responses asked by the user in the form of speech with the help of Langchain

## Working 

1) At first a audio recorder has been used to record the audio of the user. A interface has been created using Streamlit which allows you to record the question (voice).

2) Then the recording is saved in the form of Mp3 recording.

3) We load the recording and use the distil-whisper-large model to get the speech-to-text results. This steps allows us to get the user prompt.

4) That user prompt is passed to LLaMA 3.1 using  Ollama.

5) Then the output is displayed.


## Requirements

To install the following requirements use the following command -> pip install -r requirements.txt

## How to run 

To run the following script use the following command -> streamlit run speech_bot.py


