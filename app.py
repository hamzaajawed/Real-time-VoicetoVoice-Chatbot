!pip install git+https://github.com/openai/whisper.git
!pip install torch
!pip install gTTS
!pip install gradio
!pip install groq

import os

# Set your Groq API key
os.environ['GROQ_API_KEY'] = 'Your API_KEY'



import whisper
from gtts import gTTS
import os
import gradio as gr
from groq import Groq

# Load Whisper model
whisper_model = whisper.load_model("base")

# Set up Groq client with API key (make sure to add the key to your environment)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Step 1: Transcribe Audio using Whisper
def transcribe_audio(audio_file):
    result = whisper_model.transcribe(audio_file)
    return result['text']

# Step 2: Get chatbot response from Groq LLM
def get_chat_response(message):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": message}],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Step 3: Convert the chatbot response to speech using gTTS
def convert_text_to_speech(text):
    tts = gTTS(text)
    tts.save("response.mp3")
    return "response.mp3"

# Step 4: Define chatbot pipeline integrating all components
def chatbot_pipeline(audio_file):
    # Transcribe the audio input
    transcription = transcribe_audio(audio_file)
    
    # Get response from Groq API
    response = get_chat_response(transcription)
    
    # Convert the response to speech
    audio_output = convert_text_to_speech(response)
    
    return transcription, audio_output

# Step 5: Create and launch Gradio interface
gr.Interface(
    fn=chatbot_pipeline,
    inputs=gr.Audio(type="filepath"),  # Removed source argument
    outputs=[gr.Textbox(label="Transcription"), gr.Audio(label="Response")],
    live=True
).launch()
