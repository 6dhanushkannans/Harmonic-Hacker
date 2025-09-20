import streamlit as st
import fitz  # PyMuPDF
import os
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from transformers import pipeline
import faiss
import tempfile

# Load IBM Watson TTS
authenticator = IAMAuthenticator(os.getenv("IBM_API_KEY"))
tts = TextToSpeechV1(authenticator=authenticator)
tts.set_service_url(os.getenv("IBM_URL"))

# HuggingFace summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Streamlit UI
st.title("🎧 EchoVerse - AI Audiobook Generator")
uploaded_file = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"])

if uploaded_file:
    text = ""
    if uploaded_file.type == "application/pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
    else:
        text = uploaded_file.read().decode("utf-8")

    st.subheader("📄 Extracted Text")
    st.write(text[:1000] + "...")

    # Summarize
    st.subheader("🧠 Summary")
    summary = summarizer(text[:1000])[0]['summary_text']
    st.write(summary)

    # Convert to audio
    st.subheader("🔊 Audiobook Preview")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
        response = tts.synthesize(summary, voice='en-US_AllisonV3Voice', accept='audio/mp3').get_result()
        audio_file.write(response.content)
        st.audio(audio_file.name, format='audio/mp3')
