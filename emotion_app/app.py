import streamlit as st
import numpy as np
from keras.models import load_model
from utils.audio_utils import convert_audio_to_wav, extract_feature, label_encoder

# Title
st.set_page_config(page_title="Emotion Recognition", page_icon="🎵")
st.title("🎵 Real-Time Emotion Recognition from Audio")
st.markdown("Upload or record audio and get an instant emotion prediction using a deep learning model.")

# Load model once
@st.cache_resource
def load_emotion_model():
    return load_model("model/best_model.keras")

model = load_emotion_model()

# Upload/record audio
st.subheader("📂 Upload Audio File")
audio_file = st.file_uploader("Supported formats: wav, mp3, m4a", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')
    with st.spinner("Processing and predicting..."):
        try:
            wav_path = convert_audio_to_wav(audio_file)
            feature = extract_feature(wav_path)
            if feature is None:
                st.error("❌ Audio too short or invalid for prediction.")
            else:
                prediction = model.predict(feature)
                predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
                st.success(f"🎯 **Predicted Emotion: {predicted_label[0].capitalize()}**")
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")