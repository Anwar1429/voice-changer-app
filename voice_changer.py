import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import io
import os

# Configure Streamlit
st.set_page_config(page_title="🎤 Voice Changer", layout="wide")
st.title("🎤 Real-Time Voice Changer")

# Check for FFmpeg (for pydub)
def check_ffmpeg():
    try:
        AudioSegment.from_mp3("test.mp3").export("test_out.mp3", format="mp3")
        os.remove("test_out.mp3")
    except:
        st.warning("FFmpeg not found! Audio processing may not work properly.")
        st.info("On Streamlit Cloud, add 'ffmpeg' under 'Advanced settings' when deploying")

check_ffmpeg()

# Audio processing function
def process_audio(input_bytes, file_type, pitch=-3, speed=0.9, echo=False, deep=False):
    # Convert to AudioSegment
    audio = AudioSegment.from_file(io.BytesIO(input_bytes), format=file_type.split('/')[-1])
    
    # Convert to WAV in memory
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    
    # Process with Librosa
    y, sr = librosa.load(wav_buffer, sr=None)
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
    y_slow = librosa.effects.time_stretch(y_shifted, rate=speed)
    
    if echo:
        echo_signal = np.zeros_like(y_slow)
        echo_signal[int(0.15*sr):] = y_slow[:-int(0.15*sr)] * 0.3
        y_slow += echo_signal
    
    if deep:
        y_slow = librosa.resample(y_slow, orig_sr=sr, target_sr=int(sr*0.9))
        y_slow = librosa.util.fix_length(y_slow, len(y_slow))
    
    # Convert back to AudioSegment
    output_buffer = io.BytesIO()
    sf.write(output_buffer, y_slow, sr, format='WAV')
    output_buffer.seek(0)
    processed = AudioSegment.from_wav(output_buffer)
    
    if deep:
        processed = processed + 5  # Volume boost
        processed = processed.low_pass_filter(500)  # Bass boost
    
    return processed

# Streamlit UI
with st.sidebar:
    st.header("Settings")
    pitch = st.slider("Pitch Shift", -12, 12, -3)
    speed = st.slider("Speed", 0.5, 2.0, 0.9)
    echo = st.checkbox("Add Echo", True)
    deep = st.checkbox("Deep Voice Effect", True)

uploaded_file = st.file_uploader("Upload audio (MP3/WAV)", type=["mp3", "wav"])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.audio(uploaded_file, format=uploaded_file.type)
    
    if st.button("Process Audio"):
        with st.spinner("Transforming voice..."):
            try:
                processed = process_audio(
                    uploaded_file.read(),
                    uploaded_file.type,
                    pitch,
                    speed,
                    echo,
                    deep
                )
                
                output = io.BytesIO()
                processed.export(output, format="mp3", bitrate="192k")
                output.seek(0)
                
                with col2:
                    st.audio(output, format="audio/mp3")
                    st.download_button(
                        "Download Result",
                        data=output,
                        file_name="modified_voice.mp3",
                        mime="audio/mp3"
                    )
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Try a different file or adjust settings")
