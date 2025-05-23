import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import os
import io

# Streamlit page config
st.set_page_config(page_title="Voice Changer", page_icon="🎤")

# Custom speed change function
def speed_change(sound, speed=1.0):
    return sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    }).set_frame_rate(sound.frame_rate)

def process_audio(uploaded_file, pitch_shift, speed_factor, echo, deep_effect):
    # Create temp files in memory
    with io.BytesIO() as temp_input:
        # Save uploaded file to memory
        temp_input.write(uploaded_file.read())
        temp_input.seek(0)
        
        try:
            # Process audio
            sound = AudioSegment.from_file(temp_input, format=uploaded_file.type.split('/')[-1])
            temp_wav = "temp_input.wav"
            sound.export(temp_wav, format="wav")
            
            y, sr = librosa.load(temp_wav, sr=None)
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
            y_slow = librosa.effects.time_stretch(y_shifted, rate=speed_factor)
            
            if echo:
                echo_length = int(0.15 * sr)
                echo_signal = np.zeros_like(y_slow)
                echo_signal[echo_length:] = y_slow[:-echo_length] * 0.3
                y_slow = y_slow + echo_signal
            
            if deep_effect:
                original_length = len(y_slow)
                y_slow = librosa.resample(y_slow, orig_sr=sr, target_sr=int(sr * 0.9))
                y_slow = librosa.util.fix_length(y_slow, size=original_length)
            
            with io.BytesIO() as temp_output:
                sf.write(temp_output, y_slow, sr, format='wav')
                temp_output.seek(0)
                processed_audio = AudioSegment.from_wav(temp_output)
                
                if deep_effect:
                    processed_audio = processed_audio + 5
                    processed_audio = processed_audio.low_pass_filter(500)
                
                output_buffer = io.BytesIO()
                processed_audio.export(output_buffer, format="mp3", bitrate="192k")
                output_buffer.seek(0)
                
                return output_buffer
                
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

# Streamlit UI
st.title("🎤 Voice Changer App")
st.write("Upload an audio file and modify it to sound like a different person!")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        pitch_shift = st.slider("Pitch Shift (semitones)", -12, 12, -3)
        speed_factor = st.slider("Speed Factor", 0.5, 2.0, 0.9)
    
    with col2:
        echo = st.checkbox("Add Echo", True)
        deep_effect = st.checkbox("Deep Voice Effect", True)
    
    if st.button("Process Audio"):
        with st.spinner("Processing your audio..."):
            try:
                output_buffer = process_audio(
                    uploaded_file, pitch_shift, speed_factor, echo, deep_effect
                )
                
                st.audio(output_buffer, format="audio/mp3")
                
                st.download_button(
                    label="Download Processed Audio",
                    data=output_buffer,
                    file_name="modified_voice.mp3",
                    mime="audio/mp3"
                )
                
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
                st.info("Make sure you've uploaded a valid audio file and try again.")