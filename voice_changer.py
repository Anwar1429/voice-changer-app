import streamlit as st
import asyncio
import edge_tts
import os

# ----------------- Config ----------------- #
st.set_page_config(
    page_title="Vocalize AI",
    page_icon="🎙️",
    layout="centered"
)

# ------------- Custom Style ---------------- #
st.markdown("""
    <style>
    .stTextArea textarea {
        font-size: 16px;
        line-height: 1.6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5em 1em;
    }
    footer {
        visibility: hidden;
    }
    .footer {
        text-align: center;
        font-size: 0.9em;
        margin-top: 3em;
        color: #888;
    }
    </style>
""", unsafe_allow_html=True)

# ----------- Constants ---------------- #
max_chars = 2000
voice_map = {
    "English": {
        "Male": "en-GB-RyanNeural",
        "Female": "en-GB-SoniaNeural"
    },
    "Arabic": {
        "Male": "ar-SA-HamedNeural",
        "Female": "ar-SA-ZariyahNeural"
    }
}

# ----------- Async TTS Function -------- #
async def generate_tts(text, voice, file_path):
    try:
        tts = edge_tts.Communicate(text, voice=voice)
        await tts.save(file_path)
        return True, None
    except Exception as e:
        return False, str(e)

# -------------- App Title ----------------- #
st.title("🎙️ Vocalize AI")
st.markdown("##### Convert text to realistic speech in English or Arabic using AI voices.")
st.markdown("Easily listen to your message and download it as MP3. Great for education, communication, accessibility, or content creation!")

# ----------- Input Section --------------- #
col1, col2 = st.columns(2)
with col1:
    language = st.selectbox("🌍 Language", ["English", "Arabic"])
with col2:
    gender = st.radio("👤 Voice", ["Male", "Female"], horizontal=True)

user_text = st.text_area("📝 Enter your text below:", height=150, max_chars=max_chars)
chars_used = len(user_text)
st.write(f"**Characters used:** `{chars_used}/{max_chars}`")

# ----------- Button + Output -------------- #
if chars_used > max_chars:
    st.error(f"Please limit your input to {max_chars} characters.")
elif st.button("🔊 Generate and Play Audio"):
    if user_text.strip() == "":
        st.warning("Please enter some text to convert.")
    else:
        selected_voice = voice_map[language][gender]
        output_file = "user_tts_demo.mp3"
        success, error_msg = asyncio.run(generate_tts(user_text, selected_voice, output_file))

        if success:
            st.success("✅ Audio generated successfully!")
            with open(output_file, "rb") as f:
                audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/mp3")
            st.download_button("⬇️ Download MP3", audio_bytes, file_name="tts_audio.mp3", mime="audio/mp3")
        else:
            st.error(f"❌ Failed to generate audio: {error_msg}")


