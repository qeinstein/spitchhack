from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
import io
import os
from spitch import Spitch
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
app = FastAPI()

# Initialize Spitch client
spitch_client = Spitch(api_key=os.getenv("SPITCH_API_KEY"))

# Initialize OpenRouter client for Mistral
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


MODEL = "mistralai/mistral-7b-instruct:free"

# Language-to-voice mapping (adjust based on Spitch's available voices)
VOICE_MAP = {
    "en": "jude",  # English voice
    "ha": "aliyu",  # Hausa voice (example, replace with actual Spitch voice)
    "yo": "femi",
    "ig": "obinna"
}
DEFAULT_VOICE = "kani"  # Fallback voice

@app.post("/start_call")
async def start_call():
    """Asks user for preferred language in English."""
    text = "What language do you want to speak in?"
    audio_response = spitch_client.speech.generate(
        text=text,
        language="en",
        voice=DEFAULT_VOICE
    )
    return StreamingResponse(io.BytesIO(audio_response.read()), media_type="audio/wav")

@app.post("/process_response")
async def process_response(
    audio: UploadFile,
    language: str = Form(None)
):
    """Processes user audio response."""
    audio_bytes = await audio.read()
    audio_io = io.BytesIO(audio_bytes)

    if language is not None:
        # Ongoing conversation: Transcribe in specified language
        transcription = spitch_client.speech.transcribe(
            content=audio_io,
            language=language
        )
        transcribed_text = transcription.text

        # Translate to English
        english_text = spitch_client.translation.translate(
            text=transcribed_text,
            source_language=language,
            target_language="en"
        )

        # Process with Mistral
        mistral_response = openrouter_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": english_text}
            ]
        )
        english_response = mistral_response.choices[0].message.content

        # Translate back to user's language
        translated_response = spitch_client.translation.translate(
            text=english_response,
            source_language="en",
            target_language=language
        )

        # Generate TTS with language-specific voice
        voice = VOICE_MAP.get(language, DEFAULT_VOICE)
        tts_audio = spitch_client.speech.generate(
            text=translated_response,
            language=language,
            voice=voice
        )
        return {
            "response_text": translated_response,
            "audio": StreamingResponse(io.BytesIO(tts_audio.read()), media_type="audio/wav")
        }

    else:
        # Initial response: Transcribe in English
        transcription = spitch_client.speech.transcribe(
            content=audio_io,
            language="en"
        )
        transcribed_text = transcription.text

        # Detect language with Mistral
        detection_prompt = f"The user said: '{transcribed_text}'. What language do they want to speak in? Respond with the ISO 639-1 code (e.g., 'ha' for Hausa)."
        mistral_response = openrouter_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a language detector."},
                {"role": "user", "content": detection_prompt}
            ]
        )
        detected_lang = mistral_response.choices[0].message.content.strip().lower()

        # Generate confirmation response
        english_response = f"Okay, we will speak in {detected_lang}."
        translated_response = spitch_client.translation.translate(
            text=english_response,
            source_language="en",
            target_language=detected_lang
        )

        # Generate TTS with language-specific voice
        voice = VOICE_MAP.get(detected_lang, DEFAULT_VOICE)
        tts_audio = spitch_client.speech.generate(
            text=translated_response,
            language=detected_lang,
            voice=voice
        )
        return {
            "detected_language": detected_lang,
            "response_text": translated_response,
            "audio": StreamingResponse(io.BytesIO(tts_audio.read()), media_type="audio/wav")
        }