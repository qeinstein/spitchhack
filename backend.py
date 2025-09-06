from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
import io
import os
import base64
import uuid
import tempfile
import glob
import time
from spitch import Spitch
from openai import OpenAI
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse, Gather
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Ensure static directory exists
STATIC_DIR = "static"
try:
    os.makedirs(STATIC_DIR, exist_ok=True)
    logger.info(f"Created static directory: {STATIC_DIR}")
except Exception as e:
    logger.error(f"Failed to create static directory: {e}")
    raise RuntimeError(f"Cannot create static directory: {e}")

# Mount static directory
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    logger.error(f"Static directory {STATIC_DIR} does not exist")
    raise RuntimeError(f"Cannot mount static directory: {STATIC_DIR}")

load_dotenv()
try:
    spitch_client = Spitch(api_key=os.getenv("SPITCH_API_KEY"))
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
except Exception as e:
    logger.error(f"Failed to initialize API clients: {e}")
    raise RuntimeError(f"API initialization failed: {e}")

MODEL = "mistralai/mistral-7b-instruct:free"

VOICE_MAP = {
    "en": "jude",
    "ha": "aliyu",  # Replace with actual Spitch Hausa voice
    "yo": "femi",
    "ig": "obinna"
}
DEFAULT_VOICE = "jude"

def cleanup_static_files(max_age_seconds=3600):
    """Clean up old audio files."""
    try:
        for file in glob.glob(f"{STATIC_DIR}/*.wav"):
            if os.path.getmtime(file) < time.time() - max_age_seconds:
                os.remove(file)
                logger.info(f"Deleted old file: {file}")
    except Exception as e:
        logger.error(f"Failed to clean up static files: {e}")

@app.post("/start_call")
async def start_call():
    """Asks user for preferred language and allows interruption."""
    try:
        text = "What language do you want to speak in?"
        audio_response = spitch_client.speech.generate(
            text=text,
            language="en",
            voice=DEFAULT_VOICE
        )
        audio_bytes = audio_response.read()

        with tempfile.NamedTemporaryFile(suffix=".wav", dir=STATIC_DIR, delete=False) as temp_file:
            temp_file.write(audio_bytes)
            audio_filename = f"static/{os.path.basename(temp_file.name)}"

        twiml = VoiceResponse()
        gather = Gather(
            input="speech",
            action="/process_response",
            method="POST",
            speechTimeout="auto",
            timeout=10
        )
        gather.play(url=f"https://spitchhack.onrender.com/{audio_filename}")
        twiml.append(gather)
        twiml.say("Sorry, I didn't catch that. Please try again.", voice="Polly.Joanna")
        twiml.redirect("/start_call", method="POST")
        cleanup_static_files()
        return Response(content=str(twiml), media_type="application/xml")
    except Exception as e:
        logger.error(f"Error in start_call: {e}")
        raise HTTPException(status_code=500, detail=f"Error in start_call: {str(e)}")

@app.post("/process_response")
async def process_response(
    audio: UploadFile = None,
    language: str = Form(None),
    RecordingUrl: str = Form(None),
    SpeechResult: str = Form(None)
):
    """Processes user audio response."""
    try:
        if not SpeechResult and not RecordingUrl and not audio:
            logger.error("No audio input provided (SpeechResult, RecordingUrl, or UploadFile)")
            raise HTTPException(status_code=400, detail="No audio input provided")

        if SpeechResult:
            transcribed_text = SpeechResult
            audio_io = None
        elif RecordingUrl:
            response = requests.get(RecordingUrl)
            response.raise_for_status()
            audio_bytes = response.content
            audio_io = io.BytesIO(audio_bytes)
        else:
            audio_bytes = await audio.read()
            audio_io = io.BytesIO(audio_bytes)

        if language is not None:
            if audio_io:
                transcription = spitch_client.speech.transcribe(
                    content=audio_io,
                    language=language
                )
                transcribed_text = transcription.text
            else:
                transcribed_text = SpeechResult

            if not transcribed_text:
                logger.error("Transcription failed or returned empty text")
                raise HTTPException(status_code=400, detail="Transcription failed or empty")

            translation_to_en = spitch_client.text.translate(
                text=transcribed_text,
                source=language,
                target="en"
            )
            english_text = translation_to_en.text

            mistral_response = openrouter_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": english_text}
                ]
            )
            english_response = mistral_response.choices[0].message.content

            translation_back = spitch_client.text.translate(
                text=english_response,
                source="en",
                target=language
            )
            translated_response = translation_back.text

            voice = VOICE_MAP.get(language, DEFAULT_VOICE)
            tts_audio = spitch_client.speech.generate(
                text=translated_response,
                language=language,
                voice=voice
            )
            audio_bytes = tts_audio.read()

            with tempfile.NamedTemporaryFile(suffix=".wav", dir=STATIC_DIR, delete=False) as temp_file:
                temp_file.write(audio_bytes)
                audio_filename = f"static/{os.path.basename(temp_file.name)}"

            twiml = VoiceResponse()
            gather = Gather(
                input="speech",
                action="/process_response",
                method="POST",
                speechTimeout="auto",
                timeout=10
            )
            gather.play(url=f"https://spitchhack.onrender.com/{audio_filename}")
            twiml.append(gather)
            twiml.redirect(f"/process_response?language={language}", method="POST")
            cleanup_static_files()
            return Response(content=str(twiml), media_type="application/xml")

        else:
            if audio_io:
                transcription = spitch_client.speech.transcribe(
                    content=audio_io,
                    language="en"
                )
                transcribed_text = transcription.text
            else:
                transcribed_text = SpeechResult

            if not transcribed_text:
                logger.error("Transcription failed or returned empty text")
                raise HTTPException(status_code=400, detail="Transcription failed or empty")

            detection_prompt = f"The user said: '{transcribed_text}'. What language do they want to speak in? Respond with the ISO 639-1 code (e.g., 'ha' for Hausa).... please make sure you are only returning the code"
            mistral_response = openrouter_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a language detector."},
                    {"role": "user", "content": detection_prompt}
                ]
            )
            detected_lang = mistral_response.choices[0].message.content.strip().lower()

            english_response = f"Okay, we will speak in {detected_lang}."
            translation_back = spitch_client.text.translate(
                text=english_response,
                source="en",
                target=detected_lang
            )
            translated_response = translation_back.text

            voice = VOICE_MAP.get(detected_lang, DEFAULT_VOICE)
            tts_audio = spitch_client.speech.generate(
                text=translated_response,
                language=detected_lang,
                voice=voice
            )
            audio_bytes = tts_audio.read()

            with tempfile.NamedTemporaryFile(suffix=".wav", dir=STATIC_DIR, delete=False) as temp_file:
                temp_file.write(audio_bytes)
                audio_filename = f"static/{os.path.basename(temp_file.name)}"

            twiml = VoiceResponse()
            gather = Gather(
                input="speech",
                action=f"/process_response?language={detected_lang}",
                method="POST",
                speechTimeout="auto",
                timeout=10
            )
            gather.play(url=f"https://spitchhack.onrender.com/{audio_filename}")
            twiml.append(gather)
            twiml.redirect(f"/process_response?language={detected_lang}", method="POST")
            cleanup_static_files()
            return Response(content=str(twiml), media_type="application/xml")

    except Exception as e:
        logger.error(f"Error in process_response: {e}")
        raise HTTPException(status_code=500, detail=f"Error in process_response: {str(e)}")