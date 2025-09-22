import os
import logging
import base64
from typing import Dict, Any
from fastapi import FastAPI, Request, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse
from twilio.request_validator import RequestValidator
from dotenv import load_dotenv
from spitch import Spitch
from openai import AsyncOpenAI
import json
import asyncio
from urllib.parse import urlparse

# ---- Config ----
load_dotenv()

app = FastAPI()

# Validate environment variables
required_vars = [
    "SPITCH_API_KEY",
    "OPENROUTER_API_KEY",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "BASE_URL",
]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing environment variable: {var}")

SPITCH_API_KEY = os.getenv("SPITCH_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
MODEL = os.getenv("MODEL", "")
SYSTEM_PROMPT = "You are a helpful assistant named Proxy. This conversation is being translated to voice, Speak like a human. so answer carefully. When you respond, please spell out all numbers, for example twenty not 20. Do not include emojis in your responses. Do not include bullet points, asterisks, or special symbols."

# ---- Clients ----
try:
    spitch_client = Spitch(api_key=SPITCH_API_KEY)
    openrouter_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to initialize clients: {e}")

# ---- App setup ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("conversation-relay")
app = FastAPI()
twilio_validator = RequestValidator(TWILIO_AUTH_TOKEN)

# ---- Language map ----
LANGUAGE_MAP = {
    "1": ("Yoruba", "yo-NG", "yo"),  # (lang_name, lang_code_twiml, lang_code_spitch)
    "2": ("Igbo", "ig-NG", "ig"),
    "3": ("Hausa", "ha-NG", "ha"),
    "4": ("English", "en-US", "en")
}

VOICE_MAP = {
    "yo": "sade",
    "ig": "ngozi",
    "ha": "amina",
    "en": "lina"
}

# Shared state
LANGUAGE_SELECTION: Dict[str, tuple] = {}
CONVERSATION_HISTORY: Dict[str, list] = {}

# ---- Helpers ----
def spitch_translate(text: str, source: str, target: str) -> str:
    """Translate `text` from `source` language to `target` using Spitch API."""
    try:
        resp = spitch_client.text.translate(text=text, source=source, target=target)
        t = getattr(resp, "text", None)
        if not t:
            raise RuntimeError("Empty translation from Spitch")
        return t
    except Exception as e:
        logger.error(f"Spitch translation failed: {e}")
        raise RuntimeError(f"Translation error: {e}")

def spitch_tts(text: str, lang: str, voice: str) -> bytes:
    """Synthesize `text` to audio in `lang` using `voice` with Spitch API."""
    try:
        resp = spitch_client.speech.generate(
            text=text,
            language=lang,
            voice=voice
        )
        audio = resp.read()
        if not audio:
            raise RuntimeError("Empty audio from Spitch")
        return audio
    except Exception as e:
        logger.error(f"Spitch TTS failed: {e}")
        raise RuntimeError(f"TTS error: {e}")

def spitch_transcribe(audio_data: bytes, lang: str) -> str:
    """Transcribe audio data using Spitch API."""
    try:
        # Correctly pass audio data with the keyword 'content' and remove unsupported parameters
        resp = spitch_client.speech.transcribe(
            content=audio_data,
            language=lang
        )
        t = getattr(resp, "text", None)
        if not t:
            raise RuntimeError("Empty transcription from Spitch")
        return t
    except Exception as e:
        logger.error(f"Spitch transcription failed: {e}")
        raise RuntimeError(f"Transcription error: {e}")

async def openrouter_chat_reply(messages: list) -> str:
    """Get a chat completion from OpenRouter."""
    try:
        resp = await openrouter_client.chat.completions.create(model=MODEL, messages=messages)
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenRouter API error: {e}")
        return "Sorry, I couldn't process your request. Please try again."

async def stream_tts_and_play(websocket: WebSocket, call_sid: str, text: str):
    """
    Synthesizes text to audio and sends it to Twilio's stream for playback.
    """
    _, _, lang_spitch = LANGUAGE_SELECTION.get(call_sid, ("English", "en-US", "en"))
    voice = VOICE_MAP.get(lang_spitch, "lina")

    try:
        audio_bytes = spitch_tts(text, lang_spitch, voice)
        
        # Twilio requires audio to be in chunks for streaming playback
        chunk_size = 32000 # Example chunk size, you may need to adjust
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            payload = {
                "event": "media",
                "media": {
                    "payload": base64.b64encode(chunk).decode('utf-8')
                },
                "streamSid": websocket.scope["path_params"]["stream_sid"]
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.01) # Small delay to prevent network congestion
    except Exception as e:
        logger.error(f"Error during TTS and streaming audio: {e}")

# ---- Root endpoint ----
@app.get("/")
async def root():
    return {"message": "Welcome to the SpitchHack Voice Relay API. Use /health to check status."}

# ---- TwiML entry ----
@app.post("/voice")
async def voice_entry(request: Request):
    # Validate Twilio webhook
    form_data = await request.form()
    signature = request.headers.get("X-Twilio-Signature", "")
    url = str(request.url)
    if not twilio_validator.validate(url, dict(form_data), signature):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    twiml = VoiceResponse()
    # Gather language selection
    gather = twiml.gather(
        num_digits=1,
        action="/process_language",
        method="POST",
        timeout=8
    )
    gather.say("Welcome to Proxy. For Yoruba press 1. For Igbo press 2. For Hausa press 3. For English press 4.")
    # If gather doesn't get input:
    twiml.redirect("/process_language_fallback")

    return Response(content=str(twiml), media_type="application/xml")

@app.post("/process_language_fallback")
async def process_language_fallback(request: Request):
    twiml = VoiceResponse()
    twiml.say("Sorry, we did not receive input. Redirecting you back to language selection.")
    twiml.redirect("/voice")
    return Response(content=str(twiml), media_type="application/xml")

@app.post("/process_language")
async def process_language(request: Request, Digits: str = Form(None), CallSid: str = Form(None)):
    # Validate Twilio webhook
    form_data = await request.form()
    signature = request.headers.get("X-Twilio-Signature", "")
    url = str(request.url)
    if not twilio_validator.validate(url, dict(form_data), signature):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    twiml = VoiceResponse()
    if not (Digits and CallSid and Digits in LANGUAGE_MAP):
        twiml.say("Invalid selection or call ID. Please try again.")
        twiml.redirect("/voice")
        return Response(content=str(twiml), media_type="application/xml")

    lang_name, _, lang_spitch = LANGUAGE_MAP[Digits]
    LANGUAGE_SELECTION[CallSid] = (lang_name, "en-US", lang_spitch)
    logger.info("Language set for CallSid %s -> %s", CallSid, lang_name)
    
    twiml.say(f"You selected {lang_name}. Please say something after the tone.")
    twiml.pause(length=1)

    connect = twiml.connect()
    # Use a raw stream to send audio for Google Cloud STT
    connect.stream(url=f"wss://{urlparse(BASE_URL).netloc}/stream/{CallSid}")

    return Response(content=str(twiml), media_type="application/xml")


# ---- Health check ----
@app.get("/health")
async def health():
    status = {"status": "ok", "services": {}}
    # Test Spitch translate
    try:
        _ = spitch_translate("test", "en", "en")
        status["services"]["spitch_translate"] = "ok"
    except Exception as e:
        status["services"]["spitch_translate"] = f"down: {e}"
    # Test Spitch TTS
    try:
        _ = spitch_tts("test", "en", "lina")
        status["services"]["spitch_tts"] = "ok"
    except Exception as e:
        status["services"]["spitch_tts"] = f"down: {e}"
    # Test Spitch transcribe
    try:
        # Create a tiny silent audio file for testing
        test_audio = b"\xfd" * 8000
        _ = spitch_transcribe(test_audio, "en")
        status["services"]["spitch_transcribe"] = "ok"
    except Exception as e:
        status["services"]["spitch_transcribe"] = f"down: {e}"
    # Test OpenRouter
    try:
        _ = await openrouter_client.chat.completions.create(model=MODEL, messages=[{"role": "system", "content": "test"}])
        status["services"]["openrouter"] = "ok"
    except Exception as e:
        status["services"]["openrouter"] = f"down: {e}"
    return status

@app.websocket("/stream/{call_sid}")
async def stream_handler(websocket: WebSocket, call_sid: str):
    await websocket.accept()
    logger.info(f"Stream handler started for call: {call_sid}")
    
    # Store the user's conversation history
    history = []
    
    # Get language config
    _, _, lang_spitch = LANGUAGE_SELECTION.get(call_sid, ("English", "en-US", "en"))

    # Audio data buffer and silence detection state
    audio_buffer = bytearray()
    silence_counter = 0
    is_speaking = False
    
    # We use a task to handle the transcription so we don't block the WebSocket loop
    async def process_audio(audio_data: bytes):
        nonlocal history
        try:
            # Transcribe the buffered audio
            transcription = spitch_transcribe(audio_data, lang_spitch)
            logger.info(f"Final Transcription: {transcription}")
            
            if not transcription.strip():
                return

            # Translate transcription from target to English for LLM
            if lang_spitch != "en":
                english_text = spitch_translate(transcription, source=lang_spitch, target="en")
            else:
                english_text = transcription

            history.append({"role": "user", "content": english_text})
            
            # Get response from LLM
            llm_response = await openrouter_chat_reply(history)

            # Translate LLM response from English back to target language
            if lang_spitch != "en":
                local_text = spitch_translate(llm_response, source="en", target=lang_spitch)
            else:
                local_text = llm_response

            history.append({"role": "assistant", "content": llm_response})
            
            # Use Spitch to synthesize audio and stream it back to Twilio
            await stream_tts_and_play(websocket, call_sid, local_text)
            
            # Log the history and trim it to avoid memory issues
            CONVERSATION_HISTORY[call_sid] = history[-20:]
            
        except Exception as e:
            logger.error(f"Error during audio processing: {e}", exc_info=True)

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            if data['event'] == 'media':
                payload = base64.b64decode(data['media']['payload'])
                audio_buffer.extend(payload)
                
                # Simple silence detection based on audio volume
                is_silent = all(b in range(120, 136) for b in payload) # 128 is ulaw silence, give a small range
                
                if is_silent:
                    silence_counter += 1
                else:
                    silence_counter = 0
                
                if not is_speaking and not is_silent:
                    is_speaking = True
                    logger.info("Speech started.")
                
                # Check for end of speech (silence for a few chunks)
                if is_speaking and silence_counter > 5: # 5 chunks of silence
                    logger.info("Speech ended. Transcribing audio...")
                    transcription_task = asyncio.create_task(process_audio(bytes(audio_buffer)))
                    audio_buffer.clear()
                    is_speaking = False
                    silence_counter = 0
                
            elif data['event'] == 'stop':
                # Final transcription if call stops before silence is detected
                if audio_buffer:
                    transcription_task = asyncio.create_task(process_audio(bytes(audio_buffer)))
                break
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"WebSocket or application error: {e}", exc_info=True)
    finally:
        CONVERSATION_HISTORY.pop(call_sid, None)
        logger.info(f"Cleaned up for CallSid {call_sid}")
        try:
            await websocket.close()
        except RuntimeError:
            pass # Ignore if already closed
