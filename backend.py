# app_full_streaming_spitch.py
import os
import io
import json
import time
import uuid
import glob
import base64
import wave
import logging
import asyncio
import audioop
import numpy as np
import tempfile

from typing import Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Form, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from twilio.twiml.voice_response import VoiceResponse, Gather, Start
from twilio.rest import Client as TwilioClient

# provider SDKs (must be installed)
from spitch import Spitch
from openai import OpenAI  # OpenRouter-compatible client

load_dotenv()

# -----------------------
# Config / env
# -----------------------
SPITCH_API_KEY = os.getenv("SPITCH_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")  # must be public https
MODEL = os.getenv("MODEL", "mistralai/mistral-7b-instruct:free")
STATIC_DIR = os.getenv("STATIC_DIR", "static")
SILENCE_SECONDS = float(os.getenv("SILENCE_SECONDS", "1.2"))
SILENCE_ENERGY_THRESHOLD = float(os.getenv("SILENCE_ENERGY_THRESHOLD", "0.002"))
MAX_FILE_AGE = int(os.getenv("MAX_FILE_AGE", "300"))

if not BASE_URL:
    raise RuntimeError("BASE_URL must be set to your public HTTPS URL (so Twilio can fetch static files).")
if not SPITCH_API_KEY or not OPENROUTER_API_KEY or not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    raise RuntimeError("Please set SPITCH_API_KEY, OPENROUTER_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN")

# -----------------------
# Clients
# -----------------------
spitch_client = Spitch(api_key=SPITCH_API_KEY)
openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# -----------------------
# App + static
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spitch-streaming")

app = FastAPI()
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -----------------------
# Language map for IVR digits
# -----------------------
LANGUAGE_MAP = {
    "1": ("Yoruba", "yo"),
    "2": ("Igbo", "ig"),
    "3": ("Hausa", "ha"),
    "4": ("English", "en")
}

VOICE_MAP = {
    "en": "jude",
    "ha": "aliyu",
    "yo": "femi",
    "ig": "obinna",
    "am": "default"
}
DEFAULT_VOICE = "jude"

# -----------------------
# In-memory state
# -----------------------
# Map CallSid -> chosen language (set by /process_language)
LANGUAGE_SELECTION: Dict[str, str] = {}
# Map streamSid -> state dict (populated on WS 'start' event)
STREAM_STATE: Dict[str, Dict[str, Any]] = {}

# -----------------------
# Utilities
# -----------------------
def cleanup_static_files(max_age: int = MAX_FILE_AGE):
    now = time.time()
    for path in glob.glob(os.path.join(STATIC_DIR, "*.wav")):
        try:
            if os.path.getmtime(path) < now - max_age:
                os.remove(path)
                logger.info("Deleted old file: %s", path)
        except Exception:
            logger.exception("Failed to delete file %s", path)

def save_wav_pcm16(path: str, pcm_bytes: bytes, sample_rate: int = 8000, channels: int = 1):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

def read_wav_bytes_get_pcm16(wav_bytes: bytes):
    with io.BytesIO(wav_bytes) as b:
        with wave.open(b, "rb") as wf:
            channels = wf.getnchannels()
            sr = wf.getframerate()
            sampwidth = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
            if sampwidth != 2:
                raw = audioop.lin2lin(raw, sampwidth, 2)
            if channels != 1:
                raw = audioop.tomono(raw, 2, 0.5, 0.5)
                channels = 1
            return raw, sr, channels

def pcm16_resample(pcm16_bytes: bytes, src_rate: int, tgt_rate: int = 8000):
    if src_rate == tgt_rate:
        return pcm16_bytes
    new, _ = audioop.ratecv(pcm16_bytes, 2, 1, src_rate, tgt_rate, None)
    return new

def ulaw_to_pcm16(ulaw_bytes: bytes) -> bytes:
    return audioop.ulaw2lin(ulaw_bytes, 2)

def pcm16_to_mulaw(pcm16_bytes: bytes) -> bytes:
    return audioop.lin2ulaw(pcm16_bytes, 2)

def energy_of_pcm16(pcm16_bytes: bytes) -> float:
    if not pcm16_bytes:
        return 0.0
    arr = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
    if arr.size == 0:
        return 0.0
    rms = np.sqrt(np.mean(arr * arr)) / 32768.0
    return float(rms)

def save_bytes_static(bytes_data: bytes, prefix: str) -> str:
    fname = f"{prefix}-{uuid.uuid4().hex}.wav"
    path = os.path.join(STATIC_DIR, fname)
    with open(path, "wb") as f:
        f.write(bytes_data)
        f.flush()
        os.fsync(f.fileno())
    return fname

# -----------------------
# Provider wrappers using SDK calls
# -----------------------
def spitch_transcribe_wav_bytes(wav_bytes: bytes, language: str = "en") -> str:
    audio_io = io.BytesIO(wav_bytes)
    transcription = spitch_client.speech.transcribe(content=audio_io, language=language)
    text = getattr(transcription, "text", None)
    if not text:
        raise RuntimeError("Empty transcription from Spitch")
    return text

def spitch_translate(text: str, source: str, target: str) -> str:
    resp = spitch_client.text.translate(text=text, source=source, target=target)
    t = getattr(resp, "text", None)
    if not t:
        raise RuntimeError("Empty translation from Spitch")
    return t

def spitch_tts_wav_bytes(text: str, language: str = "en", voice: Optional[str] = None) -> bytes:
    if not voice:
        voice = VOICE_MAP.get(language, DEFAULT_VOICE)
    tts = spitch_client.speech.generate(text=text, language=language, voice=voice)
    data = tts.read()
    if not data:
        raise RuntimeError("Empty TTS bytes from Spitch")
    return data

def openrouter_chat_reply(messages: list) -> str:
    resp = openrouter_client.chat.completions.create(model=MODEL, messages=messages)
    try:
        return resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Model returned invalid reply: {e}")

# -----------------------
# IVR endpoints (DTMF)
# -----------------------
@app.post("/voice")
async def voice_entry():
    """
    IVR entry: asks user to press 1-4 for language selection.
    Twilio will POST the pressed digit to /process_language.
    """
    twiml = VoiceResponse()
    gather = Gather(num_digits=1, action="/process_language", method="POST", timeout=8)
    gather.say("Welcome to Proxy. For Yoruba press 1. For Igbo press 2. For Hausa press 3. For English press 4.")
    twiml.append(gather)
    twiml.say("Sorry, we did not receive input.")
    twiml.redirect("/voice")
    return Response(content=str(twiml), media_type="application/xml")

@app.post("/process_language")
async def process_language(Digits: str = Form(None), CallSid: str = Form(None)):
    """
    Handles DTMF language selection. Saves language selection in LANGUAGE_SELECTION[CallSid].
    Responds by playing a short Spitch-generated confirmation, then starts the Media Stream (websocket).
    """
    twiml = VoiceResponse()

    if not Digits or Digits not in LANGUAGE_MAP:
        twiml.say("Invalid selection. Please try again.")
        twiml.redirect("/voice")
        return Response(content=str(twiml), media_type="application/xml")

    lang_name, lang_code = LANGUAGE_MAP[Digits]
    # store selection keyed by CallSid so the WS 'start' handler can pick it up
    if CallSid:
        LANGUAGE_SELECTION[CallSid] = lang_code
        logger.info("Stored language selection for CallSid %s -> %s", CallSid, lang_code)
    else:
        logger.warning("process_language called without CallSid; language not persisted to stream state")

    # generate confirmation TTS using Spitch and play it via Twilio <Play>
    confirm_text = f"You selected {lang_name}. Connecting you to the assistant now."
    try:
        tts_bytes = spitch_tts_wav_bytes(confirm_text, language=lang_code)
        fname = save_bytes_static(tts_bytes, f"confirm-{lang_code}")
        twiml.play(f"{BASE_URL}/static/{fname}")
    except Exception:
        logger.exception("Failed to generate confirmation TTS; falling back to say()")
        twiml.say(f"You selected {lang_name}. Connecting now.")

    # Now instruct Twilio to start a Media Stream to our WS endpoint
    # Twilio expects wss URL. Convert BASE_URL -> wss://...
    if BASE_URL.startswith("https://"):
        wss = BASE_URL.replace("https://", "wss://")
    elif BASE_URL.startswith("http://"):
        wss = BASE_URL.replace("http://", "ws://")
    else:
        wss = BASE_URL
    stream_url = f"{wss}/ws/twilio_stream"
    start = Start()
    start.stream(url=stream_url)
    twiml.append(start)

    # Keep call in streaming mode; also say a tiny prompt and pause while stream runs
    twiml.pause(length=60)  # keep alive while streaming; stream will be active
    cleanup_static_files()
    return Response(content=str(twiml), media_type="application/xml")

# -----------------------
# WebSocket handler for Twilio Media Streams (bidirectional)
# -----------------------
@app.websocket("/ws/twilio_stream")
async def ws_twilio_stream(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket connected (Twilio Media Stream)")
    stream_sid = None
    call_sid = None
    try:
        async for raw in ws.iter_text():
            frame = json.loads(raw)
            event = frame.get("event")
            if event == "connected":
                logger.info("Media stream connected event")
                continue
            if event == "start":
                start = frame.get("start", {})
                stream_sid = start.get("streamSid")
                call_sid = start.get("callSid")
                logger.info("Media stream start: streamSid=%s callSid=%s", stream_sid, call_sid)
                # initialize state for this stream
                lang = LANGUAGE_SELECTION.pop(call_sid, None)  # use language chosen during IVR
                STREAM_STATE[stream_sid] = {
                    "callSid": call_sid,
                    "websocket": ws,
                    "ws_send_lock": asyncio.Lock(),
                    "buffer": bytearray(),
                    "last_audio_ts": time.time(),
                    "processing": False,
                    "lang": lang or "en",
                    "streamSid": stream_sid
                }
                continue
            if event == "media":
                media = frame.get("media", {})
                payload_b64 = media.get("payload")
                sid = frame.get("streamSid") or stream_sid
                if not payload_b64 or not sid or sid not in STREAM_STATE:
                    continue
                try:
                    ulaw = base64.b64decode(payload_b64)
                except Exception:
                    logger.exception("Failed base64 decode")
                    continue
                try:
                    pcm16 = ulaw_to_pcm16(ulaw)
                except Exception:
                    logger.exception("ulaw->pcm conversion failed")
                    continue
                state = STREAM_STATE[sid]
                state["buffer"].extend(pcm16)
                state["last_audio_ts"] = time.time()
                # if not already scheduled, spawn silence waiter
                if not state["processing"]:
                    state["processing"] = True
                    asyncio.create_task(_utterance_silence_waiter(sid))
                continue
            if event == "stop":
                sid = frame.get("streamSid")
                logger.info("Stream stop for %s", sid)
                if sid and sid in STREAM_STATE and STREAM_STATE[sid]["buffer"]:
                    asyncio.create_task(_process_and_reply(sid))
                if sid:
                    asyncio.create_task(_cleanup_stream_state(sid, delay=2.0))
                continue
            if event == "mark":
                # Twilio will send mark events when playback finishes (if we asked for marks)
                logger.info("Received mark: %s", frame.get("mark"))
                continue
            logger.debug("Unhandled event: %s", event)
    except Exception:
        logger.exception("WebSocket handler error")
    finally:
        logger.info("WebSocket closed for streamSid=%s callSid=%s", stream_sid, call_sid)
        try:
            await ws.close()
        except Exception:
            pass
        if stream_sid and stream_sid in STREAM_STATE:
            try:
                del STREAM_STATE[stream_sid]
            except Exception:
                pass

# -----------------------
# Silence waiter & processing
# -----------------------
async def _utterance_silence_waiter(streamSid: str):
    try:
        while True:
            state = STREAM_STATE.get(streamSid)
            if not state:
                return
            elapsed = time.time() - state["last_audio_ts"]
            if elapsed >= SILENCE_SECONDS:
                await _process_and_reply(streamSid)
                st = STREAM_STATE.get(streamSid)
                if st:
                    st["processing"] = False
                return
            await asyncio.sleep(0.15)
    except Exception:
        logger.exception("_utterance_silence_waiter error for %s", streamSid)
        if streamSid in STREAM_STATE:
            STREAM_STATE[streamSid]["processing"] = False

async def _process_and_reply(streamSid: str):
    try:
        state = STREAM_STATE.get(streamSid)
        if not state:
            logger.warning("process_and_reply: unknown stream %s", streamSid)
            return
        buf = bytes(state["buffer"])
        if not buf:
            logger.info("Empty buffer for %s", streamSid)
            return
        uid = uuid.uuid4().hex
        in_wav_name = f"{streamSid}-{uid}.wav"
        in_wav_path = os.path.join(STATIC_DIR, in_wav_name)
        save_wav_pcm16(in_wav_path, buf, sample_rate=8000, channels=1)
        logger.info("Saved incoming utterance to %s", in_wav_path)
        # clear buffer
        state["buffer"] = bytearray()

        # Transcribe with Spitch (use the selected language for STT if you want to set language)
        lang = state.get("lang", "en")
        try:
            with open(in_wav_path, "rb") as f:
                wav_bytes = f.read()
            transcribed = spitch_transcribe_wav_bytes(wav_bytes, language=lang)
            logger.info("Transcribed (lang=%s): %s", lang, transcribed)
        except Exception:
            logger.exception("Transcription failed; trying again with 'en'")
            try:
                transcribed = spitch_transcribe_wav_bytes(wav_bytes, language="en")
            except Exception:
                logger.exception("Transcription ultimately failed for %s", streamSid)
                return

        # Translate to English for the model if necessary
        if lang != "en":
            try:
                english_text = spitch_translate(transcribed, source=lang, target="en")
            except Exception:
                logger.exception("Translation to English failed; using original transcription")
                english_text = transcribed
        else:
            english_text = transcribed

        # Model reply
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": english_text}
            ]
            reply_en = openrouter_chat_reply(messages)
            logger.info("Model reply (en): %s", reply_en)
        except Exception:
            logger.exception("Model call failed; using fallback")
            reply_en = "Sorry, I couldn't process that."

        # Translate reply back if needed
        if lang != "en":
            try:
                reply_local = spitch_translate(reply_en, source="en", target=lang)
            except Exception:
                logger.exception("Translation back to user language failed; using english reply")
                reply_local = reply_en
        else:
            reply_local = reply_en

        # Generate TTS WAV bytes via Spitch in user's language
        try:
            tts_wav_bytes = spitch_tts_wav_bytes(reply_local, language=lang)
        except Exception:
            logger.exception("TTS generation failed")
            return

        # Convert TTS WAV bytes to PCM16 and sample rate
        try:
            pcm16_bytes, sr, channels = read_wav_bytes_get_pcm16(tts_wav_bytes)
        except Exception:
            logger.exception("Failed to parse TTS WAV bytes")
            return
        try:
            pcm16_8k = pcm16_resample(pcm16_bytes, src_rate=sr, tgt_rate=8000)
        except Exception:
            logger.exception("Resample failed; using original PCM")
            pcm16_8k = pcm16_bytes

        # Convert PCM16 -> mu-law
        try:
            mulaw_bytes = pcm16_to_mulaw(pcm16_8k)
        except Exception:
            logger.exception("PCM->mulaw conversion failed")
            return

        # Base64 encode and send back as 'media' event on the websocket
        mulaw_b64 = base64.b64encode(mulaw_bytes).decode("ascii")
        outbound = {
            "event": "media",
            "streamSid": streamSid,
            "media": {
                "payload": mulaw_b64
            }
        }
        lock: asyncio.Lock = state["ws_send_lock"]
        ws = state["websocket"]
        async with lock:
            try:
                await ws.send_text(json.dumps(outbound))
                logger.info("Sent media reply to Twilio for stream %s (bytes=%d)", streamSid, len(mulaw_bytes))
                # send a mark so Twilio will notify when playback completes
                mark_name = f"reply-{uuid.uuid4().hex[:8]}"
                mark_msg = {"event": "mark", "streamSid": streamSid, "mark": {"name": mark_name}}
                await ws.send_text(json.dumps(mark_msg))
                logger.info("Sent mark %s for stream %s", mark_name, streamSid)
            except Exception:
                logger.exception("Failed to send media back over websocket for %s", streamSid)

        # Optionally save TTS for debugging
        try:
            tts_fname = save_bytes_static(tts_wav_bytes, f"tts-{streamSid}")
            logger.info("Saved TTS to static/%s", tts_fname)
        except Exception:
            logger.exception("Failed to save TTS for debugging")

        cleanup_static_files()
    except Exception:
        logger.exception("Top-level error in _process_and_reply for %s", streamSid)

async def _cleanup_stream_state(streamSid: str, delay: float = 2.0):
    await asyncio.sleep(delay)
    if streamSid in STREAM_STATE:
        try:
            del STREAM_STATE[streamSid]
            logger.info("Cleared stream state for %s", streamSid)
        except Exception:
            logger.exception("Failed to clear stream state for %s", streamSid)

# -----------------------
# Health
# -----------------------
@app.get("/health")
async def health():
    return {"status": "ok"}






















# from fastapi import FastAPI, UploadFile, Form, HTTPException
# from fastapi.responses import Response
# from fastapi.staticfiles import StaticFiles
# import io
# import os
# import base64
# import uuid
# import tempfile
# import glob
# import time
# import re
# from spitch import Spitch
# from openai import OpenAI
# from dotenv import load_dotenv
# from twilio.twiml.voice_response import VoiceResponse, Gather
# import requests
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI()

# # Ensure static directory exists
# STATIC_DIR = "static"
# try:
#     os.makedirs(STATIC_DIR, exist_ok=True)
#     logger.info(f"Created static directory: {STATIC_DIR}")
# except Exception as e:
#     logger.error(f"Failed to create static directory: {e}")
#     raise RuntimeError(f"Cannot create static directory: {e}")

# # Mount static directory
# if os.path.exists(STATIC_DIR):
#     app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# else:
#     logger.error(f"Static directory {STATIC_DIR} does not exist")
#     raise RuntimeError(f"Cannot mount static directory: {STATIC_DIR}")

# load_dotenv()
# try:
#     spitch_client = Spitch(api_key=os.getenv("SPITCH_API_KEY"))
#     openrouter_client = OpenAI(
#         base_url="https://openrouter.ai/api/v1",
#         api_key=os.getenv("OPENROUTER_API_KEY"),
#     )
# except Exception as e:
#     logger.error(f"Failed to initialize API clients: {e}")
#     raise RuntimeError(f"API initialization failed: {e}")

# MODEL = "mistralai/mistral-7b-instruct:free"

# VOICE_MAP = {
#     "en": "jude",
#     "ha": "aliyu",
#     "yo": "femi",
#     "ig": "obinna",
#     "am": "default"  # Replace with actual Spitch Amharic voice if supported
# }
# DEFAULT_VOICE = "jude"
# VALID_LANGUAGES = {"en", "ha", "yo", "ig", "am"}

# def cleanup_static_files(max_age_seconds=3600):
#     """Clean up old audio files."""
#     try:
#         for file in glob.glob(f"{STATIC_DIR}/*.wav"):
#             if os.path.getmtime(file) < time.time() - max_age_seconds:
#                 os.remove(file)
#                 logger.info(f"Deleted old file: {file}")
#     except Exception as e:
#         logger.error(f"Failed to clean up static files: {e}")

# def clean_language_code(lang: str) -> str:
#     """Clean language code to ensure it's a valid ISO 639-1 code."""
#     if not lang:
#         return None
#     # Remove quotes, whitespace, and non-lowercase letters
#     cleaned = re.sub(r'[^a-z]', '', lang.strip().lower())
#     return cleaned if cleaned in VALID_LANGUAGES else None

# @app.post("/start_call")
# async def start_call():
#     """Asks user for preferred language and allows interruption."""
#     try:
#         text = "Welcome to proxy, What language do you want to speak in?"
#         audio_response = spitch_client.speech.generate(
#             text=text,
#             language="en",
#             voice=DEFAULT_VOICE
#         )
#         audio_bytes = audio_response.read()

#         with tempfile.NamedTemporaryFile(suffix=".wav", dir=STATIC_DIR, delete=False) as temp_file:
#             temp_file.write(audio_bytes)
#             audio_filename = f"static/{os.path.basename(temp_file.name)}"

#         twiml = VoiceResponse()
#         gather = Gather(
#             input="speech",
#             action="/process_response",
#             method="POST",
#             speechTimeout="auto",
#             timeout=10
#         )
#         gather.play(url=f"https://spitchhack.onrender.com/{audio_filename}")
#         twiml.append(gather)
#         twiml.say("Sorry, I didn't catch that. Please try again.", voice="Polly.Joanna")
#         twiml.redirect("/start_call", method="POST")
#         cleanup_static_files()
#         return Response(content=str(twiml), media_type="application/xml")
#     except Exception as e:
#         logger.error(f"Error in start_call: {e}")
#         raise HTTPException(status_code=500, detail=f"Error in start_call: {str(e)}")

# @app.post("/process_response")
# async def process_response(
#     audio: UploadFile = None,
#     language: str = Form(None),
#     RecordingUrl: str = Form(None),
#     SpeechResult: str = Form(None)
# ):
#     """Processes user audio response."""
#     try:
#         if not SpeechResult and not RecordingUrl and not audio:
#             logger.error("No audio input provided (SpeechResult, RecordingUrl, or UploadFile)")
#             raise HTTPException(status_code=400, detail="No audio input provided")

#         # Clean and validate language parameter if provided
#         if language is not None:
#             cleaned_language = clean_language_code(language)
#             if cleaned_language not in VALID_LANGUAGES:
#                 logger.error(f"Invalid language parameter: {language}")
#                 raise HTTPException(status_code=400, detail=f"Invalid language '{language}'. Supported: {VALID_LANGUAGES}")
#         else:
#             cleaned_language = None

#         if SpeechResult:
#             transcribed_text = SpeechResult
#             audio_io = None
#         elif RecordingUrl:
#             response = requests.get(RecordingUrl)
#             response.raise_for_status()
#             audio_bytes = response.content
#             audio_io = io.BytesIO(audio_bytes)
#         else:
#             audio_bytes = await audio.read()
#             audio_io = io.BytesIO(audio_bytes)

#         if cleaned_language is not None:
#             if audio_io:
#                 transcription = spitch_client.speech.transcribe(
#                     content=audio_io,
#                     language=cleaned_language
#                 )
#                 transcribed_text = transcription.text
#             else:
#                 transcribed_text = SpeechResult

#             if not transcribed_text:
#                 logger.error("Transcription failed or returned empty text")
#                 raise HTTPException(status_code=400, detail="Transcription failed or empty")

#             translation_to_en = spitch_client.text.translate(
#                 text=transcribed_text,
#                 source=cleaned_language,
#                 target="en"
#             )
#             english_text = translation_to_en.text

#             mistral_response = openrouter_client.chat.completions.create(
#                 model=MODEL,
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {"role": "user", "content": english_text}
#                 ]
#             )
#             english_response = mistral_response.choices[0].message.content

#             translation_back = spitch_client.text.translate(
#                 text=english_response,
#                 source="en",
#                 target=cleaned_language
#             )
#             translated_response = translation_back.text

#             voice = VOICE_MAP.get(cleaned_language, DEFAULT_VOICE)
#             tts_audio = spitch_client.speech.generate(
#                 text=translated_response,
#                 language=cleaned_language,
#                 voice=voice
#             )
#             audio_bytes = tts_audio.read()

#             with tempfile.NamedTemporaryFile(suffix=".wav", dir=STATIC_DIR, delete=False) as temp_file:
#                 temp_file.write(audio_bytes)
#                 audio_filename = f"static/{os.path.basename(temp_file.name)}"

#             twiml = VoiceResponse()
#             gather = Gather(
#                 input="speech",
#                 action="/process_response",
#                 method="POST",
#                 speechTimeout="auto",
#                 timeout=10
#             )
#             gather.play(url=f"https://spitchhack.onrender.com/{audio_filename}")
#             twiml.append(gather)
#             twiml.redirect(f"/process_response?language={cleaned_language}", method="POST")
#             cleanup_static_files()
#             return Response(content=str(twiml), media_type="application/xml")

#         else:
#             if audio_io:
#                 transcription = spitch_client.speech.transcribe(
#                     content=audio_io,
#                     language="en"
#                 )
#                 transcribed_text = transcription.text
#             else:
#                 transcribed_text = SpeechResult

#             if not transcribed_text:
#                 logger.error("Transcription failed or returned empty text")
#                 raise HTTPException(status_code=400, detail="Transcription failed or empty")

#             # Refined prompt to ensure only ISO 639-1 code is returned
#             valid_languages = ["en", "yo", "ig", "ha", "am"]
#             detection_prompt = (
#                 f"The user said: '{transcribed_text}'. Identify the language they want to speak in and respond with "
#                 f"only the ISO 639-1 code (e.g., 'ha' for Hausa, 'en' for English, 'yo' for Yoruba, 'ig' for Igbo, 'am' for Amharic). "
#                 f"Return exactly one of {valid_languages} with no additional text, quotes, or modifications."
#             )
#             mistral_response = openrouter_client.chat.completions.create(
#                 model=MODEL,
#                 messages=[
#                     {"role": "system", "content": "You are a language detector. Respond with only the ISO 639-1 code, without quotes or extra text."},
#                     {"role": "user", "content": detection_prompt}
#                 ]
#             )
#             detected_lang = mistral_response.choices[0].message.content.strip().lower()
#             cleaned = clean_language_code(detected_lang)

#             # Validate detected language
#             if cleaned not in VALID_LANGUAGES:
#                 logger.error(f"Invalid language detected: {cleaned}")
#                 raise HTTPException(status_code=400, detail=f"Detected language '{cleaned}' is not supported. Supported: {VALID_LANGUAGES}")

#             english_response = f"Okay, we will speak in {cleaned}."
#             translation_back = spitch_client.text.translate(
#                 text=english_response,
#                 source="en",
#                 target=cleaned
#             )
#             translated_response = translation_back.text

#             voice = VOICE_MAP.get(cleaned, DEFAULT_VOICE)
#             tts_audio = spitch_client.speech.generate(
#                 text=translated_response,
#                 language=cleaned,
#                 voice=voice
#             )
#             audio_bytes = tts_audio.read()

#             with tempfile.NamedTemporaryFile(suffix=".wav", dir=STATIC_DIR, delete=False) as temp_file:
#                 temp_file.write(audio_bytes)
#                 audio_filename = f"static/{os.path.basename(temp_file.name)}"

#             twiml = VoiceResponse()
#             gather = Gather(
#                 input="speech",
#                 action=f"/process_response?language={cleaned}",
#                 method="POST",
#                 speechTimeout="auto",
#                 timeout=10
#             )
#             gather.play(url=f"https://spitchhack.onrender.com/{audio_filename}")
#             twiml.append(gather)
#             twiml.redirect(f"/process_response?language={cleaned}", method="POST")
#             cleanup_static_files()
#             return Response(content=str(twiml), media_type="application/xml")

#     except Exception as e:
#         logger.error(f"Error in process_response: {e}")
#         raise HTTPException(status_code=500, detail=f"Error in process_response: {str(e)}")