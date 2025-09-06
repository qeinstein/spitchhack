# app_streaming_bidirectional.py
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
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from twilio.twiml.voice_response import VoiceResponse, Start
from twilio.rest import Client as TwilioClient

# External SDKs (must be installed)
from spitch import Spitch
from openai import OpenAI  # OpenRouter-compatible client

# ---- Load env ----
load_dotenv()
SPITCH_API_KEY = os.getenv("SPITCH_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")  # e.g. https://spitchhack.onrender.com
MODEL = os.getenv("MODEL", "mistralai/mistral-7b-instruct:free")
STATIC_DIR = os.getenv("STATIC_DIR", "static")
MAX_FILE_AGE = int(os.getenv("MAX_FILE_AGE", "300"))
SILENCE_SECONDS = float(os.getenv("SILENCE_SECONDS", "1.2"))
SILENCE_ENERGY_THRESHOLD = float(os.getenv("SILENCE_ENERGY_THRESHOLD", "0.002"))

if not BASE_URL:
    raise RuntimeError("BASE_URL environment variable is required and must be your public HTTPS URL.")

# ---- Clients ----
if not SPITCH_API_KEY or not OPENROUTER_API_KEY or not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    raise RuntimeError("Set SPITCH_API_KEY, OPENROUTER_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN in env")

spitch_client = Spitch(api_key=SPITCH_API_KEY)
openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ---- App ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("twilio-streaming-bidi")
app = FastAPI()
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---- Config ----
VALID_LANGUAGES = {"en", "ha", "yo", "ig", "am"}
VOICE_MAP = {"en": "jude", "ha": "aliyu", "yo": "femi", "ig": "obinna", "am": "default"}
DEFAULT_VOICE = "jude"

# ---- In-memory call/stream state (for single-process). Use Redis for multi-process. ----
# Keyed by streamSid (preferred) and also stores callSid and websocket reference + lock
STREAM_STATE: Dict[str, Dict[str, Any]] = {}
# structure:
# STREAM_STATE[streamSid] = {
#   "callSid": str,
#   "websocket": WebSocket,
#   "ws_send_lock": asyncio.Lock(),
#   "buffer": bytearray(),           # linear PCM16LE @ 8000Hz
#   "last_audio_ts": float,
#   "processing": False,
#   "lang": "en",
# }

# ---- Utilities ----
def cleanup_static_files(max_age: int = MAX_FILE_AGE):
    now = time.time()
    for path in glob.glob(f"{STATIC_DIR}/*.wav"):
        try:
            if os.path.getmtime(path) < now - max_age:
                os.remove(path)
                logger.info("Deleted old file: %s", path)
        except Exception:
            logger.exception("Failed to delete file %s", path)

def save_wav_bytes(path: str, pcm16_bytes: bytes, sample_rate: int = 8000, channels: int = 1):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16_bytes)

def read_wav_bytes_get_pcm16(wav_bytes: bytes):
    """Read WAV bytes and return (pcm16_bytes, sample_rate, channels)."""
    with io.BytesIO(wav_bytes) as b:
        with wave.open(b, "rb") as wf:
            channels = wf.getnchannels()
            sr = wf.getframerate()
            sampwidth = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
            # if not 2 bytes per sample, convert to 2 bytes
            if sampwidth != 2:
                raw = audioop.lin2lin(raw, sampwidth, 2)
            # if multiple channels, mixdown to mono
            if channels != 1:
                raw = audioop.tomono(raw, 2, 0.5, 0.5)
                channels = 1
            return raw, sr, channels

def pcm16_resample(pcm16_bytes: bytes, src_rate: int, tgt_rate: int = 8000):
    """Resample linear PCM16 bytes from src_rate to tgt_rate using audioop.ratecv."""
    if src_rate == tgt_rate:
        return pcm16_bytes
    # audioop.ratecv signature requires (frame, width, channels, inrate, outrate, state)
    # width=2, channels=1 (mono PCM16)
    new, _state = audioop.ratecv(pcm16_bytes, 2, 1, src_rate, tgt_rate, None)
    return new

def ulaw_to_pcm16(ulaw_bytes: bytes) -> bytes:
    """Convert mu-law (8-bit) bytes to PCM16LE."""
    return audioop.ulaw2lin(ulaw_bytes, 2)  # returns bytes of width 2 (16-bit)

def pcm16_to_mulaw(pcm16_bytes: bytes) -> bytes:
    """Convert PCM16LE bytes to mu-law (8-bit) bytes."""
    return audioop.lin2ulaw(pcm16_bytes, 2)  # width=2

def energy_of_pcm16(pcm16_bytes: bytes) -> float:
    if not pcm16_bytes:
        return 0.0
    arr = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
    if arr.size == 0:
        return 0.0
    rms = np.sqrt(np.mean(arr * arr)) / 32768.0
    return float(rms)

def clean_language_code(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    cleaned = "".join([c for c in lang.strip().lower() if c.isalpha()])
    return cleaned if cleaned in VALID_LANGUAGES else None

# ---- Provider wrappers (Spitch & OpenRouter) ----
def spitch_transcribe_wav_bytes(wav_bytes: bytes, language: str = "en") -> str:
    """Transcribe WAV bytes using Spitch SDK (synchronous)."""
    audio_io = io.BytesIO(wav_bytes)
    transcription = spitch_client.speech.transcribe(content=audio_io, language=language)
    text = getattr(transcription, "text", None)
    if not text:
        raise RuntimeError("Empty transcription from Spitch")
    return text

def spitch_tts_wav_bytes(text: str, language: str = "en", voice: Optional[str] = None) -> bytes:
    """Generate TTS WAV bytes from Spitch SDK (we expect WAV/PCM16LE-like bytes)."""
    if voice is None:
        voice = VOICE_MAP.get(language, DEFAULT_VOICE)
    tts = spitch_client.speech.generate(text=text, language=language, voice=voice)
    data = tts.read()
    if not data:
        raise RuntimeError("Empty TTS bytes from Spitch")
    return data

def spitch_translate(text: str, source: str, target: str) -> str:
    resp = spitch_client.text.translate(text=text, source=source, target=target)
    t = getattr(resp, "text", None)
    if not t:
        raise RuntimeError("Empty translation from Spitch")
    return t

def openrouter_chat(messages: list) -> str:
    resp = openrouter_client.chat.completions.create(model=MODEL, messages=messages)
    try:
        return resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Invalid model response: {e}")

# ---- TwiML start endpoint: instruct Twilio to start a bidirectional Stream to our WebSocket ----
@app.post("/start_call")
async def start_call():
    """
    Twilio will POST here when the call arrives. Return TwiML that starts a <Start><Stream> to our websocket.
    We use <Start><Stream> inside TwiML; Twilio will then open a WebSocket connection to /ws/twilio_stream.
    """
    try:
        twiml = VoiceResponse()
        start = Start()
        # Twilio expects a wss:// URL. Convert BASE_URL (https://...) into wss://...
        if BASE_URL.startswith("https://"):
            wss = BASE_URL.replace("https://", "wss://")
        elif BASE_URL.startswith("http://"):
            wss = BASE_URL.replace("http://", "ws://")
        else:
            wss = BASE_URL
        stream_url = f"{wss}/ws/twilio_stream"
        start.stream(url=stream_url)
        twiml.append(start)

        # Optionally play a small TTS greeting to let the caller know stream started.
        try:
            tts_bytes = spitch_tts_wav_bytes("Welcome. You can speak now.", language="en", voice=VOICE_MAP.get("en"))
            fname = save_tts_and_get_filename(tts_bytes)  # helper defined below
            twiml.play(f"{BASE_URL}/static/{fname}")
        except Exception:
            # Do not fail if TTS fails; simply continue with stream
            logger.exception("Greeting TTS failed; continuing without play.")

        return Response(content=str(twiml), media_type="application/xml")
    except Exception as e:
        logger.exception("start_call error")
        raise HTTPException(status_code=500, detail=str(e))

# helper for saving TTS wav bytes as file and returning filename
def save_tts_and_get_filename(wav_bytes: bytes) -> str:
    # ensure wav_bytes are valid WAV; if not, try to wrap raw pcm in a WAV header (we assume spitch returns WAV)
    fname = f"tts-{uuid.uuid4().hex}.wav"
    path = os.path.join(STATIC_DIR, fname)
    with open(path, "wb") as f:
        f.write(wav_bytes)
        f.flush()
        os.fsync(f.fileno())
    return fname

# ---- WebSocket handler for Twilio Media Streams (bidirectional) ----
@app.websocket("/ws/twilio_stream")
async def ws_twilio_stream(websocket: WebSocket):
    """
    Receives Twilio Media Stream messages and also sends 'media' messages back into the call.
    We're using streamSid as the primary key for the stream in STREAM_STATE.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted from Twilio Media Streams")
    stream_sid = None
    try:
        async for raw in websocket.iter_text():
            frame = json.loads(raw)
            event = frame.get("event")
            if event == "connected":
                logger.info("Twilio connected event received")
                continue
            if event == "start":
                start = frame.get("start", {})
                stream_sid = start.get("streamSid")
                call_sid = start.get("callSid")
                logger.info("Stream start: streamSid=%s callSid=%s", stream_sid, call_sid)
                # init state
                STREAM_STATE[stream_sid] = {
                    "callSid": call_sid,
                    "websocket": websocket,
                    "ws_send_lock": asyncio.Lock(),
                    "buffer": bytearray(),
                    "last_audio_ts": time.time(),
                    "processing": False,
                    "lang": None,  # detected language once we have first transcription
                    "streamSid": stream_sid
                }
                continue
            if event == "media":
                # Twilio sends payload base64 of mulaw/8000
                streamSid = frame.get("streamSid")
                media = frame.get("media", {})
                payload_b64 = media.get("payload")
                if not payload_b64:
                    continue
                try:
                    ulaw_bytes = base64.b64decode(payload_b64)
                except Exception:
                    logger.exception("Failed to base64-decode payload")
                    continue
                # Convert mu-law -> linear PCM16 (16-bit)
                try:
                    pcm16 = ulaw_to_pcm16(ulaw_bytes)  # bytes, 16-bit samples, 8000Hz
                except Exception:
                    logger.exception("ulaw->pcm conversion failed")
                    continue
                # Append to stream buffer
                sid = streamSid or stream_sid
                if not sid or sid not in STREAM_STATE:
                    logger.warning("Media for unknown streamSid %s", sid)
                    continue
                state = STREAM_STATE[sid]
                state["buffer"].extend(pcm16)
                state["last_audio_ts"] = time.time()
                # if not already waiting for silence, start a worker
                if not state["processing"]:
                    state["processing"] = True
                    asyncio.create_task(_utterance_silence_waiter(sid))
                continue
            if event == "stop":
                streamSid = frame.get("streamSid")
                logger.info("Stream stop for %s", streamSid)
                # process any remaining buffer
                if streamSid and streamSid in STREAM_STATE and STREAM_STATE[streamSid]["buffer"]:
                    asyncio.create_task(_process_and_reply(streamSid))
                # cleanup after short delay
                if streamSid:
                    asyncio.create_task(_cleanup_stream_state(streamSid, delay=2.0))
                continue
            if event == "mark":
                # marks come back from Twilio when our sent media finishes
                logger.info("Received mark event: %s", frame.get("mark", {}))
                continue
            # log others
            logger.debug("Unhandled event: %s", event)
    except Exception:
        logger.exception("WebSocket handler error")
    finally:
        logger.info("WebSocket connection closed for streamSid=%s", stream_sid)
        # attempt cleanup
        if stream_sid and stream_sid in STREAM_STATE:
            try:
                del STREAM_STATE[stream_sid]
            except Exception:
                pass

# ---- Silence waiter ----
async def _utterance_silence_waiter(streamSid: str):
    """
    Wait until no new audio appended for SILENCE_SECONDS, then process buffer.
    """
    try:
        while True:
            state = STREAM_STATE.get(streamSid)
            if not state:
                return
            elapsed = time.time() - state["last_audio_ts"]
            if elapsed >= SILENCE_SECONDS:
                # process
                await _process_and_reply(streamSid)
                # mark processing false to allow next worker
                st = STREAM_STATE.get(streamSid)
                if st:
                    st["processing"] = False
                return
            await asyncio.sleep(0.15)
    except Exception:
        logger.exception("_utterance_silence_waiter error for %s", streamSid)
        if streamSid in STREAM_STATE:
            STREAM_STATE[streamSid]["processing"] = False

# ---- Core process: transcribe -> detect language (if needed) -> model -> tts -> send back via websocket ----
async def _process_and_reply(streamSid: str):
    try:
        state = STREAM_STATE.get(streamSid)
        if not state:
            logger.warning("process_and_reply: unknown streamSid %s", streamSid)
            return
        buf = bytes(state["buffer"])
        if not buf:
            logger.info("Empty buffer for %s", streamSid)
            return

        # Save incoming utterance as WAV (PCM16 @8000)
        uid = uuid.uuid4().hex
        in_wav_name = f"{streamSid}-{uid}.wav"
        in_wav_path = os.path.join(STATIC_DIR, in_wav_name)
        save_wav_bytes(in_wav_path, buf, sample_rate=8000, channels=1)
        logger.info("Saved incoming utterance: %s", in_wav_path)

        # Clear buffer for next utterance
        state["buffer"] = bytearray()

        # Transcribe with Spitch
        with open(in_wav_path, "rb") as f:
            wav_bytes = f.read()
        try:
            transcribed = spitch_transcribe_wav_bytes(wav_bytes, language="en")
            logger.info("Transcribed (raw): %s", transcribed)
        except Exception:
            # If transcription with 'en' fails, try flexible approach: attempt with no forced language by using default API call
            try:
                transcribed = spitch_transcribe_wav_bytes(wav_bytes, language="en")
                logger.info("Transcribed fallback: %s", transcribed)
            except Exception as e:
                logger.exception("Transcription failed for stream %s: %s", streamSid, e)
                return

        # If we haven't detected language for this stream, use the model to detect language code from the transcribed snippet
        if not state.get("lang"):
            detection_prompt = (
                f"The user said: '{transcribed}'. Identify which language the user is speaking in. "
                f"Return exactly one ISO 639-1 code from this set: {list(VALID_LANGUAGES)} and nothing else."
            )
            try:
                detected = openrouter_chat([{"role": "system", "content": "You are a language detector. Reply only with one ISO 639-1 code."},
                                           {"role": "user", "content": detection_prompt}]).strip().lower()
                detected_clean = clean_language_code(detected)
                if detected_clean:
                    state["lang"] = detected_clean
                else:
                    # if detection fails, default to English
                    logger.warning("Language detection failed or unknown: '%s' -> defaulting to en", detected)
                    state["lang"] = "en"
            except Exception:
                logger.exception("Language detection model failed; defaulting to 'en'")
                state["lang"] = "en"

        user_lang = state.get("lang", "en")
        logger.info("Using language '%s' for this stream", user_lang)

        # Translate to English for the model if needed
        if user_lang != "en":
            try:
                english_text = spitch_translate(transcribed, source=user_lang, target="en")
            except Exception:
                logger.exception("Translation to English failed; using original transcription")
                english_text = transcribed
        else:
            english_text = transcribed

        # Ask chat model
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": english_text}
            ]
            model_reply_en = openrouter_chat(messages)
            logger.info("Model reply (EN): %s", model_reply_en)
        except Exception:
            logger.exception("Model call failed; using fallback reply")
            model_reply_en = "Sorry, I couldn't formulate a response."

        # Translate model reply back to user's language if necessary
        if user_lang != "en":
            try:
                model_reply_local = spitch_translate(model_reply_en, source="en", target=user_lang)
            except Exception:
                logger.exception("Translation back to user language failed; using English reply")
                model_reply_local = model_reply_en
        else:
            model_reply_local = model_reply_en

        # Generate TTS WAV bytes via Spitch (may be any sample rate)
        try:
            tts_wav_bytes = spitch_tts_wav_bytes(model_reply_local, language=user_lang, voice=VOICE_MAP.get(user_lang, DEFAULT_VOICE))
        except Exception:
            logger.exception("TTS generation failed")
            return

        # Prepare outbound payload: Twilio requires audio/x-mulaw at 8000 Hz with NO header, base64 encoded
        # Steps:
        # 1) parse WAV bytes -> PCM16 bytes and sample rate
        try:
            pcm16_bytes, sr, channels = read_wav_bytes_get_pcm16(tts_wav_bytes)
        except Exception:
            logger.exception("Failed to read TTS WAV bytes; abort outbound audio")
            return
        # 2) if sr != 8000, resample to 8000
        try:
            pcm16_8k = pcm16_resample(pcm16_bytes, src_rate=sr, tgt_rate=8000)
        except Exception:
            logger.exception("Resample failed; attempting to use original PCM")
            pcm16_8k = pcm16_bytes
        # 3) ensure mono and 16-bit (already done in read_wav_bytes_get_pcm16)
        # 4) convert PCM16LE -> mu-law 8-bit
        try:
            mulaw_bytes = pcm16_to_mulaw(pcm16_8k)
        except Exception:
            logger.exception("PCM->mulaw conversion failed")
            return
        # 5) base64 encode and send as "media" message to Twilio WebSocket
        mulaw_b64 = base64.b64encode(mulaw_bytes).decode("ascii")

        # Send via websocket (must send to the same websocket connection Twilio opened)
        send_ws = state["websocket"]
        lock: asyncio.Lock = state["ws_send_lock"]
        media_msg = {
            "event": "media",
            "streamSid": streamSid,
            "media": {
                "payload": mulaw_b64
            }
        }
        # Acquire lock to ensure we don't interleave writes
        async with lock:
            try:
                await send_ws.send_text(json.dumps(media_msg))
                logger.info("Sent outbound media to Twilio for stream %s (bytes=%d)", streamSid, len(mulaw_bytes))
                # After sending media, send a 'mark' so Twilio will notify us when playback completes
                mark_name = f"reply-{uuid.uuid4().hex[:8]}"
                mark_msg = {"event": "mark", "streamSid": streamSid, "mark": {"name": mark_name}}
                await send_ws.send_text(json.dumps(mark_msg))
                logger.info("Sent mark %s for stream %s", mark_name, streamSid)
            except Exception:
                logger.exception("Failed to send outbound media over websocket to Twilio")

        # optional: save tts to static for debugging or fallback play
        try:
            tts_fname = save_tts_and_get_filename(tts_wav_bytes)
            logger.info("Saved TTS to static/%s for debugging", tts_fname)
        except Exception:
            logger.exception("Failed to save TTS wav for debugging")

        cleanup_static_files()
    except Exception:
        logger.exception("Top-level _process_and_reply error for stream %s", streamSid)

# ---- Delayed cleanup helper ----
async def _cleanup_stream_state(streamSid: str, delay: float = 2.0):
    await asyncio.sleep(delay)
    if streamSid in STREAM_STATE:
        try:
            del STREAM_STATE[streamSid]
            logger.info("Cleared stream state for %s", streamSid)
        except Exception:
            logger.exception("Failed to cleanup stream state for %s", streamSid)

# ---- Health ----
@app.get("/health")
async def health():
    return {"status": "ok"}
















# # app_streaming_full.py
# import os
# import io
# import json
# import time
# import uuid
# import glob
# import base64
# import wave
# import logging
# import asyncio
# import tempfile
# from typing import Dict, Any, Optional
# from dotenv import load_dotenv
# from fastapi import FastAPI, WebSocket, Request, Form, UploadFile, HTTPException
# from fastapi.responses import Response
# from fastapi.staticfiles import StaticFiles
# from twilio.twiml.voice_response import VoiceResponse, Start
# from twilio.rest import Client as TwilioClient
# import numpy as np

# # External provider SDKs (must be installed)
# # pip install spitch openai twilio fastapi uvicorn python-dotenv numpy aiofiles
# # NOTE: the actual Spitch package name and OpenRouter client may vary; adjust imports as needed.
# from spitch import Spitch
# from openai import OpenAI  # this is the OpenRouter-compatible client used earlier

# # ---- Load environment ----
# load_dotenv()
# SPITCH_API_KEY = os.getenv("SPITCH_API_KEY")
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
# TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
# BASE_URL = os.getenv("BASE_URL", "https://spitchhack.onrender.com")  # must be public https url
# STATIC_DIR = os.getenv("STATIC_DIR", "static")
# MODEL = os.getenv("MODEL", "mistralai/mistral-7b-instruct:free")
# MAX_FILE_AGE = int(os.getenv("MAX_FILE_AGE", "300"))
# SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.002"))
# SILENCE_SECONDS = float(os.getenv("SILENCE_SECONDS", "1.2"))
# VALID_LANGUAGES = {"en", "ha", "yo", "ig", "am"}

# VOICE_MAP = {
#     "en": "jude",
#     "ha": "aliyu",
#     "yo": "femi",
#     "ig": "obinna",
#     "am": "default"
# }
# DEFAULT_VOICE = "jude"

# # ---- Instantiate clients ----
# if not SPITCH_API_KEY:
#     raise RuntimeError("SPITCH_API_KEY missing in environment")
# if not OPENROUTER_API_KEY:
#     raise RuntimeError("OPENROUTER_API_KEY missing in environment")
# if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
#     raise RuntimeError("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN required")

# spitch_client = Spitch(api_key=SPITCH_API_KEY)
# openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
# twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# # ---- App setup ----
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("streaming-assistant")
# app = FastAPI()
# os.makedirs(STATIC_DIR, exist_ok=True)
# app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# # ---- In-memory per-call state (use Redis for production multi-process) ----
# CALL_STATE: Dict[str, Dict[str, Any]] = {}

# # ---- Utilities ----
# def cleanup_static_files(max_age: int = MAX_FILE_AGE):
#     now = time.time()
#     for path in glob.glob(f"{STATIC_DIR}/*.wav"):
#         try:
#             if os.path.getmtime(path) < now - max_age:
#                 os.remove(path)
#                 logger.info("Deleted stale file: %s", path)
#         except Exception:
#             logger.exception("Failed to delete file: %s", path)

# def save_wav_pcm16(path: str, pcm_bytes: bytes, sample_rate: int = 8000, channels: int = 1):
#     with wave.open(path, "wb") as wf:
#         wf.setnchannels(channels)
#         wf.setsampwidth(2)  # 16-bit
#         wf.setframerate(sample_rate)
#         wf.writeframes(pcm_bytes)

# def decode_twilio_media_event(frame: dict) -> bytes:
#     payload_b64 = frame["media"]["payload"]
#     return base64.b64decode(payload_b64)

# def energy_of_pcm16(pcm_bytes: bytes) -> float:
#     if not pcm_bytes:
#         return 0.0
#     arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
#     if arr.size == 0:
#         return 0.0
#     rms = np.sqrt(np.mean(arr * arr)) / 32768.0
#     return float(rms)

# def clean_language_code(lang: Optional[str]) -> Optional[str]:
#     if not lang:
#         return None
#     cleaned = "".join([c for c in lang.strip().lower() if c.isalpha()])
#     return cleaned if cleaned in VALID_LANGUAGES else None

# def save_bytes_to_static(bytes_data: bytes, filename_prefix: str) -> str:
#     fname = f"{filename_prefix}-{uuid.uuid4().hex}.wav"
#     path = os.path.join(STATIC_DIR, fname)
#     with open(path, "wb") as f:
#         f.write(bytes_data)
#         f.flush()
#         os.fsync(f.fileno())
#     return fname  # relative name under STATIC_DIR

# # ---- Provider wrappers (implemented using SDK calls) ----
# def spitch_transcribe_bytes(audio_bytes: bytes, language: str = "en") -> str:
#     """
#     Uses spitch_client.speech.transcribe. Expects bytes of an audio file (wav or raw) acceptable by Spitch.
#     """
#     try:
#         # The SDK call shape used earlier: spitch_client.speech.transcribe(content=audio_io, language=lang)
#         audio_io = io.BytesIO(audio_bytes)
#         transcription = spitch_client.speech.transcribe(content=audio_io, language=language)
#         text = getattr(transcription, "text", None)
#         if not text:
#             raise RuntimeError("Empty transcription returned from Spitch")
#         return text
#     except Exception as e:
#         logger.exception("Spitch transcription error: %s", e)
#         raise

# def spitch_translate(text: str, source: str, target: str) -> str:
#     try:
#         resp = spitch_client.text.translate(text=text, source=source, target=target)
#         t = getattr(resp, "text", None)
#         if not t:
#             raise RuntimeError("Empty translation from Spitch")
#         return t
#     except Exception as e:
#         logger.exception("Spitch translation error: %s", e)
#         raise

# def spitch_tts_bytes(text: str, language: str = "en", voice: Optional[str] = None) -> bytes:
#     try:
#         if not voice:
#             voice = VOICE_MAP.get(language, DEFAULT_VOICE)
#         tts = spitch_client.speech.generate(text=text, language=language, voice=voice)
#         data = tts.read()
#         if not data:
#             raise RuntimeError("Empty TTS output from Spitch")
#         return data
#     except Exception as e:
#         logger.exception("Spitch TTS error: %s", e)
#         raise

# def openrouter_chat_reply(messages: list) -> str:
#     try:
#         resp = openrouter_client.chat.completions.create(model=MODEL, messages=messages)
#         # defensive extraction
#         reply = resp.choices[0].message.content
#         if reply is None:
#             raise RuntimeError("Model returned empty reply")
#         return reply
#     except Exception as e:
#         logger.exception("OpenRouter model error: %s", e)
#         raise

# # ---- TwiML start endpoint: instruct Twilio to start media stream to our WS ----
# @app.post("/start_call")
# async def start_call():
#     """
#     Twilio will POST here when a call arrives. We return TwiML that starts a Media Stream
#     to our websocket and plays an optional prompt.
#     """
#     try:
#         twiml = VoiceResponse()
#         start = Start()
#         # Twilio expects a wss url. Convert BASE_URL -> wss://...
#         if BASE_URL.startswith("https://"):
#             wss = BASE_URL.replace("https://", "wss://")
#         elif BASE_URL.startswith("http://"):
#             wss = BASE_URL.replace("http://", "ws://")
#         else:
#             wss = BASE_URL
#         wss_url = f"{wss}/ws/twilio_stream"
#         start.stream(url=wss_url)
#         twiml.append(start)

#         # Generate a short TTS greeting to play (non-essential; stream immediately receives audio)
#         try:
#             tts_bytes = spitch_tts_bytes("Welcome. You can speak now.", language="en", voice=VOICE_MAP.get("en"))
#             fname = save_bytes_to_static(tts_bytes, "greeting")
#             twiml.play(f"{BASE_URL}/static/{fname}")
#         except Exception:
#             # If TTS fails, still return the stream twiml (call continues)
#             logger.exception("Greeting TTS failed; continuing without greeting")

#         return Response(content=str(twiml), media_type="application/xml")
#     except Exception as e:
#         logger.exception("start_call failed: %s", e)
#         raise HTTPException(status_code=500, detail=str(e))

# # ---- WebSocket to receive Twilio Media Streams ----
# @app.websocket("/ws/twilio_stream")
# async def twilio_stream_ws(websocket: WebSocket):
#     await websocket.accept()
#     logger.info("WebSocket accepted connection")
#     call_sid = None
#     try:
#         async for raw_msg in websocket.iter_text():
#             frame = json.loads(raw_msg)
#             event = frame.get("event")
#             if event == "connected":
#                 logger.info("Media stream connected: %s", frame.get("streamSid"))
#                 continue
#             if event == "start":
#                 call_sid = frame.get("start", {}).get("callSid")
#                 logger.info("Stream start for callSid=%s", call_sid)
#                 # initialize per-call state
#                 CALL_STATE[call_sid] = {
#                     "buffer": bytearray(),
#                     "last_audio_ts": time.time(),
#                     "lang": "en",  # default; later can be changed with detection if needed
#                     "processing": False
#                 }
#                 continue
#             if event == "media":
#                 # decode and append payload (base64)
#                 try:
#                     pcm = decode_twilio_media_event(frame)
#                 except Exception:
#                     logger.exception("Failed to decode media frame")
#                     continue

#                 if not call_sid:
#                     # attempt to extract callSid from nested fields if missing
#                     call_sid = frame.get("start", {}).get("callSid") or frame.get("connection", {}).get("callSid")
#                     if not call_sid:
#                         logger.warning("media frame received with no callSid; skipping")
#                         continue
#                     CALL_STATE.setdefault(call_sid, {"buffer": bytearray(), "last_audio_ts": time.time(), "lang": "en", "processing": False})

#                 state = CALL_STATE[call_sid]
#                 state["buffer"].extend(pcm)
#                 state["last_audio_ts"] = time.time()

#                 # if not already processing, schedule a worker to wait for silence and then process
#                 if not state["processing"]:
#                     state["processing"] = True
#                     asyncio.create_task(_utterance_worker(call_sid))
#                 continue
#             if event == "stop":
#                 call_sid = frame.get("start", {}).get("callSid")
#                 logger.info("Stream stop for callSid=%s", call_sid)
#                 if call_sid and call_sid in CALL_STATE:
#                     # process remaining buffer and cleanup state
#                     if CALL_STATE[call_sid]["buffer"]:
#                         asyncio.create_task(_process_buffer_and_reply(call_sid))
#                     asyncio.create_task(_delayed_cleanup(call_sid, delay=2.0))
#                 continue
#             # log other events
#             logger.debug("Unhandled media event: %s", event)
#     except Exception:
#         logger.exception("Error in websocket handler")
#     finally:
#         try:
#             await websocket.close()
#         except Exception:
#             pass
#         logger.info("WebSocket closed for callSid=%s", call_sid)

# # ---- background helpers ----
# async def _utterance_worker(call_sid: str):
#     """Wait until no audio for SILENCE_SECONDS, then process utterance."""
#     try:
#         while True:
#             now = time.time()
#             state = CALL_STATE.get(call_sid)
#             if not state:
#                 logger.debug("No state in utterance worker for %s", call_sid)
#                 return
#             last = state["last_audio_ts"]
#             if now - last >= SILENCE_SECONDS:
#                 # silence detected
#                 await _process_buffer_and_reply(call_sid)
#                 state["processing"] = False
#                 return
#             await asyncio.sleep(0.2)
#     except Exception:
#         logger.exception("_utterance_worker error")
#         if call_sid in CALL_STATE:
#             CALL_STATE[call_sid]["processing"] = False

# async def _process_buffer_and_reply(call_sid: str):
#     """
#     Save buffer to wav, transcribe, call model, TTS, save TTS, and update Twilio call to play TTS.
#     """
#     try:
#         state = CALL_STATE.get(call_sid)
#         if not state:
#             logger.warning("No call state for %s", call_sid)
#             return
#         buf = bytes(state["buffer"])
#         if not buf:
#             logger.info("Empty buffer for %s - nothing to process", call_sid)
#             return

#         uid = uuid.uuid4().hex
#         wav_filename = f"{call_sid}-{uid}.wav"
#         wav_path = os.path.join(STATIC_DIR, wav_filename)
#         # save raw PCM16LE bytes to wav (Twilio streams are PCM16LE)
#         save_wav_pcm16(wav_path, buf, sample_rate=8000, channels=1)
#         logger.info("Saved incoming utterance to %s", wav_path)

#         # clear buffer for next utterance
#         state["buffer"] = bytearray()

#         # 1) Transcribe using Spitch
#         try:
#             with open(wav_path, "rb") as f:
#                 wav_bytes = f.read()
#             transcribed_text = spitch_transcribe_bytes(wav_bytes, language=state.get("lang", "en"))
#             logger.info("Transcribed text for %s: %s", call_sid, transcribed_text)
#         except Exception:
#             logger.exception("Transcription failed for %s", call_sid)
#             return

#         # 2) Use OpenRouter (Mistral) to generate a reply
#         try:
#             messages = [
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": transcribed_text}
#             ]
#             reply_en = openrouter_chat_reply(messages)
#             logger.info("Model reply for %s: %s", call_sid, reply_en)
#         except Exception:
#             logger.exception("Model call failed; using fallback reply")
#             reply_en = "Sorry, I couldn't come up with a response."

#         # 3) If caller's language != en, translate; but we assume TTS supports same language from reply_en
#         # Optionally translate reply_en to state['lang'] using spitch_translate()
#         user_lang = state.get("lang", "en")
#         if user_lang != "en":
#             try:
#                 reply_local = spitch_translate(reply_en, source="en", target=user_lang)
#             except Exception:
#                 logger.exception("Translation failed; falling back to English")
#                 reply_local = reply_en
#         else:
#             reply_local = reply_en

#         # 4) Generate TTS via Spitch
#         try:
#             tts_bytes = spitch_tts_bytes(reply_local, language=user_lang, voice=VOICE_MAP.get(user_lang, DEFAULT_VOICE))
#             tts_fname = save_bytes_to_static(tts_bytes, f"{call_sid}-reply")
#             logger.info("Saved TTS reply %s", tts_fname)
#         except Exception:
#             logger.exception("TTS generation failed")
#             return

#         # 5) Update Twilio call to play the TTS file and then return control to the stream
#         try:
#             twiml = VoiceResponse()
#             twiml.play(f"{BASE_URL}/static/{tts_fname}")
#             # After playing, instruct Twilio to resume the existing TwiML (the stream will continue)
#             update_resp = twilio_client.calls(call_sid).update(twiml=str(twiml))
#             logger.info("Twilio call %s updated to play TTS reply (update sid=%s)", call_sid, getattr(update_resp, "sid", None))
#         except Exception:
#             logger.exception("Failed to update Twilio call to play TTS")

#         # housekeeping
#         cleanup_static_files()
#     except Exception:
#         logger.exception("_process_buffer_and_reply top-level error for %s", call_sid)

# async def _delayed_cleanup(call_sid: str, delay: float = 2.0):
#     await asyncio.sleep(delay)
#     if call_sid in CALL_STATE:
#         try:
#             del CALL_STATE[call_sid]
#             logger.info("Cleared in-memory state for %s", call_sid)
#         except Exception:
#             logger.exception("Failed to clear state for %s", call_sid)

# # ---- Health/debug endpoints ----
# @app.get("/health")
# async def health():
#     return {"status": "ok"}

# # ---- Notes for deployment ----
# # 1) Ensure your Twilio phone number's Voice webhook points to /start_call (POST).
# # 2) Set BASE_URL to the public https host where this app is deployed so Twilio can reach static audio.
# # 3) For production scale:
# #    - Replace CALL_STATE with Redis or another shared store across processes.
# #    - Move static audio to S3 and serve via a public URL (CDN) instead of local disk if running multiple workers.
# #    - Consider using webrtcvad for more robust voice activity detection.
# # 4) Confirm Twilio Media Streams audio config matches assumptions (PCM16LE 8kHz mono). Adjust sample_rate if needed.













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

# # def clean_language_code(lang: str) -> str:
# #     """Clean language code to ensure it's a valid ISO 639-1 code."""
# #     if not lang:
# #         return None
# #     # Remove quotes, whitespace, and non-lowercase letters
# #     cleaned = re.sub(r'[^a-z]', '', lang.strip().lower())
# #     return cleaned if cleaned in VALID_LANGUAGES else None

# @app.post("/start_call")
# async def start_call():
#     """Asks user for preferred language and allows interruption."""
#     try:
#         text = "What language do you want to speak in?"
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
#             timeout=5
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
#             cleaned_language = "er" #clean_language_code(language) 
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
#             # detected_lang = mistral_response.choices[0].message.content.strip().lower()
#             # cleaned = clean_language_code(detected_lang)

#             # # Validate detected language
#             # if cleaned not in VALID_LANGUAGES:
#             #     logger.error(f"Invalid language detected: {cleaned}")
#             #     raise HTTPException(status_code=400, detail=f"Detected language '{cleaned}' is not supported. Supported: {VALID_LANGUAGES}")
#             cleaned = "ig"
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