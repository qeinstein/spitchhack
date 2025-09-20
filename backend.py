import os
import logging
from typing import Dict, Any
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Start, Stream
from twilio.request_validator import RequestValidator
from dotenv import load_dotenv
from spitch import Spitch
from openai import OpenAI
from urllib.parse import urlparse

# ---- Config ----
load_dotenv()

# Validate environment variables
required_vars = [
    "SPITCH_API_KEY",
    "OPENROUTER_API_KEY",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "BASE_URL",
    "CONVERSATION_SERVICE_SID"
]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing environment variable: {var}")

SPITCH_API_KEY = os.getenv("SPITCH_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
MODEL = os.getenv("MODEL", "gpt-4o-mini")

# ---- Clients ----
try:
    spitch_client = Spitch(api_key=SPITCH_API_KEY)
    openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to initialize clients: {e}")

# ---- App setup ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("conversation-relay")
app = FastAPI()
twilio_validator = RequestValidator(TWILIO_AUTH_TOKEN)

# ---- Language map ----
LANGUAGE_MAP = {
    "1": ("Yoruba", "yo"),
    "2": ("Igbo", "ig"),
    "3": ("Hausa", "ha"),
    "4": ("English", "en")
}

LANGUAGE_SELECTION: Dict[str, str] = {}  # CallSid -> lang code

# ---- Helpers ----
def spitch_translate(text: str, source: str, target: str) -> str:
    """
    Translate `text` from `source` language to `target` using Spitch API.
    """
    try:
        resp = spitch_client.text.translate(text=text, source=source, target=target)
        t = getattr(resp, "text", None)
        if not t:
            raise RuntimeError("Empty translation from Spitch")
        return t
    except Exception as e:
        logger.error(f"Spitch translation failed: {e}")
        raise RuntimeError(f"Translation error: {e}")

def openrouter_chat_reply(messages: list) -> str:
    try:
        resp = openrouter_client.chat.completions.create(model=MODEL, messages=messages)
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenRouter API error: {e}")
        return "Sorry, I couldn't process your request. Please try again."

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

    lang_name, lang_code = LANGUAGE_MAP[Digits]
    LANGUAGE_SELECTION[CallSid] = lang_code
    logger.info("Language set for CallSid %s -> %s", CallSid, lang_code)

    twiml.say(f"You selected {lang_name}. Connecting you now.")

    # Fixing BASE_URL parsing
    parsed = urlparse(BASE_URL)
    host = parsed.netloc or parsed.path  # in case BASE_URL had no scheme
    # Use websocket URL derived from BASE_URL
    ws_url = f"wss://{host}/relay"

    start = Start()
    stream = Stream(url=ws_url)
    start.append(stream)
    twiml.append(start)

    return Response(content=str(twiml), media_type="application/xml")


# ---- Health check ----
@app.get("/health")
async def health():
    status = {"status": "ok", "services": {}}
    # Test Spitch translate
    try:
        _ = spitch_translate("test", "en", "en")
        status["services"]["spitch"] = "ok"
    except Exception as e:
        status["services"]["spitch"] = f"down: {e}"
    # Test OpenRouter
    try:
        _ = openrouter_client.chat.completions.create(model=MODEL, messages=[{"role": "system", "content": "test"}])
        status["services"]["openrouter"] = "ok"
    except Exception as e:
        status["services"]["openrouter"] = f"down: {e}"
    return status

from fastapi import WebSocket

@app.websocket("/relay")
async def relay_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            logger.debug("WebSocket event: %s", data)
            event_type = data.get("type")
            call_sid = data.get("callSid")
            if not call_sid:
                logger.error("Missing callSid in event")
                await websocket.send_json({"type": "noop"})
                continue

            if event_type == "call_ended":
                LANGUAGE_SELECTION.pop(call_sid, None)
                logger.info("Cleaned up LANGUAGE_SELECTION for CallSid %s", call_sid)
                await websocket.send_json({"type": "noop"})
                continue

            if event_type == "utterance":
                user_text = data.get("text")
                if not user_text or not user_text.strip():
                    logger.error("Missing or empty text in utterance event")
                    await websocket.send_json({"type": "noop"})
                    continue

                # Optionally, filter overly repetitive or irrelevant utterances
                # Could track repeat counts per call_sid if needed

                lang = LANGUAGE_SELECTION.get(call_sid, "en")

                try:
                    if lang != "en":
                        english_text = spitch_translate(user_text, source=lang, target="en")
                    else:
                        english_text = user_text

                    reply_en = openrouter_chat_reply([
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": english_text}
                    ])

                    if lang != "en":
                        reply_local = spitch_translate(reply_en, source="en", target=lang)
                    else:
                        reply_local = reply_en

                    await websocket.send_json({"type": "reply", "text": reply_local})

                except Exception as e:
                    logger.error(f"Error processing utterance: {e}")
                    await websocket.send_json({"type": "reply", "text": "Sorry, an error occurred. Please try again."})

            else:
                await websocket.send_json({"type": "noop"})

    except Exception as e:
        logger.error("WebSocket error: %s", e)
    finally:
        await websocket.close()









#this particular code works, the one above is juat an optinized version from grok
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

# import numpy as np
# from dotenv import load_dotenv
# from fastapi import FastAPI, WebSocket, Form, Request
# from fastapi.responses import Response
# from fastapi.staticfiles import StaticFiles
# from twilio.twiml.voice_response import VoiceResponse, Gather, Start
# from twilio.rest import Client as TwilioClient

# # provider SDKs (must be installed and configured)
# from spitch import Spitch
# from openai import OpenAI  # OpenRouter-compatible client

# load_dotenv()

# # ---- Config / env ----
# SPITCH_API_KEY = os.getenv("SPITCH_API_KEY")
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
# TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
# BASE_URL = os.getenv("BASE_URL", "").rstrip("/")  # must be a public HTTPS URL
# MODEL = os.getenv("MODEL")
# STATIC_DIR = os.getenv("STATIC_DIR", "static")
# SILENCE_SECONDS = float(os.getenv("SILENCE_SECONDS", "1.2"))
# SILENCE_ENERGY_THRESHOLD = float(os.getenv("SILENCE_ENERGY_THRESHOLD", "0.002"))
# MAX_FILE_AGE = int(os.getenv("MAX_FILE_AGE", "300"))

# if not BASE_URL:
#     raise RuntimeError("BASE_URL must be set (public HTTPS URL)")

# if not (SPITCH_API_KEY and OPENROUTER_API_KEY and TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN):
#     raise RuntimeError("Please set SPITCH_API_KEY, OPENROUTER_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN")

# # ---- Clients ----
# spitch_client = Spitch(api_key=SPITCH_API_KEY)
# openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
# twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# # ---- App setup ----
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("no-audioop-streaming")
# app = FastAPI()
# os.makedirs(STATIC_DIR, exist_ok=True)
# app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# # ---- Language map for IVR digits ----
# LANGUAGE_MAP = {
#     "1": ("Yoruba", "yo"),
#     "2": ("Igbo", "ig"),
#     "3": ("Hausa", "ha"),
#     "4": ("English", "en")
# }

# VOICE_MAP = {
#     "en": "jude",
#     "ha": "aliyu",
#     "yo": "femi",
#     "ig": "obinna",
#     "am": "default"
# }
# DEFAULT_VOICE = "jude"

# # ---- In-memory state ----
# LANGUAGE_SELECTION: Dict[str, str] = {}   # CallSid -> lang code
# STREAM_STATE: Dict[str, Dict[str, Any]] = {}  # streamSid -> state

# # ---- Utilities ----
# def cleanup_static_files(max_age: int = MAX_FILE_AGE):
#     now = time.time()
#     for path in glob.glob(os.path.join(STATIC_DIR, "*.wav")):
#         try:
#             if os.path.getmtime(path) < now - max_age:
#                 os.remove(path)
#                 logger.info("Deleted old file: %s", path)
#         except Exception:
#             logger.exception("Failed to delete file: %s", path)

# def save_wav_pcm16(path: str, pcm16_bytes: bytes, sample_rate: int = 8000, channels: int = 1):
#     with wave.open(path, "wb") as wf:
#         wf.setnchannels(channels)
#         wf.setsampwidth(2)
#         wf.setframerate(sample_rate)
#         wf.writeframes(pcm16_bytes)

# def read_wav_bytes_get_pcm16(wav_bytes: bytes):
#     """Read WAV bytes and return (pcm16_bytes, sample_rate, channels)."""
#     with io.BytesIO(wav_bytes) as b:
#         with wave.open(b, "rb") as wf:
#             channels = wf.getnchannels()
#             sr = wf.getframerate()
#             sampwidth = wf.getsampwidth()
#             frames = wf.readframes(wf.getnframes())
#             # convert sample width to 2 if needed
#             if sampwidth != 2:
#                 # convert to 16-bit by scaling
#                 # this is a simple fallback and assumes integer PCM input
#                 arr = np.frombuffer(frames, dtype=np.uint8)
#                 # best effort: leave as-is if unexpected; prefer Spitch to return 16-bit WAV
#                 frames = audio_bytes_to_int16(frames, sampwidth)
#             if channels != 1:
#                 # mix down by averaging channels
#                 arr = np.frombuffer(frames, dtype=np.int16)
#                 arr = arr.reshape(-1, channels)
#                 mono = arr.mean(axis=1).astype(np.int16)
#                 return mono.tobytes(), sr, 1
#             return frames, sr, channels

# def audio_bytes_to_int16(raw: bytes, sampwidth: int) -> bytes:
#     """Naive converter from arbitrary integer sample width to int16 bytes.
#     Only called as fallback. For high quality prefer 16-bit WAV input from TTS."""
#     if sampwidth == 1:
#         # 8-bit unsigned PCM -> centered signed 16-bit
#         arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
#         arr = (arr - 128) * 256
#         return arr.tobytes()
#     elif sampwidth == 3:
#         # 24-bit little-endian to 16-bit: drop lowest byte
#         arr = np.frombuffer(raw, dtype=np.uint8)
#         arr = arr.reshape(-1, 3)
#         # take most significant two bytes (little-endian) -> combine
#         arr16 = (arr[:, 2].astype(np.int16) << 8) | arr[:, 1].astype(np.int16)
#         return arr16.tobytes()
#     else:
#         # fallback: try interpreting as int16
#         return raw

# # ---- µ-law encode/decode (pure Python) ----
# # Standard ITU G.711 µ-law implementation

# MU = 255
# BIAS = 0x84  # 132

# def pcm16_to_mulaw_bytes(pcm16_bytes: bytes) -> bytes:
#     """Convert PCM16LE bytes to mu-law bytes (8-bit)."""
#     samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.int32)
#     # Clip to 16-bit signed range (redundant)
#     samples = np.clip(samples, -32768, 32767)
#     # convert to magnitude and sign
#     sign = (samples >> 8) & 0x80  # sign bit for mu-law
#     magnitude = np.abs(samples)
#     magnitude = magnitude + BIAS
#     # get exponent and mantissa
#     exponent = np.floor(np.log2(magnitude + 1)).astype(np.int32)
#     # limit exponent to 7
#     exponent = np.minimum(exponent, 7)
#     mantissa = (magnitude >> (exponent + 3)) & 0x0F
#     mulaw = ~(sign | (exponent << 4) | mantissa) & 0xFF
#     return mulaw.astype(np.uint8).tobytes()

# def mulaw_to_pcm16_bytes(mulaw_bytes: bytes) -> bytes:
#     """Convert mu-law bytes to PCM16LE bytes."""
#     mu = np.frombuffer(mulaw_bytes, dtype=np.uint8)
#     mu = ~mu & 0xFF
#     sign = mu & 0x80
#     exponent = (mu >> 4) & 0x07
#     mantissa = mu & 0x0F
#     magnitude = ((mantissa << (exponent + 3)) + (1 << (exponent + 3)) - BIAS)
#     pcm = magnitude.astype(np.int32)
#     pcm = pcm * (~(sign - 1))  # apply sign; bit trick: if sign==0 -> +, else -> - ; simpler to do sign mask
#     # simpler: reconstruct sample with sign
#     pcm_signed = np.where(sign == 0, pcm, -pcm)
#     pcm_signed = np.clip(pcm_signed, -32768, 32767).astype(np.int16)
#     return pcm_signed.tobytes()

# # Note: The above µ-law implementation uses vectorized ops. It is not
# # byte-perfect to every reference codec edge-case but is broadly compatible.
# # If you need bit-exact G.711 behavior, use a tested library; this avoids audioop.

# # ---- Resampling (simple linear interpolation using numpy) ----
# def resample_pcm16(pcm16_bytes: bytes, src_rate: int, tgt_rate: int = 8000) -> bytes:
#     """Resample PCM16LE bytes from src_rate -> tgt_rate using numpy interpolation."""
#     if src_rate == tgt_rate:
#         return pcm16_bytes
#     arr = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
#     if arr.size == 0:
#         return b""
#     duration = arr.shape[0] / src_rate
#     new_len = int(np.round(duration * tgt_rate))
#     if new_len <= 0:
#         return b""
#     old_idx = np.linspace(0, arr.shape[0] - 1, num=arr.shape[0])
#     new_idx = np.linspace(0, arr.shape[0] - 1, num=new_len)
#     new_arr = np.interp(new_idx, old_idx, arr).astype(np.int16)
#     return new_arr.tobytes()

# # ---- Signal energy (RMS) ----
# def energy_of_pcm16(pcm16_bytes: bytes) -> float:
#     if not pcm16_bytes:
#         return 0.0
#     arr = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
#     if arr.size == 0:
#         return 0.0
#     rms = np.sqrt(np.mean(arr * arr)) / 32768.0
#     return float(rms)

# # ---- Provider wrappers using SDK calls ----
# def spitch_transcribe_wav_bytes(wav_bytes: bytes, language: str = "en") -> str:
#     audio_io = io.BytesIO(wav_bytes)
#     transcription = spitch_client.speech.transcribe(content=audio_io, language=language)
#     text = getattr(transcription, "text", None)
#     if not text:
#         raise RuntimeError("Empty transcription from Spitch")
#     return text

# def spitch_translate(text: str, source: str, target: str) -> str:
#     resp = spitch_client.text.translate(text=text, source=source, target=target)
#     t = getattr(resp, "text", None)
#     if not t:
#         raise RuntimeError("Empty translation from Spitch")
#     return t

# def spitch_tts_wav_bytes(text: str, language: str = "en", voice: Optional[str] = None) -> bytes:
#     if not voice:
#         voice = VOICE_MAP.get(language, DEFAULT_VOICE)
#     tts = spitch_client.speech.generate(text=text, language=language, voice=voice)
#     data = tts.read()
#     if not data:
#         raise RuntimeError("Empty TTS bytes from Spitch")
#     return data

# def openrouter_chat_reply(messages: list) -> str:
#     resp = openrouter_client.chat.completions.create(model=MODEL, messages=messages)
#     try:
#         return resp.choices[0].message.content
#     except Exception as e:
#         raise RuntimeError(f"Model returned invalid reply: {e}")

# # ---- IVR endpoints ----
# @app.post("/voice")
# async def voice_entry():
#     twiml = VoiceResponse()
#     gather = Gather(num_digits=1, action="/process_language", method="POST", timeout=8)
#     gather.say("Welcome to Proxy. For Yoruba press 1. For Igbo press 2. For Hausa press 3. For English press 4.")
#     twiml.append(gather)
#     twiml.say("Sorry, we did not receive input.")
#     twiml.redirect("/voice")
#     return Response(content=str(twiml), media_type="application/xml")

# @app.post("/process_language")
# async def process_language(Digits: str = Form(None), CallSid: str = Form(None)):
#     twiml = VoiceResponse()
#     if not Digits or Digits not in LANGUAGE_MAP:
#         twiml.say("Invalid selection. Please try again.")
#         twiml.redirect("/voice")
#         return Response(content=str(twiml), media_type="application/xml")

#     lang_name, lang_code = LANGUAGE_MAP[Digits]
#     if CallSid:
#         LANGUAGE_SELECTION[CallSid] = lang_code
#         logger.info("Language for CallSid %s set to %s", CallSid, lang_code)
#     else:
#         logger.warning("No CallSid in /process_language request; language will not be persisted for stream")

#     # generate confirmation TTS using Spitch and play it via Twilio <Play>
#     confirm_text = f"You selected {lang_name}. Connecting you to the assistant now."
#     try:
#         tts_bytes = spitch_tts_wav_bytes(confirm_text, language=lang_code)
#         fname = f"confirm-{lang_code}-{uuid.uuid4().hex}.wav"
#         path = os.path.join(STATIC_DIR, fname)
#         with open(path, "wb") as f:
#             f.write(tts_bytes)
#             f.flush()
#             os.fsync(f.fileno())
#         twiml.play(f"{BASE_URL}/static/{fname}")
#     except Exception:
#         logger.exception("TTS confirmation failed; falling back to say()")
#         twiml.say(f"You selected {lang_name}. Connecting now.")

#     # Start Media Stream to our websocket
#     if BASE_URL.startswith("https://"):
#         wss = BASE_URL.replace("https://", "wss://")
#     elif BASE_URL.startswith("http://"):
#         wss = BASE_URL.replace("http://", "ws://")
#     else:
#         wss = BASE_URL
#     stream_url = f"{wss}/ws/twilio_stream"
#     start = Start()
#     start.stream(url=stream_url)
#     twiml.append(start)
#     twiml.pause(length=60)
#     cleanup_static_files()
#     return Response(content=str(twiml), media_type="application/xml")

# # ---- WebSocket handler ----
# @app.websocket("/ws/twilio_stream")
# async def ws_twilio_stream(ws: WebSocket):
#     await ws.accept()
#     logger.info("WebSocket accepted (Twilio Media Stream)")
#     stream_sid = None
#     call_sid = None
#     try:
#         async for raw in ws.iter_text():
#             frame = json.loads(raw)
#             event = frame.get("event")
#             if event == "connected":
#                 logger.info("Twilio media stream connected")
#                 continue
#             if event == "start":
#                 start = frame.get("start", {})
#                 stream_sid = start.get("streamSid")
#                 call_sid = start.get("callSid")
#                 logger.info("Stream start: streamSid=%s callSid=%s", stream_sid, call_sid)
#                 lang = LANGUAGE_SELECTION.pop(call_sid, None) or "en"
#                 STREAM_STATE[stream_sid] = {
#                     "callSid": call_sid,
#                     "websocket": ws,
#                     "ws_send_lock": asyncio.Lock(),
#                     "buffer": bytearray(),
#                     "last_audio_ts": time.time(),
#                     "processing": False,
#                     "lang": lang,
#                     "streamSid": stream_sid
#                 }
#                 continue
#             if event == "media":
#                 media = frame.get("media", {})
#                 payload_b64 = media.get("payload")
#                 sid = frame.get("streamSid") or stream_sid
#                 if not payload_b64 or not sid or sid not in STREAM_STATE:
#                     continue
#                 try:
#                     ulaw = base64.b64decode(payload_b64)
#                 except Exception:
#                     logger.exception("Failed base64 decode")
#                     continue
#                 # convert mu-law -> PCM16 (bytes) WITHOUT audioop
#                 try:
#                     pcm16 = mulaw_to_pcm16_bytes(ulaw)
#                 except Exception:
#                     logger.exception("mu-law -> PCM16 conversion failed")
#                     continue
#                 state = STREAM_STATE[sid]
#                 state["buffer"].extend(pcm16)
#                 state["last_audio_ts"] = time.time()
#                 if not state["processing"]:
#                     state["processing"] = True
#                     asyncio.create_task(_utterance_silence_waiter(sid))
#                 continue
#             if event == "stop":
#                 sid = frame.get("streamSid")
#                 logger.info("Stream stop for %s", sid)
#                 if sid and sid in STREAM_STATE and STREAM_STATE[sid]["buffer"]:
#                     asyncio.create_task(_process_and_reply(sid))
#                 if sid:
#                     asyncio.create_task(_cleanup_stream_state(sid, delay=2.0))
#                 continue
#             if event == "mark":
#                 logger.info("Received mark event: %s", frame.get("mark"))
#                 continue
#             logger.debug("Unhandled event: %s", event)
#     except Exception:
#         logger.exception("WebSocket handler error")
#     finally:
#         logger.info("WebSocket closed (streamSid=%s callSid=%s)", stream_sid, call_sid)
#         try:
#             await ws.close()
#         except Exception:
#             pass
#         if stream_sid and stream_sid in STREAM_STATE:
#             try:
#                 del STREAM_STATE[stream_sid]
#             except Exception:
#                 pass

# # ---- silence waiter & processing ----
# async def _utterance_silence_waiter(streamSid: str):
#     try:
#         while True:
#             state = STREAM_STATE.get(streamSid)
#             if not state:
#                 return
#             elapsed = time.time() - state["last_audio_ts"]
#             if elapsed >= SILENCE_SECONDS:
#                 await _process_and_reply(streamSid)
#                 st = STREAM_STATE.get(streamSid)
#                 if st:
#                     st["processing"] = False
#                 return
#             await asyncio.sleep(0.15)
#     except Exception:
#         logger.exception("_utterance_silence_waiter error for %s", streamSid)
#         if streamSid in STREAM_STATE:
#             STREAM_STATE[streamSid]["processing"] = False

# async def _process_and_reply(streamSid: str):
#     try:
#         state = STREAM_STATE.get(streamSid)
#         if not state:
#             logger.warning("process_and_reply: unknown stream %s", streamSid)
#             return
#         buf = bytes(state["buffer"])
#         if not buf:
#             logger.info("Empty buffer for %s", streamSid)
#             return

#         uid = uuid.uuid4().hex
#         in_wav = f"{streamSid}-{uid}.wav"
#         in_path = os.path.join(STATIC_DIR, in_wav)
#         save_wav_pcm16(in_path, buf, sample_rate=8000, channels=1)
#         logger.info("Saved incoming utterance: %s", in_path)
#         state["buffer"] = bytearray()

#         # Transcribe using Spitch; prefer selected language as STT if available
#         lang = state.get("lang", "en")
#         try:
#             with open(in_path, "rb") as f:
#                 wav_bytes = f.read()
#             transcribed = spitch_transcribe_wav_bytes(wav_bytes, language=lang)
#             logger.info("Transcribed (lang=%s): %s", lang, transcribed)
#         except Exception:
#             logger.exception("Transcription failed; trying 'en'")
#             try:
#                 transcribed = spitch_transcribe_wav_bytes(wav_bytes, language="en")
#             except Exception:
#                 logger.exception("Transcription ultimately failed")
#                 return

#         # Translate to English for model
#         if lang != "en":
#             try:
#                 english_text = spitch_translate(transcribed, source=lang, target="en")
#             except Exception:
#                 logger.exception("Translation to English failed; using raw transcription")
#                 english_text = transcribed
#         else:
#             english_text = transcribed

#         # Model reply
#         try:
#             messages = [
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": english_text}
#             ]
#             reply_en = openrouter_chat_reply(messages)
#         except Exception:
#             logger.exception("Model call failed; using fallback")
#             reply_en = "Sorry, I couldn't process that."

#         # Translate back to user language
#         if lang != "en":
#             try:
#                 reply_local = spitch_translate(reply_en, source="en", target=lang)
#             except Exception:
#                 logger.exception("Translation back failed; using English reply")
#                 reply_local = reply_en
#         else:
#             reply_local = reply_en

#         # Generate TTS via Spitch
#         try:
#             tts_wav_bytes = spitch_tts_wav_bytes(reply_local, language=lang)
#         except Exception:
#             logger.exception("TTS generation failed")
#             return

#         # Parse TTS WAV -> PCM16 + sr
#         try:
#             pcm16_bytes, sr, channels = read_wav_bytes_get_pcm16(tts_wav_bytes)
#         except Exception:
#             logger.exception("Failed to read TTS WAV bytes")
#             return

#         # Resample to 8000 Hz if needed
#         try:
#             pcm16_8k = resample_pcm16(pcm16_bytes, src_rate=sr, tgt_rate=8000)
#         except Exception:
#             logger.exception("Resample failed; using original PCM")
#             pcm16_8k = pcm16_bytes

#         # Convert PCM16 -> mu-law bytes (G.711 u-law)
#         try:
#             mulaw_bytes = pcm16_to_mulaw_bytes(pcm16_8k)
#         except Exception:
#             logger.exception("PCM -> mu-law conversion failed")
#             return

#         mulaw_b64 = base64.b64encode(mulaw_bytes).decode("ascii")
#         outbound = {"event": "media", "streamSid": streamSid, "media": {"payload": mulaw_b64}}
#         lock: asyncio.Lock = state["ws_send_lock"]
#         ws = state["websocket"]
#         async with lock:
#             try:
#                 await ws.send_text(json.dumps(outbound))
#                 logger.info("Sent media reply for %s (bytes=%d)", streamSid, len(mulaw_bytes))
#                 # send a mark event to know when playback ends
#                 mark_name = f"reply-{uuid.uuid4().hex[:8]}"
#                 mark_msg = {"event": "mark", "streamSid": streamSid, "mark": {"name": mark_name}}
#                 await ws.send_text(json.dumps(mark_msg))
#                 logger.info("Sent mark %s for stream %s", mark_name, streamSid)
#             except Exception:
#                 logger.exception("Failed to send media back to Twilio for %s", streamSid)

#         # save TTS for debugging
#         try:
#             fname = f"tts-{streamSid}-{uuid.uuid4().hex}.wav"
#             with open(os.path.join(STATIC_DIR, fname), "wb") as f:
#                 f.write(tts_wav_bytes)
#             logger.info("Saved TTS debug file: %s", fname)
#         except Exception:
#             logger.exception("Failed to save tts debug file")

#         cleanup_static_files()
#     except Exception:
#         logger.exception("Top-level error in _process_and_reply for %s", streamSid)

# async def _cleanup_stream_state(streamSid: str, delay: float = 2.0):
#     await asyncio.sleep(delay)
#     if streamSid in STREAM_STATE:
#         try:
#             del STREAM_STATE[streamSid]
#             logger.info("Cleared stream state for %s", streamSid)
#         except Exception:
#             logger.exception("Failed to cleanup stream state for %s", streamSid)

# # ---- health ----
# @app.get("/health")
# async def health():
#     return {"status": "ok"}

