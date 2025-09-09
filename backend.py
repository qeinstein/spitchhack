# app.py
import os
import json
import uuid
import base64
import logging
import asyncio
from pathlib import Path
from typing import Optional

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed

from fastapi import FastAPI, Request, WebSocket, HTTPException
from fastapi.responses import Response, FileResponse
from twilio.twiml.voice_response import VoiceResponse, Start, Gather
from twilio.rest import Client

# Official Spitch SDK import (per docs)
# Make sure you installed the SDK: pip install spitch
from spitch import Spitch

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wagwan-ai")

# ----------------------------
# ENV / CONFIG
# ----------------------------
SPITCH_API_KEY = os.getenv("SPITCH_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
BASE_URL = os.getenv("BASE_URL")
SPITCH_WS_URL = os.getenv("SPITCH_WS_URL", "wss://api.spitch.io/realtime")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")

if not all([SPITCH_API_KEY, OPENROUTER_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, BASE_URL]):
    logger.warning("Missing one or more required env vars: SPITCH_API_KEY, OPENROUTER_API_KEY, "
                   "TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, BASE_URL")

# Twilio client
TWILIO_CLIENT = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Spitch SDK client (per docs)
spitch_client = Spitch(api_key=SPITCH_API_KEY)

# audio temp dir
AUDIO_DIR = Path("/tmp/wagwan_audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# FastAPI App & State
# ----------------------------
app = FastAPI(title="Wagwan AI - Twilio + Spitch + OpenRouter")

class ConversationState:
    def __init__(self):
        self.language: str = "en"     # 'en','yo','ig','ha'
        self.call_sid: Optional[str] = None
        self.last_partial: str = ""
        self.transcripts: list[str] = []
        self.ai_task = None
        self.spitch_ws = None
        self.twilio_ws_connected = False

session = ConversationState()

# ----------------------------
# helpers
# ----------------------------
def map_choice_to_lang(digit: str) -> str:
    return {"1": "yo", "2": "ig", "3": "ha", "4": "en"}.get(digit, "en")

def save_audio_bytes(audio_bytes: bytes, ext: str = "wav") -> str:
    p = AUDIO_DIR / f"{uuid.uuid4().hex}.{ext}"
    p.write_bytes(audio_bytes)
    return str(p)

def build_audio_url(local_path: str) -> str:
    return f"{BASE_URL}/audio/{Path(local_path).name}"

# ----------------------------
# Twilio webhook: initial IVR
# ----------------------------
@app.post("/voice")
async def voice_entry(request: Request):
    twiml = VoiceResponse()
    gather = Gather(num_digits=1, action="/select_language", method="POST", timeout=8)
    gather.say("Welcome to Wagwan AI. Press 1 for Yoruba. 2 for Igbo. 3 for Hausa. 4 for English.")
    twiml.append(gather)
    twiml.redirect("/voice")
    return Response(content=str(twiml), media_type="application/xml")

# ----------------------------
# Process language selection & start streaming
# ----------------------------
@app.post("/select_language")
async def select_language(request: Request):
    form = await request.form()
    digits = form.get("Digits", "")
    call_sid = form.get("CallSid")
    if call_sid:
        session.call_sid = call_sid

    selected = map_choice_to_lang(digits)
    session.language = selected

    twiml = VoiceResponse()
    twiml.say(f"Okay, you selected {selected}. After the beep, start speaking.")
    # Start Twilio media stream to our /ws-audio websocket
    # Twilio expects a wss://<public-host>/ws-audio
    host_for_twilio = os.getenv("HOST_FOR_TWILIO") or BASE_URL.replace("https://","").replace("http://","")
    start = Start()
    start.stream(url=f"wss://{host_for_twilio}/ws-audio")
    twiml.append(start)
    # keep open; we'll use call update to play TTS and then resume
    twiml.pause(length=3600)
    return Response(content=str(twiml), media_type="application/xml")

# ----------------------------
# Twilio Media Streams websocket endpoint
# ----------------------------
@app.websocket("/ws-audio")
async def twilio_ws(ws: WebSocket):
    """
    Receive Twilio media frames (likely JSON with base64 payload) and forward audio bytes to Spitch realtime websocket.
    Also spawn a listener (spitch_listener) to read Spitch transcription messages.
    """
    await ws.accept()
    session.twilio_ws_connected = True
    logger.info("Twilio WS connected")

    try:
        async with websockets.connect(
            SPITCH_WS_URL,
            extra_headers={"Authorization": f"Bearer {SPITCH_API_KEY}"}
        ) as spitch_ws:
            session.spitch_ws = spitch_ws
            logger.info("Connected to Spitch realtime websocket")

            listener = asyncio.create_task(spitch_listener(spitch_ws))

            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.receive":
                    # Twilio Media Streams usually sends JSON events with media.payload base64
                    if "text" in msg:
                        try:
                            j = json.loads(msg["text"])
                            if j.get("event") == "media" and j.get("media",{}).get("payload"):
                                payload = j["media"]["payload"]
                                audio_bytes = base64.b64decode(payload)
                                # forward raw audio bytes to spitch realtime endpoint
                                await spitch_ws.send(audio_bytes)
                            else:
                                # handle start/stop events if needed
                                pass
                        except Exception:
                            # if it's not JSON, ignore
                            pass
                    elif "bytes" in msg:
                        # forward binary directly
                        await spitch_ws.send(msg["bytes"])
                elif msg["type"] == "websocket.disconnect":
                    logger.info("Twilio websocket disconnected")
                    break

            listener.cancel()

    except Exception as e:
        logger.error(f"Twilio WS error: {e}")
    finally:
        session.twilio_ws_connected = False
        session.spitch_ws = None
        logger.info("Twilio WS handler ending")

# ----------------------------
# Spitch realtime listener (parses partial & final transcripts)
# ----------------------------
async def spitch_listener(spitch_ws):
    """
    Listen for Spitch realtime transcription messages and trigger the AI pipeline on final transcripts.
    The docs show a JSON shape with `type: partial|final` and alternatives[0].transcript.
    """
    try:
        async for raw in spitch_ws:
            if isinstance(raw, (bytes, bytearray)):
                try:
                    raw = raw.decode()
                except Exception:
                    continue
            try:
                data = json.loads(raw)
            except Exception:
                continue

            t = data.get("type")
            if t == "partial":
                transcript = data.get("alternatives", [{}])[0].get("transcript", "")
                if transcript and transcript != session.last_partial:
                    session.last_partial = transcript
                    logger.info(f"Spitch partial: {transcript}")
                    # (optional) you could stream an early AI reply here
            elif t == "final":
                final_text = data.get("alternatives", [{}])[0].get("transcript", "")
                if final_text:
                    logger.info(f"Spitch final: {final_text}")
                    session.transcripts.append(final_text)
                    session.last_partial = ""
                    if session.ai_task is None:
                        session.ai_task = asyncio.create_task(handle_final_transcript(final_text))
            else:
                # ignore other message types
                pass

    except Exception as e:
        logger.error(f"Spitch listener error: {e}")

# ----------------------------
# End-to-end pipeline for a final user utterance
# ----------------------------
async def handle_final_transcript(user_text: str):
    """
    1) Query OpenRouter AI with the user's transcription
    2) Translate AI's English response to user's selected language using spitch_client.text.translate
    3) TTS the translated text using spitch_client.speech.generate (returns bytes)
    4) Save audio, expose at /audio/<file>, and instruct Twilio to play the audio by updating the call's TwiML
    5) Twilio will redirect to /resume_stream which restarts streaming
    """
    logger.info("Starting AI pipeline for final utterance")
    try:
        # 1) Query OpenRouter
        ai_text = await query_openrouter_ai(user_text)
        logger.info(f"AI text: {ai_text}")

        # 2) Translate to target language (if not English)
        if session.language != "en":
            try:
                # Spitch SDK text.translate usage per docs
                result = spitch_client.text.translate(text=ai_text, source="en", target=session.language)
                translated_text = getattr(result, "text", None) or result.get("text") if isinstance(result, dict) else None
                if not translated_text:
                    # fallback to ai_text if translation fails
                    translated_text = ai_text
                logger.info(f"Translated text -> {translated_text}")
            except Exception as e:
                logger.error(f"Spitch translate failed: {e}")
                translated_text = ai_text
        else:
            translated_text = ai_text

        # 3) TTS via Spitch SDK (speech.generate) - returns response with bytes
        try:
            # per docs: client.speech.generate(text=..., language='yo', voice='femi')
            # choose voice based on language; pick defaults or let user choose
            voice_map = {"yo": "femi", "ig": "emeka", "ha": "hauwa", "en": "sade"}
            voice = voice_map.get(session.language, "sade")
            # The SDK examples show returning a file-like response with .read() for bytes
            logger.info(f"Calling Spitch TTS: lang={session.language} voice={voice}")
            resp = spitch_client.speech.generate(text=translated_text, language=session.language, voice=voice)
            # SDK response may be a file-like object or an object with .read()
            if hasattr(resp, "read"):
                audio_bytes = resp.read()
            elif isinstance(resp, (bytes, bytearray)):
                audio_bytes = bytes(resp)
            elif isinstance(resp, dict) and resp.get("audio"):
                # some SDKs return base64 in json "audio" field
                audio_bytes = base64.b64decode(resp["audio"])
            else:
                logger.error("Unknown TTS response shape from Spitch SDK")
                audio_bytes = None
        except Exception as e:
            logger.error(f"Spitch TTS generation failed: {e}")
            audio_bytes = None

        if not audio_bytes:
            logger.error("No audio produced by TTS")
            return

        # 4) Save audio locally & get public URL
        audiopath = save_audio_bytes(audio_bytes, ext="wav")
        audio_url = build_audio_url(audiopath)
        logger.info(f"Saved TTS audio: {audio_url}")

        # 5) Update Twilio Call to Play the audio, then redirect to /resume_stream to continue streaming
        if session.call_sid:
            play_twiml = f"""<Response>
    <Play>{audio_url}</Play>
    <Redirect>/resume_stream</Redirect>
</Response>"""
            try:
                TWILIO_CLIENT.calls(session.call_sid).update(twiml=play_twiml)
                logger.info("Updated Twilio call to play TTS audio")
            except Exception as e:
                logger.error(f"Failed to update Twilio call: {e}")
        else:
            logger.warning("No call SID available to instruct Twilio")

    except Exception as e:
        logger.error(f"handle_final_transcript failed: {e}")
    finally:
        session.ai_task = None

# ----------------------------
# OpenRouter AI helper (non-streaming)
# ----------------------------
async def query_openrouter_ai(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.6,
    }
    async with aiohttp.ClientSession() as s:
        async with s.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"OpenRouter error {resp.status}: {text}")
                return "Sorry, I couldn't process that right now."
            j = await resp.json()
            # adapt to response shape
            choices = j.get("choices", [])
            if choices:
                first = choices[0]
                # common shapes:
                content = (first.get("message") or {}).get("content") if isinstance(first.get("message"), dict) else first.get("text")
                if content:
                    return content
            return j.get("text") or "Sorry, I couldn't get an answer."

# ----------------------------
# Resume endpoint - Twilio redirects here after playing audio
# ----------------------------
@app.post("/resume_stream")
async def resume_stream():
    twiml = VoiceResponse()
    twiml.say("Resuming conversation.")
    host_for_twilio = os.getenv("HOST_FOR_TWILIO") or BASE_URL.replace("https://","").replace("http://","")
    start = Start()
    start.stream(url=f"wss://{host_for_twilio}/ws-audio")
    twiml.append(start)
    twiml.pause(length=3600)
    return Response(content=str(twiml), media_type="application/xml")

# ----------------------------
# Serve generated audio files
# ----------------------------
@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    p = AUDIO_DIR / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(p, media_type="audio/wav", filename=filename)

# ----------------------------
# health
# ----------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}



















# import os
# import json
# import logging
# import asyncio
# import websockets
# import aiohttp
# from fastapi import FastAPI, Request, WebSocket
# from fastapi.responses import Response
# from twilio.twiml.voice_response import VoiceResponse, Gather
# from websockets.exceptions import ConnectionClosed

# # ----------------------------
# # Logging Setup
# # ----------------------------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("WagwanAI")

# # ----------------------------
# # Environment Variables
# # ----------------------------
# SPITCH_API_KEY = os.getenv("SPITCH_API_KEY")
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# SPITCH_WS_URL = "wss://api.spitch.io/realtime"
# OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# # ----------------------------
# # FastAPI App
# # ----------------------------
# app = FastAPI()

# # ----------------------------
# # Conversation State
# # ----------------------------
# class ConversationState:
#     def __init__(self):
#         self.language = "en"
#         self.transcripts = []
#         self.last_partial = ""
#         self.ai_task = None
#         self.ws = None

# session_state = ConversationState()

# # ----------------------------
# # STEP 1: Twilio IVR Entry
# # ----------------------------
# @app.post("/voice")
# async def voice_response(request: Request):
#     """Initial IVR prompt for language selection."""
#     twiml = VoiceResponse()
#     gather = Gather(
#         num_digits=1,
#         action="/select_language",
#         method="POST",
#         timeout=5
#     )
#     gather.say("Welcome to Wagwan AI. , Press 1, for Yoruba. 2, for Igbo. 3, for Hausa. or 4, for English.")
#     twiml.append(gather)
#     twiml.redirect("/voice")
#     return Response(content=str(twiml), media_type="application/xml")

# # ----------------------------
# # STEP 2: Handle Language Selection
# # ----------------------------
# @app.post("/select_language")
# async def select_language(request: Request):
#     """Handles selected language and begins streaming."""
#     form = await request.form()
#     choice = form.get("Digits")

#     lang_map = {
#         "1": "yo",  # Yoruba
#         "2": "ig",  # Igbo
#         "3": "ha",  # Hausa
#         "4": "en",  # English
#     }
#     session_state.language = lang_map.get(choice, "en")

#     twiml = VoiceResponse()
#     twiml.say(f"Okay. You selected {session_state.language}. You can start speaking.")
#     start = twiml.start()
#     start.stream(url="wss://spitchhack.onrender.com/ws-audio")
#     twiml.pause(length=60)
#     return Response(content=str(twiml), media_type="application/xml")

# # ----------------------------
# # STEP 3: WebSocket Audio Bridge
# # ----------------------------
# @app.websocket("/ws-audio")
# async def audio_ws(ws: WebSocket):
#     """Handles Twilio <-> Spitch real-time streaming."""
#     await ws.accept()
#     session_state.ws = ws
#     logger.info("Twilio connected to WebSocket.")

#     try:
#         async with websockets.connect(
#             SPITCH_WS_URL,
#             extra_headers={"Authorization": f"Bearer {SPITCH_API_KEY}"}
#         ) as spitch_ws:
#             listener_task = asyncio.create_task(spitch_listener(spitch_ws, ws))

#             while True:
#                 audio_msg = await ws.receive_bytes()
#                 await spitch_ws.send(audio_msg)

#     except ConnectionClosed:
#         logger.warning("Twilio WebSocket closed.")
#     except Exception as e:
#         logger.error(f"Audio WS error: {e}")

# # ----------------------------
# # STEP 4: Spitch Transcription Listener
# # ----------------------------
# async def spitch_listener(spitch_ws, twilio_ws):
#     """Listens to Spitch transcriptions and triggers AI replies."""
#     try:
#         async for msg in spitch_ws:
#             data = json.loads(msg)

#             # Handle partial transcriptions
#             if data.get("type") == "partial":
#                 transcript = data["alternatives"][0]["transcript"]
#                 if transcript != session_state.last_partial:
#                     session_state.last_partial = transcript
#                     logger.info(f"Partial: {transcript}")

#                     # Stream AI reply ASAP
#                     if not session_state.ai_task:
#                         session_state.ai_task = asyncio.create_task(
#                             stream_ai_reply(transcript, twilio_ws)
#                         )

#             # Handle final transcription
#             elif data.get("type") == "final":
#                 final_text = data["alternatives"][0]["transcript"]
#                 logger.info(f"Final: {final_text}")
#                 session_state.transcripts.append(final_text)
#                 session_state.ai_task = None  # Reset AI streaming for next utterance

#     except Exception as e:
#         logger.error(f"Spitch listener failed: {e}")

# # ----------------------------
# # STEP 5: AI Realtime Streaming
# # ----------------------------
# async def stream_ai_reply(prompt: str, twilio_ws):
#     """Streams AI response tokens directly to Twilio."""
#     try:
#         headers = {
#             "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#             "Content-Type": "application/json",
#         }

#         payload = {
#             "model": "mistralai/mistral-7b-instruct:free",
#             "stream": True,
#             "messages": [{"role": "user", "content": prompt}],
#         }

#         async with aiohttp.ClientSession() as session:
#             async with session.post(OPENROUTER_API_URL, headers=headers, json=payload) as resp:
#                 async for chunk in resp.content:
#                     if chunk:
#                         try:
#                             decoded = json.loads(chunk.decode().replace("data: ", ""))
#                             token = decoded.get("choices", [{}])[0].get("delta", {}).get("content")
#                             if token:
#                                 logger.info(f"AI: {token}")

#                                 # Send streaming token to Twilio TTS
#                                 await twilio_ws.send_text(json.dumps({
#                                     "event": "media",
#                                     "media": {"text": token}
#                                 }))
#                         except Exception:
#                             continue

#     except Exception as e:
#         logger.error(f"AI streaming failed: {e}")












# #this particular code works, the one above is juat an optinized version from grok
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
# MODEL = os.getenv("MODEL", "mistralai/mistral-7b-instruct:free")
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