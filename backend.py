import os
import logging
import json
import base64
import time
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, FileResponse
from twilio.twiml.voice_response import VoiceResponse
from twilio.request_validator import RequestValidator
from dotenv import load_dotenv
import asyncio
import google.generativeai as genai
from urllib.parse import urlparse
import httpx

load_dotenv()

# -----------------------
# Configuration / Env
# -----------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
MODEL = os.getenv("MODyEL", "gemini-2.5-flash")
VOICE_ID = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
SPITCH_API_KEY = os.getenv("SPITCH_API_KEY")
SPITCH_API_BASE = os.getenv("SPITCH_API_BASE", "https://api.spi-tch.com/v1")
MEDIA_DIR = os.getenv("MEDIA_DIR", "media")

required_vars = ["GEMINI_API_KEY", "TWILIO_AUTH_TOKEN", "BASE_URL", "SPITCH_API_KEY"]
for var in required_vars:
    if not globals().get(var):
        raise RuntimeError(f"Missing environment variable: {var}")

os.makedirs(MEDIA_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Language and voice maps
# - LANGUAGE_MAP: DTMF choice -> (name, twiml-code, spitch/gemini-code)
# - SPITCH_VOICE_MAP: language code -> spitch voice identifier (replace with real voice ids)
# ------------------------------------------------------------------
LANGUAGE_MAP = {
    "1": ("Yoruba", "yo-NG", "yo"),
    "2": ("Igbo", "ig-NG", "ig"),
    "3": ("Hausa", "ha-NG", "ha"),
    "4": ("English", "en-US", "en"),
}

# These are placeholder voice ids for Spitch. Replace them with actual voice ids from Spitch dashboard or API.
SPITCH_VOICE_MAP = {
    "yo": os.getenv("SPITCH_VOICE_YO", "yo_default_voice"),
    "ig": os.getenv("SPITCH_VOICE_IG", "ig_default_voice"),
    "ha": os.getenv("SPITCH_VOICE_HA", "ha_default_voice"),
    "en": os.getenv("SPITCH_VOICE_EN", "jude"),
}

# Conversation state
LANGUAGE_SELECTION: Dict[str, tuple] = {}  # CallSid -> (lang_name, lang_code_twiml, lang_code_spitch)
CONVERSATION_HISTORY: Dict[str, list] = {}  # CallSid -> list of {role, content}

# -----------------------
# Logging + app
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spitch-conversation-relay")
app = FastAPI()

tenant_validator = RequestValidator(TWILIO_AUTH_TOKEN)

# Configure Gemini client
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to initialize Gemini client: {e}")

# -----------------------
# Helpers: Spitch (async httpx)
# -----------------------

async def spitch_translate(text: str, source: str, target: str) -> str:
    """Call Spitch translate endpoint. Returns translated text or raises."""
    url = f"{SPITCH_API_BASE}/translate"
    payload = {"source": source, "target": target, "text": text}
    headers = {"Authorization": f"Bearer {SPITCH_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        r = await client.post(url, json=payload, headers=headers, timeout=30)
    try:
        data = r.json()
    except Exception:
        logger.error("Unexpected non-json response from Spitch translate: %s", r.text)
        raise

    # Try common fields
    for key in ("translated_text", "translation", "text", "result", "translated"):
        if key in data:
            val = data[key]
            if isinstance(val, dict) and "text" in val:
                return val["text"]
            if isinstance(val, str):
                return val
    # Fallback: try to find nested
    if isinstance(data, dict):
        # search for any string value
        for v in data.values():
            if isinstance(v, str) and len(v) > 0:
                return v
    raise RuntimeError("Could not parse Spitch translate response")


async def spitch_transcribe_from_url(audio_url: str, language: Optional[str] = None) -> Dict[str, Any]:
    """Call Spitch transcription using a remote URL. Returns dict with at least 'text' and optionally 'language'."""
    url = f"{SPITCH_API_BASE}/transcriptions"
    payload = {
        "model": "mansa_v1",
        "url": audio_url,
        "language": language or "auto",
        "timestamp": "sentence",
    }
    headers = {"Authorization": f"Bearer {SPITCH_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        r = await client.post(url, json=payload, headers=headers, timeout=60)
    try:
        data = r.json()
    except Exception:
        logger.error("Unexpected non-json response from Spitch transcriptions: %s", r.text)
        raise

    # attempt to parse common fields
    text = None
    language = None
    for key in ("text", "transcript", "transcription", "result"):
        if key in data:
            if isinstance(data[key], dict) and "text" in data[key]:
                text = data[key]["text"]
            elif isinstance(data[key], str):
                text = data[key]
    # language fields
    for key in ("language", "detected_language", "lang"):
        if key in data:
            language = data[key]
            break

    if not text:
        # try scanning
        for v in data.values():
            if isinstance(v, str) and len(v) > 0:
                text = v
                break

    return {"text": text or "", "language": language or ""}


async def spitch_tts_to_url(text: str, language: str, voice: str, call_sid: str) -> str:
    """Call Spitch TTS. Try multiple response shapes: url, base64 'audio', or direct audio bytes.
    If audio bytes or base64 are returned, save locally and serve via /media/{filename} endpoint so Twilio can fetch it.
    Returns a public URL to the audio file (either Spitch-hosted or our /media endpoint).
    """
    candidate_endpoints = ["/tts", "/synthesis", "/tts/synthesize"]
    headers = {"Authorization": f"Bearer {SPITCH_API_KEY}", "Content-Type": "application/json"}
    payload = {"text": text, "language": language, "voice": voice, "format": "mp3"}

    async with httpx.AsyncClient() as client:
        for ep in candidate_endpoints:
            try:
                r = await client.post(f"{SPITCH_API_BASE}{ep}", json=payload, headers=headers, timeout=60)
            except httpx.HTTPStatusError:
                continue
            except Exception:
                continue
            # success-ish
            # if content-type is audio/* -> save bytes
            ctype = r.headers.get("content-type", "")
            if ctype.startswith("audio/"):
                filename = f"{call_sid}_{int(time.time())}_{uuid.uuid4().hex[:6]}.mp3"
                path = os.path.join(MEDIA_DIR, filename)
                with open(path, "wb") as fh:
                    fh.write(r.content)
                return f"{BASE_URL}/media/{filename}"
            # try json body with url or base64
            try:
                data = r.json()
            except Exception:
                logger.warning("TTS endpoint %s returned non-json non-audio response", ep)
                continue

            # common response shapes
            if isinstance(data, dict) and "url" in data and isinstance(data["url"], str):
                return data["url"]
            if "audio" in data and isinstance(data["audio"], str):
                # base64 audio
                try:
                    audio_b64 = data["audio"]
                    audio_bytes = base64.b64decode(audio_b64)
                    filename = f"{call_sid}_{int(time.time())}_{uuid.uuid4().hex[:6]}.mp3"
                    path = os.path.join(MEDIA_DIR, filename)
                    with open(path, "wb") as fh:
                        fh.write(audio_bytes)
                    return f"{BASE_URL}/media/{filename}"
                except Exception:
                    logger.exception("Failed to decode base64 audio from spitch tts")
            # sometimes nested
            if "result" in data and isinstance(data["result"], dict) and "url" in data["result"]:
                return data["result"]["url"]

    raise RuntimeError("Spitch TTS failed or returned unknown format")


# -----------------------
# Gemini helper
# -----------------------
async def gemini_chat_reply(messages: list, model_name: str = MODEL) -> str:
    """Send messages to Gemini (assumes messages are in English already). Returns response text.
    Keeps the system prompt minimal because we translate around it.
    """
    try:
        # Convert to Gemini format
        gemini_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_messages.append({"role": role, "parts": [msg["content"]]})

        model = genai.GenerativeModel(model_name, system_instruction="You are a helpful assistant. Reply in English.")
        chat = model.start_chat(history=gemini_messages[:-1])
        response = await chat.send_message_async(gemini_messages[-1]["parts"][0])
        return response.text
    except Exception as e:
        logger.exception("Gemini API error: %s", e)
        return "Sorry, I couldn't process your request. Please try again."


# -----------------------
# Twilio voice endpoints
# -----------------------
@app.get("/")
async def root():
    return {"message": "Welcome to the Spitch-Gemini Voice Relay. Use /health to check status."}


@app.post("/voice")
async def voice_entry(request: Request):
    try:
        form_data = await request.form()
        signature = request.headers.get("X-Twilio-Signature", "")
        url = str(request.url)
        if not tenant_validator.validate(url, dict(form_data), signature):
            logger.warning("Invalid Twilio signature.")
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

        twiml = VoiceResponse()
        gather = twiml.gather(num_digits=1, action="/process_language", method="POST", timeout=8)
        gather.say("Welcome to Proxy. For Yoruba press one. For Igbo press two. For Hausa press three. For English press four.")
        twiml.redirect("/process_language_fallback")
        return Response(content=str(twiml), media_type="application/xml")
    except Exception as e:
        logger.exception("Error in /voice: %s", e)
        twiml = VoiceResponse()
        twiml.say("An unexpected error occurred. Please try your call again.")
        return Response(content=str(twiml), media_type="application/xml", status_code=500)


@app.post("/process_language_fallback")
async def process_language_fallback(request: Request):
    twiml = VoiceResponse()
    twiml.say("Sorry, we did not receive any input. Redirecting you back to language selection.")
    twiml.redirect("/voice")
    return Response(content=str(twiml), media_type="application/xml")


@app.post("/process_language")
async def process_language(request: Request, Digits: str = Form(None), CallSid: str = Form(None)):
    try:
        form_data = await request.form()
        signature = request.headers.get("X-Twilio-Signature", "")
        url = str(request.url)
        if not tenant_validator.validate(url, dict(form_data), signature):
            logger.warning("Invalid Twilio signature.")
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

        twiml = VoiceResponse()
        if not (Digits and CallSid and Digits in LANGUAGE_MAP):
            twiml.say("Invalid selection or call ID. Please try again.")
            twiml.redirect("/voice")
            return Response(content=str(twiml), media_type="application/xml")

        lang_name, lang_code_twiml, lang_code_spitch = LANGUAGE_MAP[Digits]
        LANGUAGE_SELECTION[CallSid] = (lang_name, lang_code_twiml, lang_code_spitch)
        logger.info("Language set for CallSid %s -> %s", CallSid, lang_name)

        twiml.say(f"You selected {lang_name}. Connecting you now.")

        connect = twiml.connect()
        # We deliberately avoid asking Twilio to transcribe (so we can use Spitch). Twilio will still forward audio events to our websocket.
        conversation_relay = connect.conversation_relay(
            url=f"wss://{urlparse(BASE_URL).netloc}/relay",
            interruptible="any",
            report_input_during_agent_speech="any",
            debug="speaker-events"
        )
        # Keep Twilio TTS config minimal; we will supply audio ourselves via Spitch for the assistant replies.
        conversation_relay.language(code=lang_code_twiml, tts_provider="elevenlabs", voice=VOICE_ID)

        return Response(content=str(twiml), media_type="application/xml")
    except Exception as e:
        logger.exception("Error in /process_language: %s", e)
        twiml = VoiceResponse()
        twiml.say("An unexpected error occurred while processing your selection. Please try your call again.")
        return Response(content=str(twiml), media_type="application/xml", status_code=500)


@app.get("/health")
async def health():
    status = {"status": "ok", "services": {}}
    # Test Gemini connectivity (simple call)
    try:
        model = genai.GenerativeModel(MODEL)
        chat = model.start_chat()
        response = await chat.send_message_async("Health check ping")
        status["services"]["gemini"] = "ok"
    except Exception as e:
        logger.exception("Health check error for Gemini: %s", e)
        status["services"]["gemini"] = f"down: {e}"
    # Test Spitch by making a lightweight call (optional)
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{SPITCH_API_BASE}/health", headers={"Authorization": f"Bearer {SPITCH_API_KEY}"}, timeout=5)
            status["services"]["spitch"] = "ok" if r.status_code == 200 else f"unexpected status {r.status_code}"
    except Exception as e:
        logger.warning("Spitch health check failed: %s", e)
        status["services"]["spitch"] = f"down: {e}"
    return status


# Serve generated audio files
@app.get("/media/{filename}")
async def serve_media(filename: str):
    path = os.path.join(MEDIA_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="audio/mpeg")


# -----------------------
# WebSocket Relay Handler
# -----------------------
@app.websocket("/relay")
async def relay_websocket(websocket: WebSocket):
    await websocket.accept()
    call_sid = None
    message_queue = asyncio.Queue()
    current_response_task = None

    async def receiver():
        while True:
            try:
                data = await websocket.receive_text()
                await message_queue.put(json.loads(data))
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected.")
                await message_queue.put(None)
                break
            except Exception as e:
                logger.exception("Receiver error: %s", e)
                await message_queue.put(None)
                break

    receive_task = asyncio.create_task(receiver())

    try:
        while True:
            message = await message_queue.get()
            if message is None:
                break

            logger.debug("WebSocket event: %s", message)
            event_type = message.get("type")

            if event_type == "setup":
                call_sid = message.get("callSid")
                if not call_sid:
                    logger.error("Missing callSid in setup message.")
                    continue
                CONVERSATION_HISTORY[call_sid] = [{"role": "system", "content": "You are a helpful assistant named Proxy."}]
                logger.info("Setup complete for CallSid %s", call_sid)
                continue

            elif event_type == "prompt":
                # This can contain either a pre-transcribed 'voicePrompt' or a media/audio url. We handle both.
                if not call_sid:
                    logger.warning("Prompt received before setup")
                    continue

                user_text = None
                detected_lang = None

                # If Twilio already provided a transcript
                if message.get("voicePrompt"):
                    user_text = message.get("voicePrompt").strip()
                    logger.info("Received voicePrompt transcript: %s", user_text)
                    # assume language == selection
                    _, _, lang_code_spitch = LANGUAGE_SELECTION.get(call_sid, ("English", "en-US", "en"))
                    detected_lang = lang_code_spitch

                else:
                    # Try to discover audio url(s) in the Twilio event
                    audio_url = (
                        message.get("mediaUrl")
                        or message.get("audioUrl")
                        or (message.get("media") or {}).get("url")
                        or (message.get("audio") or {}).get("url")
                        or message.get("recordingUrl")
                    )
                    if not audio_url:
                        logger.warning("No voicePrompt or audio URL found in prompt event")
                        continue

                    try:
                        trans = await spitch_transcribe_from_url(audio_url)
                        user_text = trans.get("text", "").strip()
                        detected_lang = trans.get("language") or LANGUAGE_SELECTION.get(call_sid, ("English", "en-US", "en"))[2]
                        logger.info("Spitch transcription result: %s (lang=%s)", user_text, detected_lang)
                    except Exception as e:
                        logger.exception("Error transcribing audio: %s", e)
                        await websocket.send_text(json.dumps({"type": "text", "token": "Sorry, I couldn't transcribe your speech.", "last": True}))
                        continue

                if not user_text:
                    logger.warning("Empty user text after transcription/voicePrompt")
                    continue

                # Cancel any ongoing response if the user spoke
                if current_response_task and not current_response_task.done():
                    logger.info("Interrupting previous response task.")
                    current_response_task.cancel()

                # Determine language mapping
                _, _, user_lang = LANGUAGE_SELECTION.get(call_sid, ("English", "en-US", "en"))

                async def stream_response():
                    try:
                        # If user language is not English, translate to English for Gemini
                        if user_lang != "en":
                            try:
                                to_gemini = await spitch_translate(user_text, source=user_lang, target="en")
                            except Exception as e:
                                logger.exception("Translation to English failed: %s", e)
                                # fallback to original text
                                to_gemini = user_text
                        else:
                            to_gemini = user_text

                        # Append to history (store user message in English so Gemini gets consistent context)
                        hist = CONVERSATION_HISTORY.get(call_sid, [{"role": "system", "content": "You are a helpful assistant named Proxy."}])
                        hist.append({"role": "user", "content": to_gemini})

                        # Call Gemini
                        response_en = await gemini_chat_reply(hist)

                        # Store assistant reply in English
                        hist.append({"role": "assistant", "content": response_en})
                        CONVERSATION_HISTORY[call_sid] = hist[-20:]

                        # Translate back to user language if needed
                        if user_lang != "en":
                            try:
                                response_local = await spitch_translate(response_en, source="en", target=user_lang)
                            except Exception as e:
                                logger.exception("Back-translation failed: %s", e)
                                response_local = response_en
                        else:
                            response_local = response_en

                        # Generate TTS audio via Spitch
                        voice_to_use = SPITCH_VOICE_MAP.get(user_lang, SPITCH_VOICE_MAP.get("en"))
                        try:
                            audio_url = await spitch_tts_to_url(response_local, language=user_lang, voice=voice_to_use, call_sid=call_sid)
                        except Exception as e:
                            logger.exception("Spitch TTS failed: %s", e)
                            audio_url = None

                        # Send both a text token (fallback) and audio url (preferred)
                        payload = {
                            "type": "text",
                            "token": response_local,
                            "last": True,
                            "assistant_text_en": response_en,
                        }
                        # If we have audio, include it in a separate field. Twilio Conversation Relay may accept a media payload.
                        if audio_url:
                            payload.update({"type": "audio", "url": audio_url, "contentType": "audio/mpeg"})

                        await websocket.send_text(json.dumps(payload))
                        logger.info("Response sent to Twilio for CallSid %s", call_sid)

                    except asyncio.CancelledError:
                        logger.warning("Response task was cancelled.")
                        raise
                    except Exception as e:
                        logger.exception("Error during response streaming: %s", e)
                        await websocket.send_text(json.dumps({"type": "text", "token": "Sorry, a temporary error occurred. Please try again.", "last": True}))

                current_response_task = asyncio.create_task(stream_response())
                continue

            elif event_type == "speaker":
                if message.get("event") == "clientSpeaking" and current_response_task and not current_response_task.done():
                    logger.info("Client speaking detected. Cancelling ongoing response.")
                    current_response_task.cancel()
                continue

            elif event_type == "dtmf":
                logger.info("DTMF received: %s", message.get("digit"))
                continue

            elif event_type == "error":
                logger.error("Error received from Twilio: %s", message.get("error"))
                continue

            elif event_type == "call_ended":
                LANGUAGE_SELECTION.pop(call_sid, None)
                CONVERSATION_HISTORY.pop(call_sid, None)
                logger.info("Cleaned up state for ended call %s", call_sid)
                break

            logger.warning("Unknown event type received: %s", event_type)

    except Exception as e:
        logger.exception("Unexpected error in WebSocket handler: %s", e)
    finally:
        if current_response_task:
            current_response_task.cancel()
        if not receive_task.done():
            receive_task.cancel()
        if call_sid:
            LANGUAGE_SELECTION.pop(call_sid, None)
            CONVERSATION_HISTORY.pop(call_sid, None)
        try:
            if websocket.client_state.name == "CONNECTED":
                await websocket.close()
        except Exception:
            pass











# # import os
# # import logging
# # from typing import Dict, Any
# # from fastapi import FastAPI, Request, Form, HTTPException, WebSocket, WebSocketDisconnect
# # from fastapi.responses import Response
# # from twilio.twiml.voice_response import VoiceResponse, Start, Stream
# # from twilio.request_validator import RequestValidator
# # from dotenv import load_dotenv
# # from openai import AsyncOpenAI
# # from urllib.parse import urlparse
# # import json
# # import asyncio
# # import google.generativeai as genai


# # load_dotenv()

# # app = FastAPI()

# # # Validate environment variables
# # required_vars = [
# #     "GEMINI_API_KEY",
# #     "TWILIO_ACCOUNT_SID",
# #     "TWILIO_AUTH_TOKEN",
# #     "BASE_URL",
# #     "CONVERSATION_SERVICE_SID"
# # ]
# # for var in required_vars:
# #     if not os.getenv(var):
# #         raise RuntimeError(f"Missing environment variable: {var}")

# # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# # TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
# # BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
# # MODEL = os.getenv("MODwEL", "gemini-2.5-flash")
# # VOICE_ID = os.getenv("VOICE_ID")
# # SYSTEM_PROMPT = "You are a helpful assistant named Proxy. This conversation is being translated to voice, so answer carefully. When you respond, please spell out all numbers, for example twenty not 20. Do not include emojis in your responses. Do not include bullet points, asterisks, or special symbols."

# # # ---- Gemini Client ----
# # try:
# #     genai.configure(api_key=GEMINI_API_KEY)
# # except Exception as e:
# #     raise RuntimeError(f"Failed to initialize Gemini client: {e}")

# # # ---- App setup ----
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger("conversation-relay")
# # app = FastAPI()
# # twilio_validator = RequestValidator(TWILIO_AUTH_TOKEN)

# # # ---- Language map ----
# # LANGUAGE_MAP = {
# #     "1": ("Yoruba", "yo-NG", "yo"),
# #     "2": ("Igbo", "ig-NG", "ig"),
# #     "3": ("Hausa", "ha-NG", "ha"),
# #     "4": ("English", "en-US", "en")
# # }

# # LANGUAGE_SELECTION: Dict[str, tuple] = {}  # CallSid -> (lang_name, lang_code_twiml, lang_code_gemini)
# # CONVERSATION_HISTORY: Dict[str, list] = {}  # CallSid -> list of {"role": str, "content": str}

# # # ---- Gemini Helpers ----
# # async def gemini_chat_reply(messages: list, language: str = "en") -> str:
# #     """
# #     Get a response from Gemini, ensuring it responds in the specified language
# #     """
# #     try:
# #         # Add language instruction to the last message
# #         if messages and messages[-1]["role"] == "user":
# #             lang_instruction = f" Please respond in {language} language."
# #             messages[-1]["content"] += lang_instruction
        
# #         # Convert to Gemini format
# #         gemini_messages = []
# #         for msg in messages:
# #             # Skip system message for now, will add as instruction
# #             if msg["role"] == "system":
# #                 continue
# #             gemini_messages.append({"role": "user" if msg["role"] == "user" else "model", "parts": [msg["content"]]})
        
# #         # Create model with system prompt as instruction
# #         model = genai.GenerativeModel(
# #             MODEL,
# #             system_instruction=SYSTEM_PROMPT + f" Always respond in {language} language."
# #         )
        
# #         # Start chat with history
# #         chat = model.start_chat(history=gemini_messages[:-1])  # All but the last message
        
# #         # Get response for the last message
# #         response = await chat.send_message_async(gemini_messages[-1]["parts"][0])
        
# #         return response.text
# #     except Exception as e:
# #         logger.error(f"Gemini API error: {e}")
# #         return "Sorry, I couldn't process your request. Please try again."

# # # ---- Root endpoint ----
# # @app.get("/")
# # async def root():
# #     return {"message": "Welcome to the Gemini Voice Relay API. Use /health to check status."}

# # # ---- TwiML entry ----
# # @app.post("/voice")
# # async def voice_entry(request: Request):
# #     # Validate Twilio webhook
# #     form_data = await request.form()
# #     signature = request.headers.get("X-Twilio-Signature", "")
# #     url = str(request.url)
# #     if not twilio_validator.validate(url, dict(form_data), signature):
# #         raise HTTPException(status_code=403, detail="Invalid Twilio signature")

# #     twiml = VoiceResponse()
# #     # Gather language selection
# #     gather = twiml.gather(
# #         num_digits=1,
# #         action="/process_language",
# #         method="POST",
# #         timeout=8
# #     )
# #     gather.say("Welcome to Proxy. For Yoruba press 1. For Igbo press 2. For Hausa press 3. For English press 4.")
# #     # If gather doesn't get input:
# #     twiml.redirect("/process_language_fallback")

# #     return Response(content=str(twiml), media_type="application/xml")

# # @app.post("/process_language_fallback")
# # async def process_language_fallback(request: Request):
# #     twiml = VoiceResponse()
# #     twiml.say("Sorry, we did not receive input. Redirecting you back to language selection.")
# #     twiml.redirect("/voice")
# #     return Response(content=str(twiml), media_type="application/xml")

# # @app.post("/process_language")
# # async def process_language(request: Request, Digits: str = Form(None), CallSid: str = Form(None)):
# #     # Validate Twilio webhook
# #     form_data = await request.form()
# #     signature = request.headers.get("X-Twilio-Signature", "")
# #     url = str(request.url)
# #     if not twilio_validator.validate(url, dict(form_data), signature):
# #         raise HTTPException(status_code=403, detail="Invalid Twilio signature")

# #     twiml = VoiceResponse()
# #     if not (Digits and CallSid and Digits in LANGUAGE_MAP):
# #         twiml.say("Invalid selection or call ID. Please try again.")
# #         twiml.redirect("/voice")
# #         return Response(content=str(twiml), media_type="application/xml")

# #     lang_name, lang_code_twiml, lang_code_gemini = LANGUAGE_MAP[Digits]
# #     LANGUAGE_SELECTION[CallSid] = (lang_name, lang_code_twiml, lang_code_gemini)
# #     logger.info("Language set for CallSid %s -> %s", CallSid, lang_name)

# #     twiml.say(f"You selected {lang_name}. Connecting you now.")

# #     connect = twiml.connect()
# #     conversation_relay = connect.conversation_relay(
# #         url=f"wss://{urlparse(BASE_URL).netloc}/relay",
# #         interruptible="any",
# #         report_input_during_agent_speech="any",
# #         debug="speaker-events"
# #     )
# #     language = conversation_relay.language(
# #         code=lang_code_twiml,
# #         tts_provider="elevenlabs",
# #         voice=VOICE_ID,
# #         transcription_provider="google"
# #     )

# #     return Response(content=str(twiml), media_type="application/xml")

# # # ---- Health check ----
# # @app.get("/health")
# # async def health():
# #     status = {"status": "ok", "services": {}}
# #     # Test Gemini
# #     try:
# #         model = genai.GenerativeModel(MODEL)
# #         chat = model.start_chat()
# #         response = chat.send_message("Test message")
# #         status["services"]["gemini"] = "ok"
# #     except Exception as e:
# #         status["services"]["gemini"] = f"down: {e}"
# #     return status

# # @app.websocket("/relay")
# # async def relay_websocket(websocket: WebSocket):
# #     await websocket.accept()
# #     call_sid = None
# #     message_queue = asyncio.Queue()
# #     interrupted = False
# #     current_response_task = None

# #     async def receiver():
# #         while True:
# #             try:
# #                 data = await websocket.receive_text()
# #                 await message_queue.put(json.loads(data))
# #             except WebSocketDisconnect:
# #                 await message_queue.put(None)
# #                 break
# #             except Exception as e:
# #                 logger.error(f"Receiver error: {e}")
# #                 await message_queue.put(None)
# #                 break

# #     receive_task = asyncio.create_task(receiver())

# #     try:
# #         while True:
# #             message = await message_queue.get()
# #             if message is None:
# #                 break

# #             logger.debug("WebSocket event: %s", message)
# #             event_type = message.get("type")

# #             if event_type == "setup":
# #                 call_sid = message.get("callSid")
# #                 if not call_sid:
# #                     logger.error("Missing callSid in setup")
# #                     continue
# #                 CONVERSATION_HISTORY[call_sid] = [{"role": "system", "content": SYSTEM_PROMPT}]
# #                 logger.info("Setup for CallSid %s", call_sid)
# #                 continue

# #             elif event_type == "prompt":
# #                 user_text = message.get("voicePrompt")
# #                 if not user_text or not user_text.strip():
# #                     logger.error("Missing or empty voicePrompt")
# #                     continue

# #                 # If there's an ongoing response, interrupt it
# #                 if current_response_task and not current_response_task.done():
# #                     interrupted = True
# #                     await asyncio.sleep(0)

# #                 lang_name, _, lang_gemini = LANGUAGE_SELECTION.get(call_sid, ("English", "en-US", "en"))

# #                 try:
# #                     # Add user message to history
# #                     history = CONVERSATION_HISTORY.get(call_sid, [{"role": "system", "content": SYSTEM_PROMPT}])
# #                     history.append({"role": "user", "content": user_text})

# #                     # Reset interrupted for new response
# #                     interrupted = False

# #                     async def stream_response():
# #                         nonlocal interrupted, history
# #                         try:
# #                             # Get response from Gemini in the selected language
# #                             response_text = await gemini_chat_reply(history, lang_name)
                            
# #                             if not interrupted:
# #                                 # Stream the response
# #                                 await websocket.send_text(
# #                                     json.dumps({
# #                                         "type": "text",
# #                                         "token": response_text,
# #                                         "last": True
# #                                     })
# #                                 )
# #                                 # Add assistant response to history
# #                                 history.append({"role": "assistant", "content": response_text})
# #                                 CONVERSATION_HISTORY[call_sid] = history[-20:]  # Keep last 20 messages
# #                         except Exception as e:
# #                             logger.error(f"Error in stream_response: {e}")
# #                             if not interrupted:
# #                                 await websocket.send_text(
# #                                     json.dumps({
# #                                         "type": "text",
# #                                         "token": "Sorry, an error occurred. Please try again.",
# #                                         "last": True
# #                                     })
# #                                 )

# #                     current_response_task = asyncio.create_task(stream_response())

# #                 except Exception as e:
# #                     logger.error(f"Error processing prompt: {e}")
# #                     await websocket.send_text(
# #                         json.dumps({
# #                             "type": "text",
# #                             "token": "Sorry, an error occurred. Please try again.",
# #                             "last": True
# #                         })
# #                     )
# #                 continue

# #             elif event_type == "speaker":
# #                 if message.get("event") == "clientSpeaking":
# #                     logger.info("Client speaking detected - potential interruption")
# #                     interrupted = True
# #                 continue

# #             elif event_type == "dtmf":
# #                 logger.info("DTMF received: %s", message)
# #                 continue

# #             elif event_type == "error":
# #                 logger.error("Error received: %s", message)
# #                 continue

# #             elif event_type == "call_ended":
# #                 LANGUAGE_SELECTION.pop(call_sid, None)
# #                 CONVERSATION_HISTORY.pop(call_sid, None)
# #                 logger.info("Cleaned up for CallSid %s", call_sid)
# #                 continue

# #             logger.warning("Unknown event type: %s", event_type)

# #     except Exception as e:
# #         logger.error("WebSocket error: %s", e)
# #     finally:
# #         if current_response_task:
# #             current_response_task.cancel()
# #         receive_task.cancel()
# #         if call_sid:
# #             LANGUAGE_SELECTION.pop(call_sid, None)
# #             CONVERSATION_HISTORY.pop(call_sid, None)
        
# #         # Only close if the connection is still open
# #         if websocket.client_state.name == "CONNECTED":
# #             await websocket.close()



















# # # john code that works
# # import os
# # import logging
# # from typing import Dict, Any
# # from fastapi import FastAPI, Request, Form, HTTPException, WebSocket, WebSocketDisconnect
# # from fastapi.responses import Response
# # from fastapi import WebSocket
# # from twilio.twiml.voice_response import VoiceResponse, Start, Stream
# # from twilio.request_validator import RequestValidator
# # from dotenv import load_dotenv
# # from spitch import Spitch
# # from openai import AsyncOpenAI
# # from urllib.parse import urlparse
# # import json
# # import asyncio

# # # ---- Config ----
# # load_dotenv()


# # app = FastAPI()

# # # Validate environment variables
# # required_vars = [
# #     "SPITCH_API_KEY",
# #     "OPENROUTER_API_KEY",
# #     "TWILIO_ACCOUNT_SID",
# #     "TWILIO_AUTH_TOKEN",
# #     "BASE_URL",
# #     "CONVERSATION_SERVICE_SID"
# # ]
# # for var in required_vars:
# #     if not os.getenv(var):
# #         raise RuntimeError(f"Missing environment variable: {var}")

# # SPITCH_API_KEY = os.getenv("SPITCH_API_KEY")
# # OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# # TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
# # BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
# # MODEL = os.getenv("MODEL", "gpt-4o-mini")
# # VOICE_ID = os.getenv("VOICE_ID")
# # SYSTEM_PROMPT = "You are a helpful assistant named Proxy. This conversation is being translated to voice, so answer carefully. When you respond, please spell out all numbers, for example twenty not 20. Do not include emojis in your responses. Do not include bullet points, asterisks, or special symbols."

# # # ---- Clients ----
# # try:
# #     spitch_client = Spitch(api_key=SPITCH_API_KEY)
# #     openrouter_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
# # except Exception as e:
# #     raise RuntimeError(f"Failed to initialize clients: {e}")

# # # ---- App setup ----
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger("conversation-relay")
# # app = FastAPI()
# # twilio_validator = RequestValidator(TWILIO_AUTH_TOKEN)

# # # ---- Language map ----
# # # LANGUAGE_MAP = {
# # #     "1": ("Yoruba", "yo"),
# # #     "2": ("Igbo", "ig"),
# # #     "3": ("Hausa", "ha"),
# # #     "4": ("English", "en")
# # # }
# # LANGUAGE_MAP = {
# #     "1": ("Yoruba", "yo-NG", "yo"), #aiit so this is now (BCP-47 code, spitch code)
# #     "2": ("Igbo", "ig-NG", "ig"),
# #     "3": ("Hausa", "ha-NG", "ha"),
# #     "4": ("English", "en-US", "en")
# # }

# # # LANGUAGE_SELECTION: Dict[str, str] = {}  # CallSid -> lang code
# # LANGUAGE_SELECTION: Dict[str, tuple] = {}  # CallSid -> (lang_name, lang_code_twiml, lang_code_spitch) instead of just lang_code
# # CONVERSATION_HISTORY: Dict[str, list] = {}  # CallSid -> list of {"role": str, "content": str}
# # #the conversation history is that Proxy can retain conversation cintext between calls

# # # ---- Helpers ----
# # def spitch_translate(text: str, source: str, target: str) -> str:
# #     """
# #     Translate `text` from `source` language to `target` using Spitch API.
# #     """
# #     try:
# #         resp = spitch_client.text.translate(text=text, source=source, target=target)
# #         t = getattr(resp, "text", None)
# #         if not t:
# #             raise RuntimeError("Empty translation from Spitch")
# #         return t
# #     except Exception as e:
# #         logger.error(f"Spitch translation failed: {e}")
# #         raise RuntimeError(f"Translation error: {e}")

# # def openrouter_chat_reply(messages: list) -> str:
# #     try:
# #         resp = openrouter_client.chat.completions.create(model=MODEL, messages=messages)
# #         return resp.choices[0].message.content
# #     except Exception as e:
# #         logger.error(f"OpenRouter API error: {e}")
# #         return "Sorry, I couldn't process your request. Please try again."

# # # ---- Root endpoint ----
# # @app.get("/")
# # async def root():
# #     return {"message": "Welcome to the SpitchHack Voice Relay API. Use /health to check status."}

# # # ---- TwiML entry ----
# # @app.post("/voice")
# # async def voice_entry(request: Request):
# #     # Validate Twilio webhook
# #     form_data = await request.form()
# #     signature = request.headers.get("X-Twilio-Signature", "")
# #     url = str(request.url)
# #     if not twilio_validator.validate(url, dict(form_data), signature):
# #         raise HTTPException(status_code=403, detail="Invalid Twilio signature")

# #     twiml = VoiceResponse()
# #     # Gather language selection
# #     gather = twiml.gather(
# #         num_digits=1,
# #         action="/process_language",
# #         method="POST",
# #         timeout=8
# #     )
# #     gather.say("Welcome to Proxy. For Yoruba press 1. For Igbo press 2. For Hausa press 3. For English press 4.")
# #     # If gather doesn't get input:
# #     twiml.redirect("/process_language_fallback")

# #     return Response(content=str(twiml), media_type="application/xml")

# # @app.post("/process_language_fallback")
# # async def process_language_fallback(request: Request):
# #     twiml = VoiceResponse()
# #     twiml.say("Sorry, we did not receive input. Redirecting you back to language selection.")
# #     twiml.redirect("/voice")
# #     return Response(content=str(twiml), media_type="application/xml")

# # @app.post("/process_language")
# # async def process_language(request: Request, Digits: str = Form(None), CallSid: str = Form(None)):
# #     # Validate Twilio webhook
# #     form_data = await request.form()
# #     signature = request.headers.get("X-Twilio-Signature", "")
# #     url = str(request.url)
# #     if not twilio_validator.validate(url, dict(form_data), signature):
# #         raise HTTPException(status_code=403, detail="Invalid Twilio signature")

# #     twiml = VoiceResponse()
# #     if not (Digits and CallSid and Digits in LANGUAGE_MAP):
# #         twiml.say("Invalid selection or call ID. Please try again.")
# #         twiml.redirect("/voice")
# #         return Response(content=str(twiml), media_type="application/xml")

# #     lang_name, lang_code_twiml, lang_code_spitch = LANGUAGE_MAP[Digits]
# #     LANGUAGE_SELECTION[CallSid] = (lang_name, lang_code_twiml, lang_code_spitch)
# #     logger.info("Language set for CallSid %s -> %s", CallSid, lang_name, lang_code_twiml, lang_code_spitch)

# #     twiml.say(f"You selected {lang_name}. Connecting you now.")

# #     # # Fixing BASE_URL parsing
# #     # parsed = urlparse(BASE_URL)
# #     # host = parsed.netloc or parsed.path  # in case BASE_URL had no scheme
# #     # # Use websocket URL derived from BASE_URL
# #     # ws_url = f"wss://{host}/relay"

# #     # start = Start()
# #     # stream = Stream(url=ws_url)
# #     # start.append(stream)
# #     # twiml.append(start)

# #     connect = twiml.connect()
# #     conversation_relay = connect.conversation_relay(
# #         url=f"wss://{urlparse(BASE_URL).netloc}/relay",
# #         interruptible="any",
# #         report_input_during_agent_speech="any",
# #         debug="speaker-events"
# #         )
# #     language = conversation_relay.language(
# #         code=lang_code_twiml,
# #         tts_provider="elevenlabs",
# #         voice=VOICE_ID,
# #         transcription_provider="google"
# #         #both google and elevenlabs support yoruba/igbo/hausa
# #     )

# #     return Response(content=str(twiml), media_type="application/xml")


# # # ---- Health check ----
# # @app.get("/health")
# # async def health():
# #     status = {"status": "ok", "services": {}}
# #     # Test Spitch translate
# #     try:
# #         _ = spitch_translate("test", "en", "en")
# #         status["services"]["spitch"] = "ok"
# #     except Exception as e:
# #         status["services"]["spitch"] = f"down: {e}"
# #     # Test OpenRouter
# #     try:
# #         _ = openrouter_client.chat.completions.create(model=MODEL, messages=[{"role": "system", "content": "test"}])
# #         status["services"]["openrouter"] = "ok"
# #     except Exception as e:
# #         status["services"]["openrouter"] = f"down: {e}"
# #     return status

# # @app.websocket("/relay")
# # async def relay_websocket(websocket: WebSocket):
# #     await websocket.accept()
# #     call_sid = None
# #     message_queue = asyncio.Queue()
# #     interrupted = False
# #     current_response_task = None

# #     async def receiver():
# #         while True:
# #             try:
# #                 data = await websocket.receive_text()
# #                 await message_queue.put(json.loads(data))
# #             except WebSocketDisconnect:
# #                 await message_queue.put(None)
# #                 break
# #             except Exception as e:
# #                 logger.error(f"Receiver error: {e}")
# #                 await message_queue.put(None)
# #                 break

# #     receive_task = asyncio.create_task(receiver())

# #     try:
# #         while True:
# #             message = await message_queue.get()
# #             if message is None:
# #                 break

# #             logger.debug("WebSocket event: %s", message)
# #             event_type = message.get("type")

# #             if event_type == "setup":
# #                 call_sid = message.get("callSid")
# #                 if not call_sid:
# #                     logger.error("Missing callSid in setup")
# #                     continue
# #                 CONVERSATION_HISTORY[call_sid] = [{"role": "system", "content": SYSTEM_PROMPT}]
# #                 logger.info("Setup for CallSid %s", call_sid)
# #                 continue

# #             elif event_type == "prompt":
# #                 user_text = message.get("voicePrompt")
# #                 if not user_text or not user_text.strip():
# #                     logger.error("Missing or empty voicePrompt")
# #                     continue

# #                 #if there's an ongoing response, interrupt it
# #                 if current_response_task and not current_response_task.done():
# #                     interrupted = True
# #                     await asyncio.sleep(0)

# #                 _, _, lang_spitch = LANGUAGE_SELECTION.get(call_sid, ("English", "en-US", "en"))

# #                 try:
# #                     if lang_spitch != "en":
# #                         english_text = spitch_translate(user_text, source=lang_spitch, target="en")
# #                     else:
# #                         english_text = user_text

# #                     history = CONVERSATION_HISTORY.get(call_sid, [{"role": "system", "content": SYSTEM_PROMPT}])
# #                     history.append({"role": "user", "content": english_text})

# #                     #reset interrupted for new response
# #                     interrupted = False

# #                     async def stream_response():
# #                         nonlocal interrupted, history
# #                         reply_en = ""
# #                         try:
# #                             stream = await openrouter_client.chat.completions.create(
# #                                 model=MODEL,
# #                                 messages=history,
# #                                 stream=True
# #                             )
# #                             async for chunk in stream:
# #                                 if interrupted:
# #                                     logger.info("Response interrupted")
# #                                     break
# #                                 delta = chunk.choices[0].delta.content or ""
# #                                 if delta:
# #                                     reply_en += delta
# #                                     if lang_spitch != "en":
# #                                         #translate delta (may not be perfect, but for streaming)
# #                                         partial_local = spitch_translate(delta, source="en", target=lang_spitch)
# #                                     else:
# #                                         partial_local = delta
# #                                     await websocket.send_text(
# #                                         json.dumps({
# #                                             "type": "text",
# #                                             "token": partial_local,
# #                                             "last": False,
# #                                             "interruptible": True
# #                                         })
# #                                     )
# #                             if not interrupted:
# #                                 await websocket.send_text(
# #                                     json.dumps({
# #                                         "type": "text",
# #                                         "token": "",
# #                                         "last": True
# #                                     })
# #                                 )
# #                                 history.append({"role": "assistant", "content": reply_en})
# #                                 CONVERSATION_HISTORY[call_sid] = history[-20:]
# #                             #if interrupted, do not add assistant message to history
# #                         except Exception as e:
# #                             logger.error(f"Error in stream_response: {e}")
# #                             if not interrupted:
# #                                 await websocket.send_text(
# #                                     json.dumps({
# #                                         "type": "text",
# #                                         "token": "Sorry, an error occurred. Please try again.",
# #                                         "last": True
# #                                     })
# #                                 )

# #                     current_response_task = asyncio.create_task(stream_response())

# #                 except Exception as e:
# #                     logger.error(f"Error processing prompt: {e}")
# #                     await websocket.send_text(
# #                         json.dumps({
# #                             "type": "text",
# #                             "token": "Sorry, an error occurred. Please try again.",
# #                             "last": True
# #                         })
# #                     )
# #                 continue

# #             elif event_type == "speaker":
# #                 if message.get("event") == "clientSpeaking":
# #                     logger.info("Client speaking detected - potential interruption")
# #                     interrupted = True
# #                 continue

# #             elif event_type == "dtmf":
# #                 logger.info("DTMF received: %s", message)
# #                 continue

# #             elif event_type == "error":
# #                 logger.error("Error received: %s", message)
# #                 continue

# #             elif event_type == "call_ended":
# #                 LANGUAGE_SELECTION.pop(call_sid, None)
# #                 CONVERSATION_HISTORY.pop(call_sid, None)
# #                 logger.info("Cleaned up for CallSid %s", call_sid)
# #                 continue

# #             logger.warning("Unknown event type: %s", event_type)

# #     except Exception as e:
# #         logger.error("WebSocket error: %s", e)
# #     finally:
# #         if current_response_task:
# #             current_response_task.cancel()
# #         receive_task.cancel()
# #         if call_sid:
# #             LANGUAGE_SELECTION.pop(call_sid, None)
# #             CONVERSATION_HISTORY.pop(call_sid, None)
# #         await websocket.close()


