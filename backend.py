import os
import logging
import json
import uuid
import asyncio
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, Request, Form, HTTPException, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import google.generativeai as genai
from twilio.request_validator import RequestValidator
from twilio.twiml.voice_response import VoiceResponse

load_dotenv()

# ---------- Config ----------
REQUIRED_ENV = [
    "GEMINI_API_KEY",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "BASE_URL",
    "SPITCH_API_KEY",
]
for v in REQUIRED_ENV:
    if not os.getenv(v):
        raise RuntimeError(f"Missing environment variable: {v}")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
SPITCH_API_KEY = os.getenv("SPITCH_API_KEY")

MODEL = os.getenv("MODEeL", "gemini-2.5-flash")
VOICE_ID = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant named Proxy. This conversation is being translated to voice, so answer carefully. "
    "When you respond, please spell out all numbers, for example twenty not 20. Do not include emojis in your responses. "
    "Do not include bullet points, asterisks, or special symbols."
)

# Directory to store generated audio for Twilio to fetch
AUDIO_DIR = os.path.join(os.getcwd(), "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# ---------- Logging & App ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("conversation-relay")
app = FastAPI()
twilio_validator = RequestValidator(TWILIO_AUTH_TOKEN)

# ---------- Language maps ----------
# LANGUAGE_MAP: digit -> (lang_name, twiml_code, spitch_lang_code)
LANGUAGE_MAP = {
    "1": ("Yoruba", "yo-NG", "yo"),
    "2": ("Igbo", "ig-NG", "ig"),
    "3": ("Hausa", "ha-NG", "ha"),
    "4": ("English", "en-US", "en"),
}

# Spitch voice map (language -> voice name). Replace values with actual voice IDs/names from your Spitch account.
VOICE_MAP = {
    "yo": "yoruba_voice",   # example placeholder; replace with actual Spitch voice id for Yoruba
    "ig": "igbo_voice",
    "ha": "hausa_voice",
    "en": "jude",           # example: "sade" per your example
}

# State maps
LANGUAGE_SELECTION: Dict[str, tuple] = {}  # CallSid -> (lang_name, lang_code_twiml, spitch_lang_code)
CONVERSATION_HISTORY: Dict[str, List[Dict[str, str]]] = {}  # CallSid -> list of messages

# ---------- Gemini setup ----------
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to initialize Gemini client: {e}")

# ---------- Helpers: Spitch API calls ----------
SPITCH_BASE = "https://api.spi-tch.com/v1"


def _spitch_headers():
    return {
        "Authorization": f"Bearer {SPITCH_API_KEY}",
        "Content-Type": "application/json"
    }


def _write_audio_file(content: bytes, ext: str = "wav") -> str:
    filename = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(AUDIO_DIR, filename)
    with open(path, "wb") as f:
        f.write(content)
    return path


async def spitch_translate(text: str, source: str, target: str, timeout: int = 15) -> str:
    """
    Calls Spitch translate endpoint to translate `text` from `source` -> `target`.
    """
    payload = {"source": source, "target": target, "text": text}
    url = f"{SPITCH_BASE}/translate"
    try:
        def _req():
            r = requests.post(url, json=payload, headers=_spitch_headers(), timeout=timeout)
            r.raise_for_status()
            return r.json()
        resp_json = await asyncio.to_thread(_req)
        # Spitch translate response structure may vary; assume {"translatedText": "..."} or {"text": "..."}
        translated = resp_json.get("translatedText") or resp_json.get("text") or resp_json.get("translation") or ""
        if not translated:
            # attempt to return first string value found
            for v in resp_json.values():
                if isinstance(v, str) and v.strip():
                    translated = v
                    break
        return translated or text
    except Exception as e:
        logger.error("Spitch translate failed: %s", e, exc_info=True)
        # On failure, fallback to original text to avoid blocking conversation
        return text


async def spitch_synthesize(text: str, language: str, voice: Optional[str] = None, ext: str = "wav", timeout: int = 30) -> Optional[str]:
    """
    Calls Spitch synthesize endpoint and saves returned audio to disk, returning a public URL.
    """
    url = f"{SPITCH_BASE}/synthesize"
    payload = {
        "language": language,
        "voice": voice or VOICE_MAP.get(language, "sade"),
        "text": text
    }
    try:
        def _req():
            r = requests.post(url, json=payload, headers=_spitch_headers(), timeout=timeout)
            r.raise_for_status()
            return r.content
        audio_bytes = await asyncio.to_thread(_req)
        path = _write_audio_file(audio_bytes, ext)
        # Expose via BASE_URL
        filename = os.path.basename(path)
        public_url = f"{BASE_URL}/audio/{filename}"
        return public_url
    except Exception as e:
        logger.error("Spitch synthesize failed: %s", e, exc_info=True)
        return None


async def spitch_transcribe_audio_file(file_path: str, language: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Demonstration helper if you wanted to call Spitch transcription on an audio file.
    Not used in primary flow because you asked to keep Twilio built-in transcription.
    """
    url = f"{SPITCH_BASE}/transcriptions"
    # The Spitch transcribe expects multipart/form-data with "content" file param and other payload fields.
    try:
        def _req():
            with open(file_path, "rb") as fh:
                files = {"content": fh}
                data = {"model": "mansa_v1", "language": language, "timestamp": "sentence"}
                r = requests.post(url, data=data, files=files, headers={"Authorization": f"Bearer {SPITCH_API_KEY}"}, timeout=timeout)
                r.raise_for_status()
                return r.json()
        return await asyncio.to_thread(_req)
    except Exception as e:
        logger.error("Spitch transcribe failed: %s", e, exc_info=True)
        return {}


# ---------- Helpers: Gemini chat ----------
async def gemini_chat_reply(messages: List[Dict[str, str]], target_language_code: str = "en") -> str:
    """
    Sends chat to Gemini. messages is a list of {"role": "system|user|assistant", "content": "..."}
    We ensure system prompt and language instruction are present.
    """
    try:
        # Append language instruction to system instruction and last user message
        # Convert to gemini message format
        gemini_messages = []
        for msg in messages:
            # keep system separately
            if msg["role"] == "system":
                continue
            role = "user" if msg["role"] == "user" else "model"
            gemini_messages.append({"role": role, "parts": [msg["content"]]})

        # Build model with system instruction
        model = genai.GenerativeModel(
            MODEL,
            system_instruction=SYSTEM_PROMPT + f" Always respond in {target_language_code} language."
        )

        if not gemini_messages:
            # fallback
            chat = model.start_chat()
            resp = await chat.send_message_async("Hello")
            return resp.text

        # Start a chat with all but last message in history, then send final message
        history = gemini_messages[:-1]
        final_msg = gemini_messages[-1]["parts"][0]

        chat = model.start_chat(history=history)
        response = await chat.send_message_async(final_msg)
        return response.text
    except Exception as e:
        logger.error(f"Gemini API error in gemini_chat_reply: {e}", exc_info=True)
        return "Sorry, I couldn't process your request. Please try again."


# ---------- Utilities ----------
def split_text_into_chunks(text: str, max_chars: int = 2500) -> List[str]:
    """
    Splits text roughly into chunks not exceeding max_chars, trying to split on sentence boundaries.
    """
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end == len(text):
            chunks.append(text[start:end].strip())
            break
        # Try to find last sentence boundary within this window
        window = text[start:end]
        idx = max(window.rfind("."), window.rfind("!"), window.rfind("?"))
        if idx <= 0:
            # no sentence boundary; just split at max_chars
            idx = end - start
        chunks.append(text[start:start + idx + 1].strip())
        start = start + idx + 1
    return [c for c in chunks if c]


# ---------- Routes ----------
@app.get("/")
async def root():
    return {"message": "Welcome to the Gemini + Spitch Voice Relay API. Use /health to check status."}


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """
    Serves generated audio files for Twilio to fetch/play. Ensure BASE_URL is public.
    """
    path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio not found")
    # Let FastAPI serve the file; Twilio will GET this URL
    return FileResponse(path, media_type="audio/wav")


@app.post("/voice")
async def voice_entry(request: Request):
    try:
        form_data = await request.form()
        signature = request.headers.get("X-Twilio-Signature", "")
        url = str(request.url)
        if not twilio_validator.validate(url, dict(form_data), signature):
            logger.warning("Invalid Twilio signature.")
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

        twiml = VoiceResponse()
        gather = twiml.gather(
            num_digits=1,
            action="/process_language",
            method="POST",
            timeout=8
        )
        gather.say("Welcome to Proxy. For Yoruba press one. For Igbo press two. For Hausa press three. For English press four.")
        twiml.redirect("/process_language_fallback")

        return Response(content=str(twiml), media_type="application/xml")
    except Exception as e:
        logger.error(f"Error in /voice endpoint: {e}", exc_info=True)
        twiml = VoiceResponse()
        twiml.say("An unexpected error occurred. Please try your call again.")
        return Response(content=str(twiml), media_type="application/xml", status_code=500)


@app.post("/process_language_fallback")
async def process_language_fallback(request: Request):
    try:
        twiml = VoiceResponse()
        twiml.say("Sorry, we did not receive any input. Redirecting you back to language selection.")
        twiml.redirect("/voice")
        return Response(content=str(twiml), media_type="application/xml")
    except Exception as e:
        logger.error(f"Error in /process_language_fallback: {e}", exc_info=True)
        twiml = VoiceResponse()
        twiml.say("An unexpected error occurred. Please try your call again.")
        return Response(content=str(twiml), media_type="application/xml", status_code=500)


@app.post("/process_language")
async def process_language(request: Request, Digits: str = Form(None), CallSid: str = Form(None)):
    try:
        form_data = await request.form()
        signature = request.headers.get("X-Twilio-Signature", "")
        url = str(request.url)
        if not twilio_validator.validate(url, dict(form_data), signature):
            logger.warning("Invalid Twilio signature.")
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

        twiml = VoiceResponse()
        if not (Digits and CallSid and Digits in LANGUAGE_MAP):
            logger.warning("Invalid language selection or missing CallSid. Digits: %s", Digits)
            twiml.say("Invalid selection or call ID. Please try again.")
            twiml.redirect("/voice")
            return Response(content=str(twiml), media_type="application/xml")

        lang_name, lang_code_twiml, spitch_lang = LANGUAGE_MAP[Digits]
        LANGUAGE_SELECTION[CallSid] = (lang_name, lang_code_twiml, spitch_lang)
        logger.info("Language set for CallSid %s -> %s", CallSid, lang_name)

        twiml.say(f"You selected {lang_name}. Connecting you now.")

        connect = twiml.connect()
        conversation_relay = connect.conversation_relay(
            url=f"wss://{urlparse(BASE_URL).netloc}/relay",
            interruptible="any",
            report_input_during_agent_speech="any",
            debug="speaker-events"
        )
        language = conversation_relay.language(
            code=lang_code_twiml,
            tts_provider="elevenlabs",  # Twilio conversation language block - Twilio will still use our audio files
            voice=VOICE_ID,
            transcription_provider="google"
        )

        return Response(content=str(twiml), media_type="application/xml")
    except Exception as e:
        logger.error(f"Error in /process_language endpoint: {e}", exc_info=True)
        twiml = VoiceResponse()
        twiml.say("An unexpected error occurred while processing your selection. Please try your call again.")
        return Response(content=str(twiml), media_type="application/xml", status_code=500)


@app.get("/health")
async def health():
    status = {"status": "ok", "services": {}}
    # Test Gemini (light)
    try:
        model = genai.GenerativeModel(MODEL)
        chat = model.start_chat()
        response = await chat.send_message_async("Health check")
        status["services"]["gemini"] = "ok"
    except Exception as e:
        status["services"]["gemini"] = f"down: {e}"
        logger.error(f"Health check failed for Gemini: {e}", exc_info=True)
    # Spitch (light)
    try:
        # A simple GET to Spitch translate (we won't send heavy payload)
        r = requests.get(f"{SPITCH_BASE}/health", headers=_spitch_headers(), timeout=5)
        status["services"]["spitch"] = "ok" if r.status_code == 200 else f"warning: {r.status_code}"
    except Exception as e:
        status["services"]["spitch"] = f"down: {e}"
    return status


# ---------- WebSocket relay ----------
@app.websocket("/relay")
async def relay_websocket(websocket: WebSocket):
    await websocket.accept()
    call_sid = None
    message_queue = asyncio.Queue()
    current_response_task: Optional[asyncio.Task] = None

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
                logger.error(f"Receiver error: {e}", exc_info=True)
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
                CONVERSATION_HISTORY[call_sid] = [{"role": "system", "content": SYSTEM_PROMPT}]
                logger.info("Setup complete for CallSid %s", call_sid)
                continue

            elif event_type == "prompt":
                # Twilio should be providing the transcribed text in 'voicePrompt'
                user_text = message.get("voicePrompt")
                if not user_text or not user_text.strip():
                    logger.warning("Received empty or missing voicePrompt. Skipping.")
                    continue

                # Cancel previous response task if any (interrupt)
                if current_response_task and not current_response_task.done():
                    logger.info("Interrupting previous response task.")
                    current_response_task.cancel()

                # Resolve language selection for this call
                lang_name, _, spitch_lang = LANGUAGE_SELECTION.get(call_sid, ("English", "en-US", "en"))
                history = CONVERSATION_HISTORY.get(call_sid, [{"role": "system", "content": SYSTEM_PROMPT}])
                history.append({"role": "user", "content": user_text})

                async def stream_response():
                    try:
                        # 1) If user language is not English, translate incoming user_text -> English
                        user_text_for_gemini = user_text
                        if spitch_lang != "en":
                            user_text_for_gemini = await spitch_translate(user_text, source=spitch_lang, target="en")
                            logger.info("Translated user input to English for Gemini: %s", user_text_for_gemini)

                        # 2) Send to Gemini (asking for an English response)
                        gemini_messages = history.copy()
                        # Ensure system present
                        if not any(m["role"] == "system" for m in gemini_messages):
                            gemini_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
                        # Replace last user content with the translated-to-english content (so model sees English)
                        gemini_messages[-1] = {"role": "user", "content": user_text_for_gemini}

                        gemini_reply = await gemini_chat_reply(gemini_messages, target_language_code="en")
                        logger.info("Gemini replied (in English): %s", gemini_reply)

                        # 3) If user language is not English, translate Gemini reply back to user's language
                        reply_for_user_text = gemini_reply
                        if spitch_lang != "en":
                            reply_for_user_text = await spitch_translate(gemini_reply, source="en", target=spitch_lang)
                            logger.info("Translated Gemini reply to user language (%s): %s", spitch_lang, reply_for_user_text)

                        # 4) Chunk response text to reduce latency and synthesize each chunk
                        chunks = split_text_into_chunks(reply_for_user_text, max_chars=2000)
                        logger.info("Reply split into %d chunk(s) for synthesis.", len(chunks))

                        for idx, chunk in enumerate(chunks):
                            # Choose voice for user's language
                            voice_name = VOICE_MAP.get(spitch_lang, VOICE_MAP.get("en", "sade"))
                            audio_url = await spitch_synthesize(chunk, language=spitch_lang, voice=voice_name)
                            if audio_url:
                                # Send audio event to Twilio via websocket so the conversation can play it
                                await websocket.send_text(json.dumps({
                                    "type": "audio",
                                    "audio_url": audio_url,
                                    "chunk_index": idx,
                                    "last": idx == len(chunks) - 1
                                }))
                                logger.info("Sent audio chunk %d to Twilio for CallSid %s", idx, call_sid)
                            else:
                                # If synthesis failed, fall back to sending text
                                await websocket.send_text(json.dumps({
                                    "type": "text",
                                    "token": chunk,
                                    "last": idx == len(chunks) - 1
                                }))
                                logger.warning("Synthesis failed; sent text chunk to Twilio instead.")

                        # 5) Update conversation history and persist last 20 messages
                        history.append({"role": "assistant", "content": gemini_reply})
                        CONVERSATION_HISTORY[call_sid] = history[-20:]
                        logger.info("Completed response for CallSid %s", call_sid)

                    except asyncio.CancelledError:
                        logger.warning("Response task was cancelled.")
                        # Optionally notify Twilio that response was interrupted
                        try:
                            await websocket.send_text(json.dumps({"type": "text", "token": "Response interrupted by user.", "last": True}))
                        except Exception:
                            pass
                        raise
                    except Exception as e:
                        logger.error(f"Error during response streaming: {e}", exc_info=True)
                        try:
                            await websocket.send_text(json.dumps({
                                "type": "text",
                                "token": "Sorry, a temporary error occurred. Please try again.",
                                "last": True
                            }))
                        except Exception:
                            pass

                current_response_task = asyncio.create_task(stream_response())
                continue

            elif event_type == "speaker":
                # If Twilio notifies the client is speaking, cancel current response
                if message.get("event") == "clientSpeaking" and current_response_task and not current_response_task.done():
                    logger.info("Client speaking detected. Cancelling ongoing response.")
                    current_response_task.cancel()
                continue

            elif event_type == "dtmf":
                logger.info("DTMF received: %s", message.get("digit"))
                continue

            elif event_type == "error":
                logger.error("Error from Twilio side: %s", message.get("error"))
                continue

            elif event_type == "call_ended":
                LANGUAGE_SELECTION.pop(call_sid, None)
                CONVERSATION_HISTORY.pop(call_sid, None)
                logger.info("Cleaned up state for ended call %s", call_sid)
                break

            logger.warning("Unknown event type received: %s", event_type)

    except Exception as e:
        logger.error(f"Unexpected error in WebSocket handler: {e}", exc_info=True)
    finally:
        if current_response_task:
            current_response_task.cancel()
        if not receive_task.done():
            receive_task.cancel()
        if call_sid:
            LANGUAGE_SELECTION.pop(call_sid, None)
            CONVERSATION_HISTORY.pop(call_sid, None)
        # Close websocket if still open
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


