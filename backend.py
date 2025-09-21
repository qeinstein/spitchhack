import os
import logging
import structlog
from typing import Dict, Any
from fastapi import FastAPI, Request, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse
from twilio.request_validator import RequestValidator
from dotenv import load_dotenv
from spitch import Spitch
import google.generativeai as genai
from urllib.parse import urlparse
import json
import asyncio
import base64

# ---- Config ----
load_dotenv()

app = FastAPI()

# Validate environment variables
required_vars = [
    "SPITCH_API_KEY",
    "GEMINI_API_KEY",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "BASE_URL",
    "CONVERSATION_SERVICE_SID"
]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing environment variable: {var}")

SPITCH_API_KEY = os.getenv("SPITCH_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
MODEL = os.getenv("MODEL", "gemini-2.5-flash")
SYSTEM_PROMPT = "You are a helpful assistant named Proxy.Respond Like a normal human. This conversation is being translated to voice, so answer carefully. When you respond, please spell out all numbers, for example twenty not 20. Do not include emojis in your responses. Do not include bullet points, asterisks, or special symbols."

# ---- Logging Setup ----
logging.basicConfig(level=logging.INFO)
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger("conversation-relay")

# ---- Clients ----
try:
    spitch_client = Spitch(api_key=SPITCH_API_KEY)
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_client = genai.GenerativeModel(MODEL)
except Exception as e:
    logger.error("Failed to initialize clients", error=str(e))
    raise RuntimeError(f"Failed to initialize clients: {e}")

# ---- Language map ----
LANGUAGE_MAP = {
    "1": ("Yoruba", "yo-NG", "yo"),
    "2": ("Igbo", "ig-NG", "ig"),
    "3": ("Hausa", "ha-NG", "ha"),
    "4": ("English", "en-US", "en")
}

LANGUAGE_SELECTION: Dict[str, tuple] = {}
CONVERSATION_HISTORY: Dict[str, list] = {}

# ---- Helpers ----
def spitch_translate(text: str, source: str, target: str) -> str:
    try:
        resp = spitch_client.text.translate(text=text, source=source, target=target)
        t = getattr(resp, "text", None)
        if not t:
            raise RuntimeError("Empty translation from Spitch")
        logger.info("Translation successful", source=source, target=target, text_preview=t[:50])
        return t
    except Exception as e:
        logger.error("Spitch translation failed", source=source, target=target, error=str(e))
        raise RuntimeError(f"Translation error: {e}")

async def gemini_chat_reply(messages: list) -> str:
    try:
        prompt = SYSTEM_PROMPT + "\n\n" + "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages
        )
        response = await gemini_client.generate_content_async(prompt)
        if not response.text:
            raise RuntimeError("Empty response from Gemini")
        logger.info("Gemini reply generated", reply_preview=response.text[:50])
        return response.text
    except Exception as e:
        logger.error("Gemini API error", error=str(e))
        return "Sorry, I couldn't process your request. Please try again."

# ---- Root endpoint ----
@app.get("/")
async def root():
    return {"message": "Welcome to the SpitchHack Voice Relay API. Use /health to check status."}

# ---- TwiML entry ----
@app.post("/voice")
async def voice_entry(request: Request):
    form_data = await request.form()
    signature = request.headers.get("X-Twilio-Signature", "")
    url = str(request.url)
    if not twilio_validator.validate(url, dict(form_data), signature):
        logger.error("Invalid Twilio signature", url=url)
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    twiml = VoiceResponse()
    gather = twiml.gather(
        num_digits=1,
        action="/process_language",
        method="POST",
        timeout=8
    )
    gather.say("Welcome to Proxy. For Yoruba press 1. For Igbo press 2. For Hausa press 3. For English press 4.")
    twiml.redirect("/process_language_fallback")
    logger.info("Voice endpoint called", url=url)
    return Response(content=str(twiml), media_type="application/xml")

@app.post("/process_language_fallback")
async def process_language_fallback(request: Request):
    twiml = VoiceResponse()
    twiml.say("Sorry, we did not receive input. Redirecting you back to language selection.")
    twiml.redirect("/voice")
    logger.info("Language selection fallback triggered")
    return Response(content=str(twiml), media_type="application/xml")

@app.post("/process_language")
async def process_language(request: Request, Digits: str = Form(None), CallSid: str = Form(None)):
    form_data = await request.form()
    signature = request.headers.get("X-Twilio-Signature", "")
    url = str(request.url)
    if not twilio_validator.validate(url, dict(form_data), signature):
        logger.error("Invalid Twilio signature", url=url)
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    twiml = VoiceResponse()
    if not (Digits and CallSid and Digits in LANGUAGE_MAP):
        twiml.say("Invalid selection or call ID. Please try again.")
        twiml.redirect("/voice")
        logger.error("Invalid language selection", digits=Digits, call_sid=CallSid)
        return Response(content=str(twiml), media_type="application/xml")

    lang_name, lang_code_twiml, lang_code_spitch = LANGUAGE_MAP[Digits]
    LANGUAGE_SELECTION[CallSid] = (lang_name, lang_code_twiml, lang_code_spitch)
    logger.info(
        "Language set",
        call_sid=CallSid,
        language=lang_name,
        twilio_code=lang_code_twiml,
        spitch_code=lang_code_spitch
    )

    twiml.say(f"You selected {lang_name}. Connecting you now.")
    connect = twiml.connect()
    conversation_relay = connect.conversation_relay(
        url=f"wss://{urlparse(BASE_URL).netloc}/relay",
        interruptible="any",
        report_input_during_agent_speech="any",
        debug="speaker-events"
    )
    conversation_relay.language(
        code=lang_code_twiml,
        transcription_provider="google"
    )
    return Response(content=str(twiml), media_type="application/xml")

# ---- Health check ----
@app.get("/health")
async def health():
    status = {"status": "ok", "services": {}}
    try:
        _ = spitch_translate("test", "en", "en")
        status["services"]["spitch"] = "ok"
    except Exception as e:
        status["services"]["spitch"] = f"down: {e}"
        logger.error("Spitch health check failed", error=str(e))
    try:
        _ = await gemini_chat_reply([{"role": "system", "content": "test"}])
        status["services"]["gemini"] = "ok"
    except Exception as e:
        status["services"]["gemini"] = f"down: {e}"
        logger.error("Gemini health check failed", error=str(e))
    logger.info("Health check completed", status=status)
    return status

@app.websocket("/relay")
async def relay_websocket(websocket: WebSocket):
    await websocket.accept()
    call_sid = None
    stream_sid = None
    message_queue = asyncio.Queue()
    interrupted = False
    current_response_task = None
    closed = False

    VOICES = {
        "yo": "kayode",
        "ig": "emeka",
        "ha": "danjuma",
        "en": "jude"
    }

    async def receiver():
        while not closed:
            try:
                data = await websocket.receive_text()
                try:
                    parsed = json.loads(data)
                    await message_queue.put(parsed)
                    logger.debug("WebSocket message received", raw_message=parsed)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON received", raw_data=data)
                    await message_queue.put(None)
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected", call_sid=call_sid)
                await message_queue.put(None)
                break
            except Exception as e:
                logger.error("Receiver error", error=str(e), call_sid=call_sid)
                await message_queue.put(None)
                break

    receive_task = asyncio.create_task(receiver())

    try:
        while not closed:
            message = await message_queue.get()
            if message is None:
                logger.info("Null message received, closing", call_sid=call_sid)
                break

            event_type = message.get("event")
            if not event_type:
                logger.warning("Unknown event received", raw_message=message, call_sid=call_sid)
                continue

            logger.debug("Processing WebSocket event", event_type=event_type, call_sid=call_sid)

            if event_type == "connected":
                call_sid = message.get("callSid")
                stream_sid = message.get("streamSid")
                if not call_sid or not stream_sid:
                    logger.error("Missing callSid or streamSid", message=message)
                    continue
                CONVERSATION_HISTORY[call_sid] = [{"role": "system", "content": SYSTEM_PROMPT}]
                logger.info("Connected", call_sid=call_sid, stream_sid=stream_sid)
                continue

            elif event_type == "start":
                logger.info("Stream started", stream_sid=message.get("streamSid"))
                continue

            elif event_type == "media":
                user_text = message.get("media", {}).get("payload")
                if not user_text or not user_text.strip():
                    logger.error("Missing or empty media payload", call_sid=call_sid)
                    continue

                _, _, lang_spitch = LANGUAGE_SELECTION.get(call_sid, ("English", "en-US", "en"))
                logger.info("Processing prompt", language=lang_spitch, text_preview=user_text[:50], call_sid=call_sid)

                try:
                    if lang_spitch != "en":
                        try:
                            user_text = base64.b64decode(user_text).decode("utf-8")
                            logger.info("Decoded base64 input", decoded_preview=user_text[:50], call_sid=call_sid)
                        except Exception:
                            pass

                    if lang_spitch != "en":
                        english_text = spitch_translate(user_text, source=lang_spitch, target="en")
                    else:
                        english_text = user_text

                    history = CONVERSATION_HISTORY.get(call_sid, [{"role": "system", "content": SYSTEM_PROMPT}])
                    history.append({"role": "user", "content": english_text})

                    interrupted = False

                    async def process_response():
                        nonlocal interrupted, history
                        try:
                            reply_en = await gemini_chat_reply(history)

                            if lang_spitch != "en":
                                reply_local = spitch_translate(reply_en, source="en", target=lang_spitch)
                            else:
                                reply_local = reply_en

                            voice_id = VOICES.get(lang_spitch, VOICES["en"])
                            logger.info(
                                "Generating Spitch audio",
                                language=lang_spitch,
                                voice=voice_id,
                                text_preview=reply_local[:50],
                                call_sid=call_sid
                            )
                            audio_resp = spitch_client.speech.generate(
                                text=reply_local,
                                language=lang_spitch,
                                voice=voice_id
                            )

                            audio_data = audio_resp.read()
                            if not audio_data:
                                raise RuntimeError("Spitch returned empty audio")

                            chunk_size = 4096
                            offset = 0
                            total_len = len(audio_data)
                            logger.info("Streaming audio", total_bytes=total_len, call_sid=call_sid)

                            while offset < total_len and not interrupted:
                                chunk = audio_data[offset:offset + chunk_size]
                                encoded_chunk = base64.b64encode(chunk).decode("utf-8")
                                await websocket.send_text(json.dumps({
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {"payload": encoded_chunk}
                                }))
                                offset += chunk_size
                                await asyncio.sleep(0.01)

                            if not interrupted:
                                await websocket.send_text(json.dumps({
                                    "event": "mark",
                                    "streamSid": stream_sid,
                                    "mark": {"name": "end"}
                                }))
                                logger.info("Audio stream complete", call_sid=call_sid)

                            history.append({"role": "assistant", "content": reply_en})
                            CONVERSATION_HISTORY[call_sid] = history[-20:]

                        except Exception as e:
                            logger.error("Error in process_response", error=str(e), call_sid=call_sid)
                            if not interrupted:
                                audio_data = spitch_client.speech.generate(
                                    text="Sorry, an error occurred. Please try again.",
                                    language="en",
                                    voice=VOICES["en"]
                                ).read()
                                await websocket.send_text(json.dumps({
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {"payload": base64.b64encode(audio_data).decode("utf-8")}
                                }))
                                await websocket.send_text(json.dumps({
                                    "event": "mark",
                                    "streamSid": stream_sid,
                                    "mark": {"name": "end"}
                                }))

                    current_response_task = asyncio.create_task(process_response())

                except Exception as e:
                    logger.error("Error processing prompt", error=str(e), call_sid=call_sid)
                    if not interrupted:
                        audio_data = spitch_client.speech.generate(
                            text="Sorry, an error occurred. Please try again.",
                            language="en",
                            voice=VOICES["en"]
                        ).read()
                        await websocket.send_text(json.dumps({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": base64.b64encode(audio_data).decode("utf-8")}
                        }))
                        await websocket.send_text(json.dumps({
                            "event": "mark",
                            "streamSid": stream_sid,
                            "mark": {"name": "end"}
                        }))
                continue

            elif event_type == "stop":
                logger.info("Stream stopped", stream_sid=message.get("streamSid"), call_sid=call_sid)
                break

            elif event_type == "info":
                logger.info("Info event received", message=message, call_sid=call_sid)
                continue

            logger.warning("Unknown event type", event_type=event_type, message=message, call_sid=call_sid)

    except Exception as e:
        logger.error("WebSocket error", error=str(e), call_sid=call_sid)
    finally:
        if current_response_task:
            current_response_task.cancel()
        receive_task.cancel()
        if call_sid:
            LANGUAGE_SELECTION.pop(call_sid, None)
            CONVERSATION_HISTORY.pop(call_sid, None)
        if not closed and websocket.client_state == 1:  # 1 = CONNECTED
            try:
                await websocket.send_text(json.dumps({
                    "event": "stop",
                    "streamSid": stream_sid
                }))
                await websocket.close()
                closed = True
                logger.info("WebSocket closed cleanly", call_sid=call_sid)
            except Exception as e:
                logger.error("Error closing WebSocket", error=str(e), call_sid=call_sid)















#works for english
# import os
# import logging
# from typing import Dict, Any
# from fastapi import FastAPI, Request, Form, HTTPException, WebSocket, WebSocketDisconnect
# from fastapi.responses import Response
# from fastapi import WebSocket
# from twilio.twiml.voice_response import VoiceResponse, Start, Stream
# from twilio.request_validator import RequestValidator
# from dotenv import load_dotenv
# from spitch import Spitch
# from openai import AsyncOpenAI
# from urllib.parse import urlparse
# import json
# import asyncio

# # ---- Config ----
# load_dotenv()

# app = FastAPI()

# # Validate environment variables
# required_vars = [
#     "SPITCH_API_KEY",
#     "OPENROUTER_API_KEY",
#     "TWILIO_ACCOUNT_SID",
#     "TWILIO_AUTH_TOKEN",
#     "BASE_URL",
#     "CONVERSATION_SERVICE_SID"
# ]
# for var in required_vars:
#     if not os.getenv(var):
#         raise RuntimeError(f"Missing environment variable: {var}")

# SPITCH_API_KEY = os.getenv("SPITCH_API_KEY")
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
# BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
# MODEL = os.getenv("MODEL", "gpt-4o-mini")
# VOICE_ID = os.getenv("VOICE_ID")
# SYSTEM_PROMPT = "You are a helpful assistant named Proxy. This conversation is being translated to voice, so answer carefully. When you respond, please spell out all numbers, for example twenty not 20. Do not include emojis in your responses. Do not include bullet points, asterisks, or special symbols."

# # ---- Clients ----
# try:
#     spitch_client = Spitch(api_key=SPITCH_API_KEY)
#     openrouter_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
# except Exception as e:
#     raise RuntimeError(f"Failed to initialize clients: {e}")

# # ---- App setup ----
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("conversation-relay")
# app = FastAPI()
# twilio_validator = RequestValidator(TWILIO_AUTH_TOKEN)

# # ---- Language map ----
# # LANGUAGE_MAP = {
# #     "1": ("Yoruba", "yo"),
# #     "2": ("Igbo", "ig"),
# #     "3": ("Hausa", "ha"),
# #     "4": ("English", "en")
# # }
# LANGUAGE_MAP = {
#     "1": ("Yoruba", "yo-NG", "yo"), #aiit so this is now (BCP-47 code, spitch code)
#     "2": ("Igbo", "ig-NG", "ig"),
#     "3": ("Hausa", "ha-NG", "ha"),
#     "4": ("English", "en-US", "en")
# }

# # LANGUAGE_SELECTION: Dict[str, str] = {}  # CallSid -> lang code
# LANGUAGE_SELECTION: Dict[str, tuple] = {}  # CallSid -> (lang_name, lang_code_twiml, lang_code_spitch) instead of just lang_code
# CONVERSATION_HISTORY: Dict[str, list] = {}  # CallSid -> list of {"role": str, "content": str}
# #the conversation history is that Proxy can retain conversation cintext between calls

# # ---- Helpers ----
# def spitch_translate(text: str, source: str, target: str) -> str:
#     """
#     Translate `text` from `source` language to `target` using Spitch API.
#     """
#     try:
#         resp = spitch_client.text.translate(text=text, source=source, target=target)
#         t = getattr(resp, "text", None)
#         if not t:
#             raise RuntimeError("Empty translation from Spitch")
#         return t
#     except Exception as e:
#         logger.error(f"Spitch translation failed: {e}")
#         raise RuntimeError(f"Translation error: {e}")

# def openrouter_chat_reply(messages: list) -> str:
#     try:
#         resp = openrouter_client.chat.completions.create(model=MODEL, messages=messages)
#         return resp.choices[0].message.content
#     except Exception as e:
#         logger.error(f"OpenRouter API error: {e}")
#         return "Sorry, I couldn't process your request. Please try again."

# # ---- Root endpoint ----
# @app.get("/")
# async def root():
#     return {"message": "Welcome to the SpitchHack Voice Relay API. Use /health to check status."}

# # ---- TwiML entry ----
# @app.post("/voice")
# async def voice_entry(request: Request):
#     # Validate Twilio webhook
#     form_data = await request.form()
#     signature = request.headers.get("X-Twilio-Signature", "")
#     url = str(request.url)
#     if not twilio_validator.validate(url, dict(form_data), signature):
#         raise HTTPException(status_code=403, detail="Invalid Twilio signature")

#     twiml = VoiceResponse()
#     # Gather language selection
#     gather = twiml.gather(
#         num_digits=1,
#         action="/process_language",
#         method="POST",
#         timeout=8
#     )
#     gather.say("Welcome to Proxy. For Yoruba press 1. For Igbo press 2. For Hausa press 3. For English press 4.")
#     # If gather doesn't get input:
#     twiml.redirect("/process_language_fallback")

#     return Response(content=str(twiml), media_type="application/xml")

# @app.post("/process_language_fallback")
# async def process_language_fallback(request: Request):
#     twiml = VoiceResponse()
#     twiml.say("Sorry, we did not receive input. Redirecting you back to language selection.")
#     twiml.redirect("/voice")
#     return Response(content=str(twiml), media_type="application/xml")

# @app.post("/process_language")
# async def process_language(request: Request, Digits: str = Form(None), CallSid: str = Form(None)):
#     # Validate Twilio webhook
#     form_data = await request.form()
#     signature = request.headers.get("X-Twilio-Signature", "")
#     url = str(request.url)
#     if not twilio_validator.validate(url, dict(form_data), signature):
#         raise HTTPException(status_code=403, detail="Invalid Twilio signature")

#     twiml = VoiceResponse()
#     if not (Digits and CallSid and Digits in LANGUAGE_MAP):
#         twiml.say("Invalid selection or call ID. Please try again.")
#         twiml.redirect("/voice")
#         return Response(content=str(twiml), media_type="application/xml")

#     lang_name, lang_code_twiml, lang_code_spitch = LANGUAGE_MAP[Digits]
#     LANGUAGE_SELECTION[CallSid] = (lang_name, lang_code_twiml, lang_code_spitch)
#     logger.info(
#         "Language set for CallSid %s -> %s (Twilio code: %s, Spitch code: %s)",
#         CallSid, lang_name, lang_code_twiml, lang_code_spitch
#     )

#     twiml.say(f"You selected {lang_name}. Connecting you now.")

#     # # Fixing BASE_URL parsing
#     # parsed = urlparse(BASE_URL)
#     # host = parsed.netloc or parsed.path  # in case BASE_URL had no scheme
#     # # Use websocket URL derived from BASE_URL
#     # ws_url = f"wss://{host}/relay"

#     # start = Start()
#     # stream = Stream(url=ws_url)
#     # start.append(stream)
#     # twiml.append(start)

#     connect = twiml.connect()
#     conversation_relay = connect.conversation_relay(
#         url=f"wss://{urlparse(BASE_URL).netloc}/relay",
#         interruptible="any",
#         report_input_during_agent_speech="any",
#         debug="speaker-events"
#         )
#     language = conversation_relay.language(
#         code=lang_code_twiml,
#         tts_provider="elevenlabs",
#         voice=VOICE_ID,
#         transcription_provider="google"
#         #both google and elevenlabs support yoruba/igbo/hausa
#     )

#     return Response(content=str(twiml), media_type="application/xml")


# # ---- Health check ----
# @app.get("/health")
# async def health():
#     status = {"status": "ok", "services": {}}
#     # Test Spitch translate
#     try:
#         _ = spitch_translate("test", "en", "en")
#         status["services"]["spitch"] = "ok"
#     except Exception as e:
#         status["services"]["spitch"] = f"down: {e}"
#     # Test OpenRouter
#     try:
#         _ = openrouter_client.chat.completions.create(model=MODEL, messages=[{"role": "system", "content": "test"}])
#         status["services"]["openrouter"] = "ok"
#     except Exception as e:
#         status["services"]["openrouter"] = f"down: {e}"
#     return status

# @app.websocket("/relay")
# async def relay_websocket(websocket: WebSocket):
#     await websocket.accept()
#     call_sid = None
#     message_queue = asyncio.Queue()
#     interrupted = False
#     current_response_task = None

#     # Mocked voices for each language (replace with real Spitch voice IDs)
#     VOICES = {
#         "yo": "femi",     # Yoruba
#         "ig": "chioma",   # Igbo
#         "ha": "aminu",    # Hausa
#         "en": "jude"   # English
#     }

#     async def receiver():
#         while True:
#             try:
#                 data = await websocket.receive_text()
#                 await message_queue.put(json.loads(data))
#             except WebSocketDisconnect:
#                 await message_queue.put(None)
#                 break
#             except Exception as e:
#                 logger.error(f"Receiver error: {e}")
#                 await message_queue.put(None)
#                 break

#     receive_task = asyncio.create_task(receiver())

#     try:
#         while True:
#             message = await message_queue.get()
#             if message is None:
#                 break

#             event_type = message.get("type")
#             logger.debug("WebSocket event: %s", message)

#             if event_type == "setup":
#                 call_sid = message.get("callSid")
#                 if not call_sid:
#                     logger.error("Missing callSid in setup")
#                     continue
#                 CONVERSATION_HISTORY[call_sid] = [{"role": "system", "content": SYSTEM_PROMPT}]
#                 logger.info("Setup for CallSid %s", call_sid)
#                 continue

#             elif event_type == "prompt":
#                 user_text = message.get("voicePrompt")
#                 if not user_text or not user_text.strip():
#                     logger.error("Missing or empty voicePrompt")
#                     continue

#                 _, _, lang_spitch = LANGUAGE_SELECTION.get(call_sid, ("English", "en-US", "en"))

#                 try:
#                     # Translate input → English
#                     if lang_spitch != "en":
#                         english_text = spitch_translate(user_text, source=lang_spitch, target="en")
#                     else:
#                         english_text = user_text

#                     history = CONVERSATION_HISTORY.get(call_sid, [{"role": "system", "content": SYSTEM_PROMPT}])
#                     history.append({"role": "user", "content": english_text})

#                     # reset interrupted for new response
#                     interrupted = False

#                     async def process_response():
#                         nonlocal interrupted, history
#                         reply_en = ""

#                         try:
#                             # get final LLM response in English
#                             resp = await openrouter_client.chat.completions.create(
#                                 model=MODEL,
#                                 messages=history
#                             )
#                             reply_en = resp.choices[0].message.content

#                             # Translate back to user language
#                             if lang_spitch != "en":
#                                 reply_local = spitch_translate(reply_en, source="en", target=lang_spitch)
#                             else:
#                                 reply_local = reply_en

#                             # TTS via Spitch
#                             voice_id = VOICES.get(lang_spitch, VOICES["en"])
#                             audio_resp = spitch_client.speech.generate(
#                                 text=reply_local,
#                                 language=lang_spitch,
#                                 voice=voice_id
#                             )

#                             # Stream audio back to Twilio
#                             chunk_size = 3200  # adjust depending on Twilio format
#                             while True:
#                                 chunk = audio_resp.read(chunk_size)
#                                 if not chunk:
#                                     break
#                                 await websocket.send_bytes(chunk)

#                             # End of stream marker
#                             await websocket.send_text(json.dumps({
#                                 "type": "audio",
#                                 "last": True
#                             }))

#                             # Save history
#                             history.append({"role": "assistant", "content": reply_en})
#                             CONVERSATION_HISTORY[call_sid] = history[-20:]

#                         except Exception as e:
#                             logger.error(f"Error in process_response: {e}")
#                             if not interrupted:
#                                 await websocket.send_text(json.dumps({
#                                     "type": "text",
#                                     "token": "Sorry, an error occurred. Please try again.",
#                                     "last": True
#                                 }))

#                     current_response_task = asyncio.create_task(process_response())

#                 except Exception as e:
#                     logger.error(f"Error processing prompt: {e}")
#                     await websocket.send_text(json.dumps({
#                         "type": "text",
#                         "token": "Sorry, an error occurred. Please try again.",
#                         "last": True
#                     }))
#                 continue

#             elif event_type == "speaker":
#                 if message.get("event") == "clientSpeaking":
#                     logger.info("Client speaking detected - potential interruption")
#                     interrupted = True
#                 continue

#             elif event_type == "dtmf":
#                 logger.info("DTMF received: %s", message)
#                 continue

#             elif event_type == "error":
#                 logger.error("Error received: %s", message)
#                 continue

#             elif event_type == "call_ended":
#                 LANGUAGE_SELECTION.pop(call_sid, None)
#                 CONVERSATION_HISTORY.pop(call_sid, None)
#                 logger.info("Cleaned up for CallSid %s", call_sid)
#                 continue

#             logger.warning("Unknown event type: %s", event_type)

#     except Exception as e:
#         logger.error("WebSocket error: %s", e)
#     finally:
#         if current_response_task:
#             current_response_task.cancel()
#         receive_task.cancel()
#         if call_sid:
#             LANGUAGE_SELECTION.pop(call_sid, None)
#             CONVERSATION_HISTORY.pop(call_sid, None)
#         await websocket.close()

# @app.websocket("/relay")
# async def relay_websocket(websocket: WebSocket):
#     await websocket.accept()
#     call_sid = None
#     message_queue = asyncio.Queue()
#     interrupted = False
#     current_response_task = None

#     async def receiver():
#         while True:
#             try:
#                 data = await websocket.receive_text()
#                 await message_queue.put(json.loads(data))
#             except WebSocketDisconnect:
#                 await message_queue.put(None)
#                 break
#             except Exception as e:
#                 logger.error(f"Receiver error: {e}")
#                 await message_queue.put(None)
#                 break

#     receive_task = asyncio.create_task(receiver())

#     try:
#         while True:
#             message = await message_queue.get()
#             if message is None:
#                 break

#             logger.debug("WebSocket event: %s", message)
#             event_type = message.get("type")

#             if event_type == "setup":
#                 call_sid = message.get("callSid")
#                 if not call_sid:
#                     logger.error("Missing callSid in setup")
#                     continue
#                 CONVERSATION_HISTORY[call_sid] = [{"role": "system", "content": SYSTEM_PROMPT}]
#                 logger.info("Setup for CallSid %s", call_sid)
#                 continue

#             elif event_type == "prompt":
#                 user_text = message.get("voicePrompt")
#                 if not user_text or not user_text.strip():
#                     logger.error("Missing or empty voicePrompt")
#                     continue

#                 #if there's an ongoing response, interrupt it
#                 if current_response_task and not current_response_task.done():
#                     interrupted = True
#                     await asyncio.sleep(0)

#                 _, _, lang_spitch = LANGUAGE_SELECTION.get(call_sid, ("English", "en-US", "en"))

#                 try:
#                     if lang_spitch != "en":
#                         english_text = spitch_translate(user_text, source=lang_spitch, target="en")
#                     else:
#                         english_text = user_text

#                     history = CONVERSATION_HISTORY.get(call_sid, [{"role": "system", "content": SYSTEM_PROMPT}])
#                     history.append({"role": "user", "content": english_text})

#                     #reset interrupted for new response
#                     interrupted = False

#                     async def stream_response():
#                         nonlocal interrupted, history
#                         reply_en = ""
#                         try:
#                             stream = await openrouter_client.chat.completions.create(
#                                 model=MODEL,
#                                 messages=history,
#                                 stream=True
#                             )
#                             async for chunk in stream:
#                                 if interrupted:
#                                     logger.info("Response interrupted")
#                                     break
#                                 delta = chunk.choices[0].delta.content or ""
#                                 if delta:
#                                     reply_en += delta
#                                     if lang_spitch != "en":
#                                         #translate delta (may not be perfect, but for streaming)
#                                         partial_local = spitch_translate(delta, source="en", target=lang_spitch)
#                                     else:
#                                         partial_local = delta
#                                     await websocket.send_text(
#                                         json.dumps({
#                                             "type": "text",
#                                             "token": partial_local,
#                                             "last": False,
#                                             "interruptible": True
#                                         })
#                                     )
#                             if not interrupted:
#                                 await websocket.send_text(
#                                     json.dumps({
#                                         "type": "text",
#                                         "token": "",
#                                         "last": True
#                                     })
#                                 )
#                                 history.append({"role": "assistant", "content": reply_en})
#                                 CONVERSATION_HISTORY[call_sid] = history[-20:]
#                             #if interrupted, do not add assistant message to history
#                         except Exception as e:
#                             logger.error(f"Error in stream_response: {e}")
#                             if not interrupted:
#                                 await websocket.send_text(
#                                     json.dumps({
#                                         "type": "text",
#                                         "token": "Sorry, an error occurred. Please try again.",
#                                         "last": True
#                                     })
#                                 )

#                     current_response_task = asyncio.create_task(stream_response())

#                 except Exception as e:
#                     logger.error(f"Error processing prompt: {e}")
#                     await websocket.send_text(
#                         json.dumps({
#                             "type": "text",
#                             "token": "Sorry, an error occurred. Please try again.",
#                             "last": True
#                         })
#                     )
#                 continue

#             elif event_type == "speaker":
#                 if message.get("event") == "clientSpeaking":
#                     logger.info("Client speaking detected - potential interruption")
#                     interrupted = True
#                 continue

#             elif event_type == "dtmf":
#                 logger.info("DTMF received: %s", message)
#                 continue

#             elif event_type == "error":
#                 logger.error("Error received: %s", message)
#                 continue

#             elif event_type == "call_ended":
#                 LANGUAGE_SELECTION.pop(call_sid, None)
#                 CONVERSATION_HISTORY.pop(call_sid, None)
#                 logger.info("Cleaned up for CallSid %s", call_sid)
#                 continue

#             logger.warning("Unknown event type: %s", event_type)

#     except Exception as e:
#         logger.error("WebSocket error: %s", e)
#     finally:
#         if current_response_task:
#             current_response_task.cancel()
#         receive_task.cancel()
#         if call_sid:
#             LANGUAGE_SELECTION.pop(call_sid, None)
#             CONVERSATION_HISTORY.pop(call_sid, None)
#         await websocket.close()


#this commendted out block of code is for if you don't want Proxy to be able to be interrupted


# import os
# import logging
# from typing import Dict, Any
# from fastapi import FastAPI, Request, Form, HTTPException, WebSocket, WebSocketDisconnect
# from fastapi.responses import Response
# from fastapi import WebSocket
# from twilio.twiml.voice_response import VoiceResponse, Start, Stream
# from twilio.request_validator import RequestValidator
# from dotenv import load_dotenv
# from spitch import Spitch
# from openai import OpenAI
# from urllib.parse import urlparse
# import json

# # ---- Config ----
# load_dotenv()

# # Validate environment variables
# required_vars = [
#     "SPITCH_API_KEY",
#     "OPENROUTER_API_KEY",
#     "TWILIO_ACCOUNT_SID",
#     "TWILIO_AUTH_TOKEN",
#     "BASE_URL",
#     "CONVERSATION_SERVICE_SID"
# ]
# for var in required_vars:
#     if not os.getenv(var):
#         raise RuntimeError(f"Missing environment variable: {var}")

# SPITCH_API_KEY = os.getenv("SPITCH_API_KEY")
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
# BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
# MODEL = os.getenv("MODEL", "gpt-4o-mini")
# VOICE_ID = os.getenv("VOICE_ID")
# SYSTEM_PROMPT = "You are a helpful assistant named Proxy. This conversation is being translated to voice, so answer carefully. When you respond, please spell out all numbers, for example twenty not 20. Do not include emojis in your responses. Do not include bullet points, asterisks, or special symbols."

# # ---- Clients ----
# try:
#     spitch_client = Spitch(api_key=SPITCH_API_KEY)
#     openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
# except Exception as e:
#     raise RuntimeError(f"Failed to initialize clients: {e}")

# # ---- App setup ----
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("conversation-relay")
# app = FastAPI()
# twilio_validator = RequestValidator(TWILIO_AUTH_TOKEN)

# # ---- Language map ----
# # LANGUAGE_MAP = {
# #     "1": ("Yoruba", "yo"),
# #     "2": ("Igbo", "ig"),
# #     "3": ("Hausa", "ha"),
# #     "4": ("English", "en")
# # }
# LANGUAGE_MAP = {
#     "1": ("Yoruba", "yo-NG", "yo"), #aiit so this is now (BCP-47 code, spitch code)
#     "2": ("Igbo", "ig-NG", "ig"),
#     "3": ("Hausa", "ha-NG", "ha"),
#     "4": ("English", "en-US", "en")
# }

# # LANGUAGE_SELECTION: Dict[str, str] = {}  # CallSid -> lang code
# LANGUAGE_SELECTION: Dict[str, tuple] = {}  # CallSid -> (lang_name, lang_code_twiml, lang_code_spitch) instead of just lang_code
# CONVERSATION_HISTORY: Dict[str, list] = {}  # CallSid -> list of {"role": str, "content": str}
# #the conversation history is that Proxy can retain conversation cintext between calls

# # ---- Helpers ----
# def spitch_translate(text: str, source: str, target: str) -> str:
#     """
#     Translate `text` from `source` language to `target` using Spitch API.
#     """
#     try:
#         resp = spitch_client.text.translate(text=text, source=source, target=target)
#         t = getattr(resp, "text", None)
#         if not t:
#             raise RuntimeError("Empty translation from Spitch")
#         return t
#     except Exception as e:
#         logger.error(f"Spitch translation failed: {e}")
#         raise RuntimeError(f"Translation error: {e}")

# def openrouter_chat_reply(messages: list) -> str:
#     try:
#         resp = openrouter_client.chat.completions.create(model=MODEL, messages=messages)
#         return resp.choices[0].message.content
#     except Exception as e:
#         logger.error(f"OpenRouter API error: {e}")
#         return "Sorry, I couldn't process your request. Please try again."

# # ---- Root endpoint ----
# @app.get("/")
# async def root():
#     return {"message": "Welcome to the SpitchHack Voice Relay API. Use /health to check status."}

# # ---- TwiML entry ----
# @app.post("/voice")
# async def voice_entry(request: Request):
#     # Validate Twilio webhook
#     form_data = await request.form()
#     signature = request.headers.get("X-Twilio-Signature", "")
#     url = str(request.url)
#     if not twilio_validator.validate(url, dict(form_data), signature):
#         raise HTTPException(status_code=403, detail="Invalid Twilio signature")

#     twiml = VoiceResponse()
#     # Gather language selection
#     gather = twiml.gather(
#         num_digits=1,
#         action="/process_language",
#         method="POST",
#         timeout=8
#     )
#     gather.say("Welcome to Proxy. For Yoruba press 1. For Igbo press 2. For Hausa press 3. For English press 4.")
#     # If gather doesn't get input:
#     twiml.redirect("/process_language_fallback")

#     return Response(content=str(twiml), media_type="application/xml")

# @app.post("/process_language_fallback")
# async def process_language_fallback(request: Request):
#     twiml = VoiceResponse()
#     twiml.say("Sorry, we did not receive input. Redirecting you back to language selection.")
#     twiml.redirect("/voice")
#     return Response(content=str(twiml), media_type="application/xml")

# @app.post("/process_language")
# async def process_language(request: Request, Digits: str = Form(None), CallSid: str = Form(None)):
#     # Validate Twilio webhook
#     form_data = await request.form()
#     signature = request.headers.get("X-Twilio-Signature", "")
#     url = str(request.url)
#     if not twilio_validator.validate(url, dict(form_data), signature):
#         raise HTTPException(status_code=403, detail="Invalid Twilio signature")

#     twiml = VoiceResponse()
#     if not (Digits and CallSid and Digits in LANGUAGE_MAP):
#         twiml.say("Invalid selection or call ID. Please try again.")
#         twiml.redirect("/voice")
#         return Response(content=str(twiml), media_type="application/xml")

#     lang_name, lang_code_twiml, lang_code_spitch = LANGUAGE_MAP[Digits]
#     LANGUAGE_SELECTION[CallSid] = (lang_name, lang_code_twiml, lang_code_spitch)
#     logger.info("Language set for CallSid %s -> %s", CallSid, lang_name, lang_code_twiml, lang_code_spitch)

#     twiml.say(f"You selected {lang_name}. Connecting you now.")

#     # # Fixing BASE_URL parsing
#     # parsed = urlparse(BASE_URL)
#     # host = parsed.netloc or parsed.path  # in case BASE_URL had no scheme
#     # # Use websocket URL derived from BASE_URL
#     # ws_url = f"wss://{host}/relay"

#     # start = Start()
#     # stream = Stream(url=ws_url)
#     # start.append(stream)
#     # twiml.append(start)

#     connect = twiml.connect()
#     conversation_relay = connect.conversation_relay(url=f"wss://{urlparse(BASE_URL).netloc}/relay")
#     language = conversation_relay.language(
#         code=lang_code_twiml,
#         tts_provider="elevenlabs",
#         voice=VOICE_ID,
#         transcription_provider="google"
#         #both google and elevenlabs support yoruba/igbo/hausa
#     )

#     return Response(content=str(twiml), media_type="application/xml")


# # ---- Health check ----
# @app.get("/health")
# async def health():
#     status = {"status": "ok", "services": {}}
#     # Test Spitch translate
#     try:
#         _ = spitch_translate("test", "en", "en")
#         status["services"]["spitch"] = "ok"
#     except Exception as e:
#         status["services"]["spitch"] = f"down: {e}"
#     # Test OpenRouter
#     try:
#         _ = openrouter_client.chat.completions.create(model=MODEL, messages=[{"role": "system", "content": "test"}])
#         status["services"]["openrouter"] = "ok"
#     except Exception as e:
#         status["services"]["openrouter"] = f"down: {e}"
#     return status


# @app.websocket("/relay")
# async def relay_websocket(websocket: WebSocket):
#     await websocket.accept()
#     call_sid = None
#     try:
#         while True:
#             data = await websocket.receive_text()
#             message = json.loads(data)
#             logger.debug("WebSocket event: %s", message)
#             event_type = message.get("type")

#             if event_type == "setup":
#                 call_sid = message.get("callSid")
#                 if not call_sid:
#                     logger.error("Missing callSid in setup")
#                     continue
#                 websocket.call_sid = call_sid
#                 CONVERSATION_HISTORY[call_sid] = [{"role": "system", "content": SYSTEM_PROMPT}]
#                 logger.info("Setup for CallSid %s", call_sid)
#                 continue

#             if event_type == "prompt":
#                 user_text = message.get("voicePrompt")
#                 if not user_text or not user_text.strip():
#                     logger.error("Missing or empty voicePrompt")
#                     continue

#                 _, _, lang_spitch = LANGUAGE_SELECTION.get(call_sid, ("English", "en-US", "en"))

#                 try:
#                     if lang_spitch != "en":
#                         english_text = spitch_translate(user_text, source=lang_spitch, target="en")
#                     else:
#                         english_text = user_text

#                     #get or create history
#                     history = CONVERSATION_HISTORY.get(call_sid, [{"role": "system", "content": SYSTEM_PROMPT}])
#                     history.append({"role": "user", "content": english_text})

#                     reply_en = openrouter_chat_reply(history)

#                     history.append({"role": "assistant", "content": reply_en})
#                     CONVERSATION_HISTORY[call_sid] = history[-20:]  #here, the messages in the conversation history were limited to twenty to reduce halluciantions and those type stuff

#                     if lang_spitch != "en":
#                         reply_local = spitch_translate(reply_en, source="en", target=lang_spitch)
#                     else:
#                         reply_local = reply_en

#                     await websocket.send_text(
#                         json.dumps({
#                             "type": "text",
#                             "token": reply_local,
#                             "last": True,
#                             "interruptible": True  
#                         })
#                     )

#                 except Exception as e:
#                     logger.error(f"Error processing prompt: {e}")
#                     await websocket.send_text(
#                         json.dumps({
#                             "type": "text",
#                             "token": "Sorry, could you take that again",
#                             "last": True
#                         })
#                     )
#                 continue

#             if event_type == "interrupt":
#                 logger.info("Interrupt received: %s", message)
#                 continue

#             if event_type == "error":
#                 logger.error("Error received: %s", message)
#                 continue

#             if event_type == "dtmf":
#                 logger.info("DTMF received: %s", message)
#                 continue

#             logger.warning("Unknown event type: %s", event_type)

#     except WebSocketDisconnect:
#         logger.info("WebSocket disconnected for CallSid %s", call_sid)
#     except Exception as e:
#         logger.error("WebSocket error: %s", e)
#     finally:
#         if call_sid:
#             LANGUAGE_SELECTION.pop(call_sid, None)
#             CONVERSATION_HISTORY.pop(call_sid, None)
#         await websocket.close()










# @app.websocket("/relay")
# async def relay_websocket(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         while True:
#             data = await websocket.receive_json()
#             logger.debug("WebSocket event: %s", data)
#             event_type = data.get("type")
#             call_sid = data.get("callSid")
#             if not call_sid:
#                 logger.error("Missing callSid in event")
#                 await websocket.send_json({"type": "noop"})
#                 continue

#             if event_type == "call_ended":
#                 LANGUAGE_SELECTION.pop(call_sid, None)
#                 logger.info("Cleaned up LANGUAGE_SELECTION for CallSid %s", call_sid)
#                 await websocket.send_json({"type": "noop"})
#                 continue

#             if event_type == "utterance":
#                 user_text = data.get("text")
#                 if not user_text or not user_text.strip():
#                     logger.error("Missing or empty text in utterance event")
#                     await websocket.send_json({"type": "noop"})
#                     continue

#                 # Optionally, filter overly repetitive or irrelevant utterances
#                 # Could track repeat counts per call_sid if needed

#                 lang = LANGUAGE_SELECTION.get(call_sid, "en")

#                 try:
#                     if lang != "en":
#                         english_text = spitch_translate(user_text, source=lang, target="en")
#                     else:
#                         english_text = user_text

#                     reply_en = openrouter_chat_reply([
#                         {"role": "system", "content": "You are a helpful assistant."},
#                         {"role": "user", "content": english_text}
#                     ])

#                     if lang != "en":
#                         reply_local = spitch_translate(reply_en, source="en", target=lang)
#                     else:
#                         reply_local = reply_en

#                     await websocket.send_json({"type": "reply", "text": reply_local})

#                 except Exception as e:
#                     logger.error(f"Error processing utterance: {e}")
#                     await websocket.send_json({"type": "reply", "text": "Sorry, an error occurred. Please try again."})

#             else:
#                 await websocket.send_json({"type": "noop"})

#     except Exception as e:
#         logger.error("WebSocket error: %s", e)
#     finally:
#         await websocket.close()

























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

