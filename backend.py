import os
import logging
import json
import asyncio
import base64
from typing import Dict, Any
from fastapi import FastAPI, Request, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse
from twilio.request_validator import RequestValidator
from dotenv import load_dotenv
from spitch import Spitch
from openai import AsyncOpenAI
from urllib.parse import urlparse

# ---- Config ----
load_dotenv()

required_vars = [
    "SPITCH_API_KEY",
    "OPENROUTER_API_KEY",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "BASE_URL",
    "CONVERSATION_SERVICE_SID",
]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing environment variable: {var}")

SPITCH_API_KEY = os.getenv("SPITCH_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
MODEL = os.getenv("MODEL", "")
VOICE_ID = os.getenv("VOICE_ID")  # might be used for English if still using ElevenLabs

SYSTEM_PROMPT = (
    "You are a helpful assistant named Proxy. This conversation is being translated "
    "to voice, so answer carefully. When you respond, please spell out all numbers, "
    "for example twenty not twenty-zero. Do not include emojis in your responses. "
    "Do not include bullet points, asterisks, or special symbols. And respond in the language "
    "the user messages you in."
)

# ---- Per-language Spitch Voice IDs (placeholders) ----
SPITCH_VOICE_MAP = {
    "yo": "sade",   # e.g. "spitch_yo_female_1"
    "ig": "ngozi",
    "ha": "aminu",
    "en": "jude"   # maybe you want a Spitch voice for English too
}

# ---- Clients ----
spitch_client = Spitch(api_key=SPITCH_API_KEY)
openrouter_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# ---- App & logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("conversation-relay")
app = FastAPI()
twilio_validator = RequestValidator(TWILIO_AUTH_TOKEN)

# ---- Language map ----
LANGUAGE_MAP = {
    "1": ("Yoruba", "yo-NG", "yo"),
    "2": ("Igbo", "ig-NG", "ig"),
    "3": ("Hausa", "ha-NG", "ha"),
    "4": ("English", "en-US", "en")
}

LANGUAGE_SELECTION: Dict[str, tuple] = {}  # CallSid -> (lang_name, lang_code_twiml, lang_code_spitch)
CONVERSATION_HISTORY: Dict[str, list] = {}  # CallSid -> history

# ---- Helpers ----

def spitch_translate(text: str, source: str, target: str) -> str:
    resp = spitch_client.text.translate(text=text, source=source, target=target)
    t = getattr(resp, "text", None)
    if not t:
        raise RuntimeError("Empty translation from Spitch")
    return t

def spitch_tts(text: str, voice_id: str, lang: str = "en") -> bytes:
    url = "https://api.spi-tch.com/v1/synthesize"
    headers = {
        "Authorization": f"Bearer {SPITCH_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "language": lang,   # e.g. "yo", "ig", "ha", "en"
        "voice": voice_id,
        "text": text
    }
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Spitch TTS error: {resp.status_code} {resp.text}")
    return resp.content  # raw audio bytes

# ---- Endpoints ----

@app.post("/voice")
async def voice_entry(request: Request):
    form_data = await request.form()
    signature = request.headers.get("X-Twilio-Signature", "")
    url = str(request.url)
    if not twilio_validator.validate(url, dict(form_data), signature):
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
    return Response(content=str(twiml), media_type="application/xml")

@app.post("/process_language_fallback")
async def process_language_fallback(request: Request):
    twiml = VoiceResponse()
    twiml.say("Sorry, we did not receive input. Redirecting you back to language selection.")
    twiml.redirect("/voice")
    return Response(content=str(twiml), media_type="application/xml")

@app.post("/process_language")
async def process_language(request: Request, Digits: str = Form(None), CallSid: str = Form(None)):
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

    lang_name, lang_code_twiml, lang_code_spitch = LANGUAGE_MAP[Digits]
    LANGUAGE_SELECTION[CallSid] = (lang_name, lang_code_twiml, lang_code_spitch)
    logger.info("Language set for CallSid %s -> %s", CallSid, lang_name)

    twiml.say(f"You selected {lang_name}. Connecting you now.")

    connect = twiml.connect()
    conversation_relay = connect.conversation_relay(
        url=f"wss://{urlparse(BASE_URL).netloc}/relay",
        interruptible="any",
        report_input_during_agent_speech="any",
        debug="speaker-events"
    )

    # Only include <Language> for English (or any Twilio-supported code) or skip
    if lang_code_twiml == "en-US":
        conversation_relay.language(
            code="en-US",
            transcription_provider="google"
            # you could also include tts_provider if English via Twilio,
            # but since we’re using Spitch for all languages you might skip tts_provider here
        )
    else:
        logger.info("Skipping <Language> block for %s", lang_name)

    return Response(content=str(twiml), media_type="application/xml")


@app.websocket("/relay")
async def relay_websocket(websocket: WebSocket):
    await websocket.accept()
    call_sid = None
    message_queue = asyncio.Queue()
    interrupted = False
    current_response_task = None

    async def receiver():
        while True:
            try:
                data = await websocket.receive_text()
                await message_queue.put(json.loads(data))
            except WebSocketDisconnect:
                await message_queue.put(None)
                break
            except Exception as e:
                logger.error(f"Receiver error: {e}")
                await message_queue.put(None)
                break

    receive_task = asyncio.create_task(receiver())

    try:
        while True:
            message = await message_queue.get()
            if message is None:
                break

            event_type = message.get("type")

            if event_type == "setup":
                call_sid = message.get("callSid")
                CONVERSATION_HISTORY[call_sid] = [{"role": "system", "content": SYSTEM_PROMPT}]
                continue

            elif event_type == "prompt":
                user_text = message.get("voicePrompt")
                if not user_text or not user_text.strip():
                    continue

                if current_response_task and not current_response_task.done():
                    interrupted = True
                    await asyncio.sleep(0)

                _, _, lang_spitch = LANGUAGE_SELECTION.get(call_sid, ("English", "en-US", "en"))

                # Translate input from local language to English if needed
                if lang_spitch != "en":
                    try:
                        english_text = spitch_translate(user_text, source=lang_spitch, target="en")
                    except Exception as e:
                        logger.error(f"Translation error (input): {e}")
                        # fallback to user_text if translation fails
                        english_text = user_text
                else:
                    english_text = user_text

                history = CONVERSATION_HISTORY.get(call_sid, [{"role": "system", "content": SYSTEM_PROMPT}])
                history.append({"role": "user", "content": english_text})
                interrupted = False

                async def stream_response():
                    nonlocal interrupted, history
                    reply_en = ""
                    try:
                        stream = await openrouter_client.chat.completions.create(
                            model=MODEL,
                            messages=history,
                            stream=True
                        )
                        async for chunk in stream:
                            if interrupted:
                                break
                            delta = chunk.choices[0].delta.content or ""
                            if delta:
                                reply_en += delta
                                # Now translate back to local language if needed
                                if lang_spitch != "en":
                                    try:
                                        partial_local = spitch_translate(delta, source="en", target=lang_spitch)
                                    except Exception as e:
                                        logger.error(f"Translation error (output): {e}")
                                        partial_local = delta
                                else:
                                    partial_local = delta

                                # Now produce audio via Spitch TTS for partial_local
                                voice_for_lang = SPITCH_VOICE_MAP.get(lang_spitch, SPITCH_VOICE_MAP["en"])
                                try:
                                    audio_bytes = spitch_tts(partial_local, voice_for_lang, lang=lang_spitch)
                                    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                                    await websocket.send_text(
                                        json.dumps({
                                            "type": "audio",
                                            "audio": audio_b64,
                                            "last": False
                                        })
                                    )
                                except Exception as e:
                                    logger.error(f"TTS error: {e}")
                                    # fallback: send text (if any), so the flow can still continue
                                    await websocket.send_text(
                                        json.dumps({
                                            "type": "text",
                                            "token": partial_local,
                                            "last": False,
                                            "interruptible": True
                                        })
                                    )

                        if not interrupted:
                            # signal done
                            await websocket.send_text(
                                json.dumps({
                                    "type": "audio",
                                    "audio": "",
                                    "last": True
                                })
                            )
                            CONVERSATION_HISTORY[call_sid] = history[-20:]
                    except Exception as e:
                        logger.error(f"Error in stream_response: {e}")
                        if not interrupted:
                            await websocket.send_text(
                                json.dumps({
                                    "type": "audio",
                                    "audio": "",
                                    "last": True
                                })
                            )

                current_response_task = asyncio.create_task(stream_response())

            elif event_type == "speaker":
                if message.get("event") == "clientSpeaking":
                    interrupted = True
                continue

            elif event_type == "call_ended":
                LANGUAGE_SELECTION.pop(call_sid, None)
                CONVERSATION_HISTORY.pop(call_sid, None)
                continue

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if current_response_task:
            current_response_task.cancel()
        receive_task.cancel()
        if call_sid:
            LANGUAGE_SELECTION.pop(call_sid, None)
            CONVERSATION_HISTORY.pop(call_sid, None)
        await websocket.close()














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
# MODEL = os.getenv("MODEL", "")
# VOICE_ID = os.getenv("VOICE_ID")
# SYSTEM_PROMPT = "You are a helpful assistant named Proxy. This conversation is being translated to voice, so answer carefully. When you respond, please spell out all numbers, for example twenty not 20. Do not include emojis in your responses. Do not include bullet points, asterisks, or special symbols. and respond in the language the user messages you in"

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
# async def process_language(
#     request: Request,
#     Digits: str = Form(None),
#     CallSid: str = Form(None)
# ):
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
#     logger.info("Language set for CallSid %s -> %s", CallSid, lang_name)

#     twiml.say(f"You selected {lang_name}. Connecting you now.")

#     # ConversationRelay setup
#     connect = twiml.connect()
#     conversation_relay = connect.conversation_relay(
#         url=f"wss://{urlparse(BASE_URL).netloc}/relay",
#         interruptible="any",
#         report_input_during_agent_speech="any",
#         debug="speaker-events"
#     )

#     # ✅ Only add <Language> for English (Twilio supports en-US)
#     if lang_code_twiml == "en-US":
#         conversation_relay.language(
#             code="en-US",
#             tts_provider="elevenlabs",
#             transcription_provider="google"
#         )
#     else:
#         # ❌ Skip <Language> for Yoruba/Igbo/Hausa
#         # Their transcription + TTS will be handled via Spitch/ElevenLabs in your backend
#         logger.info("Skipping <Language> block for %s (unsupported by Twilio)", lang_name)

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