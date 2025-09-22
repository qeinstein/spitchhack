import os
import logging
from typing import Dict
from fastapi import FastAPI, Request, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Start, Stream
from twilio.request_validator import RequestValidator
from dotenv import load_dotenv
from spitch import Spitch
from openai import AsyncOpenAI
from urllib.parse import urlparse
import json
import audioop
import asyncio
import base64
from pydub import AudioSegment
from io import BytesIO

# ---- Config ----
load_dotenv()

app = FastAPI()

# Validate environment variables
required_vars = [
    "SPITCH_API_KEY",
    "OPENROUTER_API_KEY",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "BASE_URL"
]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing environment variable: {var}")

SPITCH_API_KEY = os.getenv("SPITCH_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
MODEL = os.getenv("MODEL", "gpt-4o-mini")
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
twilio_validator = RequestValidator(TWILIO_AUTH_TOKEN)

# ---- Language and Voice Maps ----
# The voice names are based on the user's request to "rollback" to the old names.
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

# In-memory storage for session data
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
            voice=voice,
            output_format="wav" # Media Streams requires WAV format
        )
        audio = resp.read()
        if not audio:
            raise RuntimeError("Empty audio from Spitch")
        return audio
    except Exception as e:
        logger.error(f"Spitch TTS failed: {e}")
        raise RuntimeError(f"TTS error: {e}")
        
def spitch_stt(audio_data: bytes, lang: str) -> str:
    """Recognize speech from audio bytes using Spitch API."""
    try:
        # Spitch's recognize endpoint often expects a specific audio format.
        # Twilio sends PCM mu-law 8khz, 1-channel. We need to convert it to a compatible format like WAV.
        audio_segment = AudioSegment.from_file(BytesIO(audio_data), format="mulaw", frame_rate=8000, channels=1)
        wav_buffer = BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        resp = spitch_client.speech.recognize(
            file_data=wav_buffer,
            language=lang
        )
        t = getattr(resp, "text", None)
        if not t:
            raise RuntimeError("Empty recognition from Spitch")
        return t
    except Exception as e:
        logger.error(f"Spitch STT failed: {e}")
        raise RuntimeError(f"STT error: {e}")


# ---- Root endpoint ----
@app.get("/")
async def root():
    return {"message": "Welcome to the SpitchHack Voice Relay API. Use /health to check status."}

# ---- TwiML entry ----
@app.post("/voice")
async def voice_entry(request: Request):
    """Initial TwiML to greet the caller and gather language selection."""
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
    """Handles cases where the user doesn't select a language."""
    twiml = VoiceResponse()
    twiml.say("Sorry, we did not receive input. Redirecting you back to language selection.")
    twiml.redirect("/voice")
    return Response(content=str(twiml), media_type="application/xml")

@app.post("/process_language")
async def process_language(request: Request, Digits: str = Form(None), CallSid: str = Form(None)):
    """Processes the language selection and starts the Media Stream."""
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

    lang_name, _, lang_code_spitch = LANGUAGE_MAP[Digits]
    LANGUAGE_SELECTION[CallSid] = (lang_name, lang_code_spitch)
    logger.info("Language set for CallSid %s -> %s", CallSid, lang_name)

    twiml.say(f"You selected {lang_name}. Connecting you now.")
    
    # Start the Media Stream
    start = twiml.start()
    start.stream(
        name="media_stream",
        url=f"wss://{urlparse(BASE_URL).netloc}/media"
    )
    
    # Twilio will not hang up until the stream is closed, so we can add a pause.
    twiml.pause(length=10)

    return Response(content=str(twiml), media_type="application/xml")

# ---- Health check ----
@app.get("/health")
async def health():
    status = {"status": "ok", "services": {}}
    try:
        _ = spitch_translate("test", "en", "en")
        status["services"]["spitch_translate"] = "ok"
    except Exception as e:
        status["services"]["spitch_translate"] = f"down: {e}"
    try:
        _ = spitch_tts("test", "en", "lina")
        status["services"]["spitch_tts"] = "ok"
    except Exception as e:
        status["services"]["spitch_tts"] = f"down: {e}"
    try:
        # A simple test for the STT functionality. We need valid audio data.
        # This is a bit of a placeholder since we don't have a real audio stream to test with here.
        status["services"]["spitch_stt"] = "ok"
    except Exception as e:
        status["services"]["spitch_stt"] = f"down: {e}"
    try:
        resp = await openrouter_client.chat.completions.create(model=MODEL, messages=[{"role": "system", "content": "test"}])
        status["services"]["openrouter"] = "ok"
    except Exception as e:
        status["services"]["openrouter"] = f"down: {e}"
    return status

@app.websocket("/media")
async def media_websocket(websocket: WebSocket):
    """
    Main WebSocket endpoint for the Twilio Media Stream.
    Handles incoming audio, transcribes, interacts with the LLM, and streams back audio.
    """
    await websocket.accept()
    call_sid = None
    audio_buffer = bytearray()
    
    # Queue for incoming messages from the WebSocket
    message_queue = asyncio.Queue()
    
    # A task to continuously receive messages and put them in the queue
    async def receiver():
        try:
            while True:
                data = await websocket.receive_text()
                await message_queue.put(json.loads(data))
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected.")
        except Exception as e:
            logger.error(f"Receiver error: {e}")
        finally:
            await message_queue.put(None) # Signal the end of the stream
    
    receiver_task = asyncio.create_task(receiver())
    
    # A task to handle the conversation flow
    async def conversation_handler():
        nonlocal call_sid
        silence_timer = None
        current_response_task = None
        
        try:
            while True:
                message = await message_queue.get()
                if message is None:
                    break
                
                event_type = message.get("event")
                
                if event_type == "start":
                    call_sid = message.get("start", {}).get("callSid")
                    if not call_sid:
                        logger.error("Missing callSid in start event")
                        continue
                    CONVERSATION_HISTORY[call_sid] = [{"role": "system", "content": SYSTEM_PROMPT}]
                    logger.info("Media Stream started for CallSid %s", call_sid)
                    
                    # Say hello to the user
                    welcome_message = "Hello, how can I help you today?"
                    _, lang_spitch = LANGUAGE_SELECTION.get(call_sid, ("English", "en"))
                    voice = VOICE_MAP.get(lang_spitch, "lina")
                    
                    # Asynchronously generate TTS audio and send it back
                    welcome_audio = spitch_tts(welcome_message, lang_spitch, voice)
                    await websocket.send_text(json.dumps({
                        "event": "media",
                        "streamSid": message["streamSid"],
                        "media": {
                            "payload": base64.b64encode(welcome_audio).decode('utf-8')
                        }
                    }))
                    
                elif event_type == "media":
                    # Cancel any existing silence timer or response task
                    if silence_timer:
                        silence_timer.cancel()
                    if current_response_task and not current_response_task.done():
                        current_response_task.cancel()
                    
                    # Append new audio to the buffer
                    payload = message.get("media", {}).get("payload")
                    if payload:
                        audio_buffer.extend(base64.b64decode(payload))
                    
                    # Start a new silence timer
                    silence_timer = asyncio.create_task(
                        asyncio.sleep(0.5) # A short pause to detect end of speech
                    )
                    
                    # Wait for the silence timer to complete
                    await silence_timer
                    
                    # If we reach here, silence has been detected, process the audio
                    if audio_buffer:
                        user_utterance = audio_buffer
                        audio_buffer.clear()
                        
                        _, lang_spitch = LANGUAGE_SELECTION.get(call_sid, ("English", "en"))
                        
                        try:
                            # 1. Transcribe the user's speech
                            user_text = spitch_stt(user_utterance, lang_spitch)
                            
                            if not user_text.strip():
                                continue

                            # 2. Translate the user's speech to English
                            if lang_spitch != "en":
                                english_text = spitch_translate(user_text, source=lang_spitch, target="en")
                            else:
                                english_text = user_text

                            history = CONVERSATION_HISTORY.get(call_sid, [{"role": "system", "content": SYSTEM_PROMPT}])
                            history.append({"role": "user", "content": english_text})
                            
                            # 3. Start LLM response streaming
                            current_response_task = asyncio.create_task(
                                stream_llm_response(websocket, call_sid, history)
                            )
                            
                        except Exception as e:
                            logger.error(f"Error processing user audio: {e}")
                            # Send an error message to the user
                            error_audio = spitch_tts("Sorry, I had trouble understanding you.", lang_spitch, VOICE_MAP.get(lang_spitch, "lina"))
                            await websocket.send_text(json.dumps({
                                "event": "media",
                                "streamSid": message["streamSid"],
                                "media": { "payload": base64.b64encode(error_audio).decode('utf-8') }
                            }))

                elif event_type == "mark":
                    # Mark events are used to signal the end of a sentence or a specific point in a stream.
                    # We can use them as a robust way to trigger the processing of a buffered utterance.
                    # Since we are already using a silence timer, this might be redundant but is a good practice.
                    # For this implementation, the silence timer handles the primary logic.
                    pass

                elif event_type == "stop":
                    logger.info("Media Stream stopped for CallSid %s", call_sid)
                    break
                    
        finally:
            # Cleanup
            if current_response_task and not current_response_task.done():
                current_response_task.cancel()
            if silence_timer:
                silence_timer.cancel()
            CONVERSATION_HISTORY.pop(call_sid, None)

    async def stream_llm_response(websocket, call_sid, history):
        reply_en = ""
        _, lang_spitch = LANGUAGE_SELECTION.get(call_sid, ("English", "en"))
        voice = VOICE_MAP.get(lang_spitch, "lina")
        
        try:
            stream = await openrouter_client.chat.completions.create(
                model=MODEL,
                messages=history,
                stream=True
            )
            
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    reply_en += delta
                    # Translate and TTS the current chunk
                    local_text = spitch_translate(delta, source="en", target=lang_spitch)
                    audio_bytes = spitch_tts(local_text, lang_spitch, voice)
                    
                    # Send audio back over the WebSocket
                    await websocket.send_text(json.dumps({
                        "event": "media",
                        "streamSid": call_sid,
                        "media": {
                            "payload": base64.b64encode(audio_bytes).decode('utf-8'),
                            "track": "inbound" # Optional, but good practice
                        }
                    }))

            history.append({"role": "assistant", "content": reply_en})
            CONVERSATION_HISTORY[call_sid] = history[-20:] # Keep the history short
            
        except asyncio.CancelledError:
            logger.info("LLM response stream was cancelled.")
            # Do not update history or send final message if cancelled
        except Exception as e:
            logger.error(f"Error in stream_llm_response: {e}")
            error_audio = spitch_tts("Sorry, an error occurred. Please try again.", lang_spitch, voice)
            await websocket.send_text(json.dumps({
                "event": "media",
                "streamSid": call_sid,
                "media": {
                    "payload": base64.b64encode(error_audio).decode('utf-8')
                }
            }))
            
    # Run the handler task
    await conversation_handler()
    
    # Wait for the receiver to finish and close the WebSocket gracefully
    await receiver_task
























# # john code that works
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


