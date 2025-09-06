from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse, Response
import io
import os
import base64
from spitch import Spitch
from openai import OpenAI
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse, Gather

app = FastAPI()

load_dotenv()
spitch_client = Spitch(api_key=os.getenv("SPITCH_API_KEY"))
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "mistralai/mistral-7b-instruct:free"

VOICE_MAP = {
    "en": "jude",
    "ha": "aliyu",  # Replace with actual Spitch Hausa voice
    "yo": "femi",
    "ig": "obinna"
}
DEFAULT_VOICE = "jude"

@app.post("/start_call")
async def start_call():
    """Asks user for preferred language and allows interruption."""
    text = "What language do you want to speak in?"
    audio_response = spitch_client.speech.generate(
        text=text,
        language="en",
        voice=DEFAULT_VOICE
    )
    audio_bytes = audio_response.read()
    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

    twiml = VoiceResponse()
    gather = Gather(
        input="speech",  # Allow speech input
        action="/process_response",
        method="POST",
        speechTimeout="auto",  # Stop prompt when user speaks
        timeout=10  # Max wait for user input (seconds)
    )
    gather.play(url=f"data:audio/wav;base64,{base64_audio}")
    twiml.append(gather)
    # Fallback if no input
    twiml.say("Sorry, I didn't catch that. Please try again.", voice="Polly.Joanna")
    twiml.redirect("/start_call", method="POST")
    return Response(content=str(twiml), media_type="application/xml")

@app.post("/process_response")
async def process_response(
    audio: UploadFile = None,
    language: str = Form(None),
    RecordingUrl: str = Form(None),
    SpeechResult: str = Form(None)  # For Gather speech input
):
    """Processes user audio response."""
    if SpeechResult:  # Handle Gather speech input
        transcribed_text = SpeechResult
        audio_io = None
    elif RecordingUrl:
        import requests
        audio_response = requests.get(RecordingUrl)
        audio_bytes = audio_response.content
        audio_io = io.BytesIO(audio_bytes)
    else:
        audio_bytes = await audio.read()
        audio_io = io.BytesIO(audio_bytes)

    if language is not None:
        # Ongoing conversation: Transcribe in specified language
        if audio_io:
            transcription = spitch_client.speech.transcribe(
                content=audio_io,
                language=language
            )
            transcribed_text = transcription.text
        else:
            transcribed_text = SpeechResult

        english_text = spitch_client.translation.translate(
            text=transcribed_text,
            source_language=language,
            target_language="en"
        )

        mistral_response = openrouter_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": english_text}
            ]
        )
        english_response = mistral_response.choices[0].message.content

        translated_response = spitch_client.translation.translate(
            text=english_response,
            source_language="en",
            target_language=language
        )

        voice = VOICE_MAP.get(language, DEFAULT_VOICE)
        tts_audio = spitch_client.speech.generate(
            text=translated_response,
            language=language,
            voice=voice
        )
        audio_bytes = tts_audio.read()
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

        twiml = VoiceResponse()
        gather = Gather(
            input="speech",
            action="/process_response",
            method="POST",
            speechTimeout="auto",
            timeout=10
        )
        gather.play(url=f"data:audio/wav;base64,{base64_audio}")
        twiml.append(gather)
        twiml.redirect(f"/process_response?language={language}", method="POST")
        return Response(content=str(twiml), media_type="application/xml")

    else:
        # Initial response: Transcribe in English
        if audio_io:
            transcription = spitch_client.speech.transcribe(
                content=audio_io,
                language="en"
            )
            transcribed_text = transcription.text
        else:
            transcribed_text = SpeechResult

        detection_prompt = f"The user said: '{transcribed_text}'. What language do they want to speak in? Respond with the ISO 639-1 code (e.g., 'ha' for Hausa)."
        mistral_response = openrouter_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a language detector."},
                {"role": "user", "content": detection_prompt}
            ]
        )
        detected_lang = mistral_response.choices[0].message.content.strip().lower()

        english_response = f"Okay, we will speak in {detected_lang}."
        translated_response = spitch_client.translation.translate(
            text=english_response,
            source_language="en",
            target_language=detected_lang
        )

        voice = VOICE_MAP.get(detected_lang, DEFAULT_VOICE)
        tts_audio = spitch_client.speech.generate(
            text=translated_response,
            language=detected_lang,
            voice=voice
        )
        audio_bytes = tts_audio.read()
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

        twiml = VoiceResponse()
        gather = Gather(
            input="speech",
            action=f"/process_response?language={detected_lang}",
            method="POST",
            speechTimeout="auto",
            timeout=10
        )
        gather.play(url=f"data:audio/wav;base64,{base64_audio}")
        twiml.append(gather)
        twiml.redirect(f"/process_response?language={detected_lang}", method="POST")
        return Response(content=str(twiml), media_type="application/xml")