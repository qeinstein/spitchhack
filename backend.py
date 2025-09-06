from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse, Response
import io
import os
from spitch import Spitch
from openai import OpenAI
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse  # Add Twilio TwiML for call flow

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
    """Asks user for preferred language and collects response."""
    text = "What language do you want to speak in?"
    audio_response = spitch_client.speech.generate(
        text=text,
        language="en",
        voice=DEFAULT_VOICE
    )
    audio_bytes = audio_response.read()

    # Create TwiML response to play audio and collect user input
    twiml = VoiceResponse()
    twiml.play(url="data:audio/wav;base64," + audio_bytes.encode('base64'))  # Adjust URL if hosting audio
    twiml.record(
        action="/process_response",  # Next endpoint to handle user response
        method="POST",
        max_length=10,  # Limit recording length (adjust as needed)
        transcribe=False  # We'll use Spitch for transcription
    )
    return Response(content=str(twiml), media_type="application/xml")

@app.post("/process_response")
async def process_response(
    audio: UploadFile = None,
    language: str = Form(None),
    RecordingUrl: str = Form(None)  # For Twilio-recorded audio
):
    """Processes user audio response."""
    # Handle Twilio's recorded audio if no direct UploadFile
    if RecordingUrl:
        import requests
        audio_response = requests.get(RecordingUrl)
        audio_bytes = audio_response.content
    else:
        audio_bytes = await audio.read()
    audio_io = io.BytesIO(audio_bytes)

    if language is not None:
        # Ongoing conversation: Transcribe in specified language
        transcription = spitch_client.speech.transcribe(
            content=audio_io,
            language=language
        )
        transcribed_text = transcription.text

        # Translate to English
        english_text = spitch_client.translation.translate(
            text=transcribed_text,
            source_language=language,
            target_language="en"
        )

        # Process with Mistral
        mistral_response = openrouter_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": english_text}
            ]
        )
        english_response = mistral_response.choices[0].message.content

        # Translate back to user's language
        translated_response = spitch_client.translation.translate(
            text=english_response,
            source_language="en",
            target_language=language
        )

        # Generate TTS with language-specific voice
        voice = VOICE_MAP.get(language, DEFAULT_VOICE)
        tts_audio = spitch_client.speech.generate(
            text=translated_response,
            language=language,
            voice=voice
        )
        audio_bytes = tts_audio.read()

        # Return TwiML to play response and continue recording
        twiml = VoiceResponse()
        twiml.play(url="data:audio/wav;base64," + audio_bytes.encode('base64'))
        twiml.record(action="/process_response", method="POST", max_length=10)
        return Response(content=str(twiml), media_type="application/xml")

    else:
        # Initial response: Transcribe in English
        transcription = spitch_client.speech.transcribe(
            content=audio_io,
            language="en"
        )
        transcribed_text = transcription.text

        # Detect language with Mistral
        detection_prompt = f"The user said: '{transcribed_text}'. What language do they want to speak in? Respond with the ISO 639-1 code (e.g., 'ha' for Hausa)."
        mistral_response = openrouter_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a language detector."},
                {"role": "user", "content": detection_prompt}
            ]
        )
        detected_lang = mistral_response.choices[0].message.content.strip().lower()

        # Generate confirmation response
        english_response = f"Okay, we will speak in {detected_lang}."
        translated_response = spitch_client.translation.translate(
            text=english_response,
            source_language="en",
            target_language=detected_lang
        )

        # Generate TTS with language-specific voice
        voice = VOICE_MAP.get(detected_lang, DEFAULT_VOICE)
        tts_audio = spitch_client.speech.generate(
            text=translated_response,
            language=detected_lang,
            voice=voice
        )
        audio_bytes = tts_audio.read()

        # Return TwiML to play response and continue recording
        twiml = VoiceResponse()
        twiml.play(url="data:audio/wav;base64," + audio_bytes.encode('base64'))
        twiml.record(
            action="/process_response",
            method="POST",
            max_length=10,
            transcribe=False,
            recording_status_callback="/process_response?language=" + detected_lang  # Pass language
        )
        return Response(content=str(twiml), media_type="application/xml")