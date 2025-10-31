

---

# 📞 Proxy Voice — Local AI Voice Calls with African Language Support

**Proxy Voice** is an AI-powered call assistant that lets anyone place or receive **real phone calls** and have a **real-time voice conversation** in **African and global languages** — including **Yoruba, Igbo, Hausa, Amharic, and English**.

Built using **Twilio**, **Spitch TTS/Translate**, and **OpenRouter LLMs**, Proxy Voice makes inclusive voice AI accessible to local communities through **ordinary phone calls** (no app required).

---

## 🌍 Why Proxy Voice?

Most voice AI tools assume everyone speaks English or has fast internet.  
Proxy Voice changes that by allowing real phone conversations in **local African languages**, powered by low-latency **Spitch voice synthesis** and **Twilio call integration**.

With Proxy Voice, you can:
- Dial a number and **talk to an AI in Yoruba, Igbo, Hausa, or English**
- Get **real-time responses** — no lag, no awkward pauses  
- Use it for **customer service, education, or accessibility**
- Experience **AI that actually sounds local**

---

## 🧠 How It Works

sequenceDiagram
    participant User
    participant Twilio
    participant FastAPI Server
    participant OpenRouter
    participant Spitch

    User->>Twilio: Makes a phone call
    Twilio->>FastAPI Server: Webhook event (/voice)
    FastAPI Server->>User: Language selection prompt (press 1–4)
    User->>FastAPI Server: Chooses language
    FastAPI Server->>OpenRouter: Sends user speech as text for LLM response
    FastAPI Server->>Spitch: Translates and synthesizes AI reply (TTS)
    Spitch-->>FastAPI Server: Returns WAV audio
    FastAPI Server-->>Twilio: Streams audio back to caller
    Twilio-->>User: Plays AI’s spoken response

---

## ⚙️ Features

✅ **Multilingual Support** — Yoruba, Igbo, Hausa, Amharic, English
✅ **Local Voices** — Uses Spitch native-sounding voices (`sade`, `ngozi`, `amina`, `jude`, etc.)
✅ **Real-Time Response** — Streams AI output as it’s generated
✅ **Voice Translation Layer** — Translates between local and English automatically
✅ **Works on Any Phone** — No app, no internet needed for caller
✅ **Interruptible Speech** — Caller can interrupt and speak anytime
✅ **LLM-Powered Understanding** — Uses OpenRouter for contextual reasoning

---

## 🧩 Tech Stack

| Component            | Role                                                          |
| -------------------- | ------------------------------------------------------------- |
| **FastAPI**          | Core backend server handling webhooks and WebSocket streaming |
| **Twilio Voice API** | Handles phone call routing and conversation relay             |
| **Spitch API**       | Provides multilingual text-to-speech (TTS) and translation    |
| **OpenRouter API**   | Powers the LLM that understands and generates responses       |
| **Python (async)**   | For concurrency between translation, AI, and streaming        |
| **WebSockets**       | For bidirectional Twilio <-> Server audio relay               |

---

## 🏗️ Project Architecture

```
src/
├── app.py                # Main FastAPI app (entry point)
├── requirements.txt      # Dependencies
├── .env                  # Environment variables
├── README.md             # This file
└── /utils
    ├── spitch_client.py  # Spitch translation & TTS wrapper
    └── twilio_helpers.py # Twilio signature + call helpers
```

---

## 🔑 Environment Variables

You must define these in your `.env` file:

| Variable                   | Description                                                     |
| -------------------------- | --------------------------------------------------------------- |
| `SPITCH_API_KEY`           | API key from [Spitch](https://spitch.ai)                        |
| `OPENROUTER_API_KEY`       | API key from [OpenRouter](https://openrouter.ai)                |
| `TWILIO_ACCOUNT_SID`       | From Twilio console                                             |
| `TWILIO_AUTH_TOKEN`        | From Twilio console                                             |
| `BASE_URL`                 | Your public FastAPI endpoint (e.g. from ngrok or render)        |
| `CONVERSATION_SERVICE_SID` | Twilio Conversation SID                                         |
| `MODEL`                    | OpenRouter model name (e.g. `gpt-4o-mini`, `claude-3.5-sonnet`) |

---

## ⚡ Language & Voice Mapping

| Language | Code | Voice   |
| -------- | ---- | ------- |
| Yoruba   | `yo` | `sade`  |
| Igbo     | `ig` | `ngozi` |
| Hausa    | `ha` | `amina` |
| English  | `en` | `jude`  |

You can update these in the `SPITCH_VOICE_MAP` dictionary.

---

## 🧬 Example Flow

1. User calls your Twilio number
2. System prompts:

   ```
   For Yoruba press 1. For Igbo press 2. For Hausa press 3. For English press 4.
   ```
3. User selects a language
4. User speaks — e.g., “Báwo ni o?” (Yoruba for “How are you?”)
5. Speech → Text (via Twilio)
6. Text → English (via Spitch Translate)
7. English → AI Response (via OpenRouter)
8. AI → Local Language (via Spitch Translate)
9. Local Text → Audio (via Spitch TTS)
10. Audio streamed back to the caller

---

## 🧪 Running Locally

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set your environment variables**

   ```bash
   cp .env.example .env
   ```

3. **Run the FastAPI server**

   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

4. **Expose your server**

   ```bash
   ngrok http 8000
   ```

5. **Set Twilio Webhook**
   Point your Twilio voice webhook to:

   ```
   https://<your-ngrok-id>.ngrok.io/voice
   ```

---

## 🎥 Demo

🎬 Watch the full demo here → [your video link]

In the demo:

* A real phone call is made to the Twilio number
* The caller speaks Yoruba, and the AI responds in Yoruba
* The LLM handles the context and tone of the conversation

---

## 🚀 Possible Use Cases

* Multilingual **customer support** or IVR systems
* **Education bots** for local language learning
* **Accessibility tools** for non-English speakers
* **Community hotlines** powered by AI

---

## 🧱 Roadmap

* [ ] Add voice activity detection & silence tuning
* [ ] Add more African languages (Swahili, Zulu, Shona)
* [ ] Add persistence (e.g., store conversations in DB)
* [ ] Add local caching for Spitch TTS responses

---

## 🤝 Credits

* [Twilio Voice](https://www.twilio.com/voice)
* [Spitch AI](https://spitch.ai)
* [OpenRouter](https://openrouter.ai)
* Built with ❤️ by [@yourhandle](https://x.com/yourhandle)

---

## 📜 License

© 2025 [Toheeb Ogunade]

```


