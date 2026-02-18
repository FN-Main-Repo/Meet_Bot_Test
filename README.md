# 🤖 Mochan - Emotion-Aware Hindi Voice Assistant for Google Meet

Mochan is a low-latency, emotionally intelligent voice assistant optimized for Hindi and Hinglish conversations in Google Meet. Built on an advanced Speech-to-Speech architecture, Mochan doesn't just respond with text; it perceives the user's mood and adapts its voice tone in real-time.

## ⚡ Performance Targets (on NVIDIA RTX A5000)

| Metric | Target | Status |
|--------|--------|----------|
| STT Latency (Groq Whisper) | ~200ms | 🎯 |
| LLM TTFT (Groq Llama 3.3) | ~150ms | 🎯 |
| TTS TTFA (Svara-v1 Streaming) | ~400ms | 🎯 |
| **Total Time-to-First-Audio** | **<1000ms** | 🚀 |

## 📦 Architecture

```
       Google Meet
            ↕
      MeetStream.ai Bot
            ↕
    WebSocket (wss://)
            ↕
┌─────────────────────────────────────┐
│          MOCHAN AGENT               │
├─────────────────────────────────────┤
│  SimpleVAD (Energy-based)           │
│            ↓                        │
│  Groq Whisper STT (Hindi/English)   │
│            ↓                        │
│  Emotion Detector (Keyword Based)   │
│            ↓                        │
│  Groq LLM (Llama 3.3 70B)           │
│            ↓                        │
│  Svara-v1 (Speech-LLM / Orpheus)    │
└─────────────────────────────────────┘
            ↕
 Audio back to Meeting (24kHz → 16kHz)
```

## 🎭 Emotional Intelligence
Mochan maps the user's emotional state to an appropriate response tone using its **Emotional Palette**:

| User Tone | Mochan Response | Svara Tag |
|-----------|-----------------|-----------|
| Frustrated| Calm/Patient    | `<neutral>`|
| Sad       | Empathetic      | `<sad>`    |
| Happy     | Cheerful        | `<happy>`  |
| Angry     | De-escalating   | `<neutral>`|
| Excited   | Upbeat          | `<happy>`  |

## 🚀 Quick Start (RunPod / Ubuntu)

### 1. Setup Environment
```bash
git clone https://github.com/FN-Main-Repo/Meet_Bot_Test.git
cd Meet_Bot_Test
chmod +x setup_runpod.sh
./setup_runpod.sh
```

### 2. Configure API Keys
Edit `.env` and add your keys:
```env
GROQ_API_KEY=gsk_...
MEETSTREAM_API_KEY=ms_...
HUGGINGFACE_TOKEN=hf_...
MEETING_URL=https://meet.google.com/xxx-yyyy-zzz
```

### 3. Start Cloudflare Tunnel
In a separate terminal:
```bash
cloudflared tunnel --url http://localhost:8765
```
Copy the `https://...` link and update `WEBSOCKET_PUBLIC_URL` in `.env` as `wss://...`.

### 4. Run Mochan
```bash
python3 run_mvp.py
```

## 🎤 How to Use
1. The bot joins your meeting as "Mochan".
2. Say **"Mochan"** to activate (or just speak if in active mode).
3. Speak in Hindi, English, or Hinglish.
4. Mochan will respond with an appropriate emotional tone.

## 📁 Key Files
- `svara_tts_client.py`: The core Speech-LLM engine (Svara-v1).
- `optimized_pipeline.py`: Orchestrates the streaming STT → LLM → TTS flow.
- `meetstream_bridge.py`: Manages the WebSocket connection to Google Meet.
- `emotion_detector.py`: Heuristics for real-time mood detection.

## 🔧 Deployment Optimizations
- **Bfloat16 Precision:** Optimized for NVIDIA Ampere (A5000/A6000) GPUs.
- **Streaming Tokens:** Generates and decodes SNAC tokens in chunks for sub-1s latency.
- **MPS Support:** Compatible with Mac Metal (MPS) for local testing.

## 📝 License
MIT License - Developed for FN-Main-Repo.
