#!/usr/bin/env python3
"""
Mochan MVP - Single Entry Point

Runs the full Mochan voice assistant on Google Meet via MeetStream.

Prerequisites:
  1. pip install -r requirements.txt
  2. Set API keys in .env
  3. Start cloudflare tunnel: cloudflared tunnel --url http://localhost:8765
  4. Set WEBSOCKET_PUBLIC_URL and MEETING_URL in .env

Usage:
  python run_mvp.py                              # Use .env settings
  python run_mvp.py https://meet.google.com/xxx  # Override meeting URL
  python run_mvp.py --test                        # Run local tests first

Stack (ALL FREE):
  STT:     Groq Whisper large-v3-turbo (free tier)
  LLM:     Groq Llama 3.3 70B (free tier)
  TTS:     Edge TTS (free, no API key, neural Hindi voices)
  Emotion: Text-based keyword detection (zero latency)
  Meeting: MeetStream.ai bot
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.local", override=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mochan-mvp")

# Silence noisy loggers
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def print_banner():
    print("""
============================================================
  MOCHAN MVP - AI Voice Assistant for Google Meet

  Stack:
    STT:     Groq Whisper (FREE, Hindi/English)
    LLM:     Groq Llama 3.3 70B (FREE, streaming)
    TTS:     Svara-v1 (Speech-LLM, Indic)
    Emotion: Text-based auto-detection
    Bot:     MeetStream.ai
============================================================
""")


def check_env():
    """Validate all required environment variables."""
    required = {
        "GROQ_API_KEY": "https://console.groq.com",
        "MEETSTREAM_API_KEY": "Contact MeetStream.ai",
        "WEBSOCKET_PUBLIC_URL": "Run: cloudflared tunnel --url http://localhost:8765",
        "MEETING_URL": "Your Google Meet URL",
        "HUGGINGFACE_TOKEN": "https://huggingface.co/settings/tokens",
    }

    missing = []
    for key, help_text in required.items():
        val = os.getenv(key, "")
        if not val or "your_" in val or "xxx" in val or "your-tunnel" in val:
            missing.append((key, help_text))

    if missing:
        print("Missing environment variables:\n")
        for key, help_text in missing:
            print(f"  {key}")
            print(f"    -> {help_text}\n")
        print("Add these to your .env file and try again.")
        return False

    # Print config summary
    ws_url = os.getenv("WEBSOCKET_PUBLIC_URL", "")
    meeting_url = os.getenv("MEETING_URL", "")
    voice_id = os.getenv("TTS_VOICE_ID", "hindi_female")

    print("Configuration:")
    print(f"  Meeting:  {meeting_url}")
    print(f"  WS URL:   {ws_url}")
    print(f"  TTS:      Svara-v1 ({voice_id})")
    print(f"  STT:      Groq Whisper large-v3-turbo")
    print(f"  LLM:      Groq Llama 3.3 70B")
    print()

    return True


async def run():
    """Main run function."""
    from meetstream_bridge import MochanAgent

    api_key = os.getenv("MEETSTREAM_API_KEY")
    meeting_url = os.getenv("MEETING_URL")
    websocket_url = os.getenv("WEBSOCKET_PUBLIC_URL")
    port = int(os.getenv("WEBSOCKET_SERVER_PORT", "8765"))

    # Handle meeting URL from command line
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        meeting_url = sys.argv[1]

    # Ensure wss://
    if not websocket_url.startswith("wss://"):
        websocket_url = websocket_url.replace("https://", "wss://").replace(
            "http://", "wss://"
        )
        if not websocket_url.startswith("wss://"):
            websocket_url = f"wss://{websocket_url}"

    print(f"Starting Mochan on: {meeting_url}")
    print(f"WebSocket tunnel:   {websocket_url}")
    print(f"Local port:         {port}")
    print()

    agent = MochanAgent(meetstream_api_key=api_key, bot_name="Mochan")
    await agent.run(meeting_url, websocket_url, port=port)


async def run_tests():
    """Run local pipeline tests."""
    from test_local import (
        test_emotion_detection,
        test_svara_tts,
        test_groq_whisper_stt,
        test_groq_llm,
        test_full_pipeline,
    )

    print("Running pipeline tests...\n")

    emotion_ok = await test_emotion_detection()
    tts_ok = await test_svara_tts()
    stt_ok = await test_groq_whisper_stt()
    llm_ok = await test_groq_llm()

    pipeline_ok = False
    if tts_ok and llm_ok:
        pipeline_ok = await test_full_pipeline()

    print("\n" + "=" * 50)
    print("Results:")
    print(f"  Emotion: {'PASS' if emotion_ok else 'FAIL'}")
    print(f"  TTS:     {'PASS' if tts_ok else 'FAIL'}")
    print(f"  STT:     {'PASS' if stt_ok else 'FAIL'}")
    print(f"  LLM:     {'PASS' if llm_ok else 'FAIL'}")
    print(f"  Pipeline:{'PASS' if pipeline_ok else 'FAIL'}")
    print("=" * 50)

    return all([emotion_ok, tts_ok, stt_ok, llm_ok, pipeline_ok])


def main():
    print_banner()

    # Handle --test flag
    if "--test" in sys.argv:
        if not check_env():
            sys.exit(1)
        success = asyncio.run(run_tests())
        if success:
            print("\nAll tests passed! Run without --test to start the agent.")
        sys.exit(0 if success else 1)

    # Handle --help
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage:")
        print("  python run_mvp.py                              # Start agent")
        print("  python run_mvp.py https://meet.google.com/xxx  # With meeting URL")
        print("  python run_mvp.py --test                        # Run tests first")
        print()
        print("Setup:")
        print("  1. pip install -r requirements.txt")
        print("  2. Configure .env (API keys, meeting URL, tunnel URL)")
        print("  3. cloudflared tunnel --url http://localhost:8765")
        print("  4. python run_mvp.py")
        sys.exit(0)

    # Validate env
    if not check_env():
        sys.exit(1)

    # Run agent
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n\nMochan stopped.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
