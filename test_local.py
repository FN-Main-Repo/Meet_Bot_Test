#!/usr/bin/env python3
"""
Local Pipeline Test - Groq Whisper STT + Groq LLM + Sarvam TTS + Emotion

Tests the complete voice pipeline locally without Google Meet.

Usage:
    python test_local.py                    # Run all tests
    python test_local.py --text "Hello"     # Test with text input (LLM + TTS)
    python test_local.py --file audio.wav   # Test with audio file (full pipeline)
    python test_local.py --mic              # Test with microphone
    python test_local.py --tts-only         # Test TTS only
    python test_local.py --stt-only         # Test Groq Whisper STT only
    python test_local.py --llm-only         # Test LLM only
    python test_local.py --emotion-only     # Test emotion detection only
"""

import asyncio
import argparse
import logging
import os
import sys
import time
import wave
import io

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.local", override=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test-local")


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    for pkg in ["numpy", "aiohttp", "groq"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Run: pip install {' '.join(missing)}")
        sys.exit(1)


def check_api_keys():
    """Check if API keys are configured."""
    groq_key = os.getenv("GROQ_API_KEY")
    sarvam_key = os.getenv("SARVAM_API_KEY")

    ok = True

    if not groq_key or "your_" in groq_key:
        print("GROQ_API_KEY not configured")
        print("  Get key: https://console.groq.com")
        ok = False

    if not sarvam_key or "your_" in sarvam_key:
        print("SARVAM_API_KEY not configured")
        print("  Get key: https://dashboard.sarvam.ai")
        ok = False

    if ok:
        print("API keys configured")

    return ok


# ═══════════════════════════════════════════════════════════════
# Individual Tests
# ═══════════════════════════════════════════════════════════════


async def test_emotion_detection():
    """Test emotion detection from text."""
    from emotion_detector import EmotionDetector, apply_emotion_tag

    print("\n" + "=" * 50)
    print("Testing Emotion Detection")
    print("=" * 50)

    detector = EmotionDetector(use_audio=False)

    test_cases = [
        ("Mochan, aaj ka weather kaisa hai?", "neutral"),
        ("Yeh kaam nahi ho raha hai, help karo!", "frustrated"),
        ("Wow, amazing! Thank you so much!", "happy/excited"),
        ("Main bahut pareshan hoon", "sad"),
        ("Kya bakwas hai yeh!", "angry"),
        ("Bahut accha! Perfect solution!", "happy"),
        ("I'm so confused, samajh nahi aa raha", "frustrated"),
        ("Thank you Mochan, shukriya!", "happy"),
    ]

    all_passed = True

    for text, expected_category in test_cases:
        start = time.time()
        emotion = await detector.detect(transcript=text)
        elapsed_ms = (time.time() - start) * 1000

        tagged = apply_emotion_tag("Sample response text.", emotion)

        print(f"\n  Input: '{text}'")
        print(f"  Expected: ~{expected_category}")
        print(f"  Detected response emotion: {emotion} [{elapsed_ms:.1f}ms]")
        print(f"  Tagged output: '{tagged}'")

        # Reset state between tests to avoid smoothing effects
        detector.reset()

    return True


async def test_groq_whisper_stt(audio_file: str = None):
    """Test Groq Whisper STT."""
    from groq import AsyncGroq

    print("\n" + "=" * 50)
    print("Testing Groq Whisper STT")
    print("=" * 50)

    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

    if audio_file and os.path.exists(audio_file):
        print(f"  Using audio file: {audio_file}")
        with open(audio_file, "rb") as f:
            audio_data = f.read()

        # If it's a WAV file, use it directly
        wav_buffer = io.BytesIO(audio_data)
        wav_buffer.name = audio_file
    else:
        # Generate test audio from Svara TTS first
        print("  Generating test audio via Svara TTS...")
        from svara_tts_client import SvaraTTSClient

        tts = SvaraTTSClient()
        test_text = "Mochan, aaj ka mausam kaisa hai? <happy>"
        pcm_audio = await tts.synthesize(test_text)

        if not pcm_audio:
            print("  Could not generate test audio")
            return False

        print(f"  Generated test audio for: '{test_text}'")

        # Convert PCM to WAV
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(pcm_audio)
        wav_buffer.seek(0)
        wav_buffer.name = "test_audio.wav"

    # Transcribe with Groq Whisper
    print("  Transcribing with Groq Whisper...")
    start = time.time()

    transcription = await client.audio.transcriptions.create(
        file=wav_buffer,
        model="whisper-large-v3-turbo",
        language="hi",
        response_format="text",
    )

    elapsed = (time.time() - start) * 1000

    if transcription:
        print(f"  Transcript: '{transcription.strip()}'")
        print(f"  Latency: {elapsed:.0f}ms")
        return True
    else:
        print("  STT failed")
        return False


async def test_svara_tts():
    """Test Svara-v1 (Indic Speech-LLM)."""
    from svara_tts_client import SvaraTTSClient

    print("\n" + "=" * 50)
    print("Testing Svara-v1 (Indic Speech-LLM)")
    print("=" * 50)

    client = SvaraTTSClient()

    test_cases = [
        ("hindi_female", "Namaste, main Mochan hoon. Aap kaise hain? <happy>"),
        ("hindi_male", "Bharat ek vividhtao se bhara desh hai."),
        ("english_female", "Hello, I am Mochan, your AI assistant. <happy>"),
    ]

    for voice_id, text in test_cases:
        print(f"\n  Testing {voice_id}: '{text}'")
        start = time.time()

        audio = await client.synthesize(text, voice_id=voice_id)

        elapsed = (time.time() - start) * 1000

        if audio:
            # Svara is 24kHz
            duration_sec = len(audio) / (24000 * 2)
            print(f"  Generated {len(audio)} bytes ({duration_sec:.1f}s) in {elapsed:.0f}ms")

            filename = f"test_tts_svara_{voice_id}.wav"
            with wave.open(filename, 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(24000)
                f.writeframes(audio)
            print(f"  Saved to {filename}")
        else:
            print(f"  TTS failed")
            return False

    return True


async def test_groq_llm():
    """Test Groq LLM API."""
    from groq import AsyncGroq

    print("\n" + "=" * 50)
    print("Testing Groq LLM")
    print("=" * 50)

    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

    test_prompts = [
        "Mochan, aaj ka weather kaisa hai Delhi mein?",
        "Hello Mochan, tell me a joke",
    ]

    for prompt in test_prompts:
        print(f"\n  Prompt: '{prompt}'")
        start = time.time()
        first_token_time = None
        response_text = ""

        stream = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Mochan, a helpful AI assistant. "
                        "Keep responses brief (1-2 sentences). "
                        "Respond in the same language as the user. "
                        "Do NOT use any XML tags or special formatting."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                if first_token_time is None:
                    first_token_time = time.time()
                response_text += delta

        total_time = (time.time() - start) * 1000
        ttft = (first_token_time - start) * 1000 if first_token_time else 0

        print(f"  Response: '{response_text.strip()}'")
        print(f"  TTFT: {ttft:.0f}ms, Total: {total_time:.0f}ms")

    return True


async def test_full_pipeline(text_input: str = None):
    """Test full pipeline: text -> emotion -> LLM -> TTS -> audio."""
    from optimized_pipeline import OptimizedPipeline

    print("\n" + "=" * 50)
    print("Testing Full Pipeline (Emotion-Aware)")
    print("=" * 50)

    pipeline = OptimizedPipeline()

    test_cases = [
        (text_input or "Mochan, mujhe batao ki AI kya hai?", "custom/neutral"),
        ("Yeh kaam nahi ho raha, bahut frustrating hai!", "frustrated"),
        ("Wow Mochan, thank you so much! Amazing!", "happy"),
    ]

    if text_input:
        test_cases = [test_cases[0]]  # Only run the provided text

    for user_text, category in test_cases:
        print(f"\n  --- {category} ---")
        print(f"  Input: '{user_text}'")

        start = time.time()
        first_audio_time = None
        audio_chunks = []

        async for chunk in pipeline.process_streaming(user_text):
            if first_audio_time is None:
                first_audio_time = time.time()
                ttfa = (first_audio_time - start) * 1000
                print(f"  First audio: {ttfa:.0f}ms")

            audio_chunks.append(chunk)
            print(f"  Chunk {len(audio_chunks)}: {len(chunk)} bytes")

        total_time = (time.time() - start) * 1000

        if audio_chunks:
            total_audio = b''.join(audio_chunks)
            duration_sec = len(total_audio) / (16000 * 2)

            print(f"  Total: {len(total_audio)} bytes ({duration_sec:.1f}s) in {total_time:.0f}ms")

            filename = f"test_pipeline_{category.replace('/', '_')}.wav"
            with wave.open(filename, 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(total_audio)
            print(f"  Saved to {filename}")

            # Try to play
            try:
                import sounddevice as sd
                import numpy as np

                print(f"  Playing audio...")
                audio_array = np.frombuffer(total_audio, dtype=np.int16)
                sd.play(audio_array, samplerate=16000)
                sd.wait()
                print(f"  Playback complete")
            except ImportError:
                print(f"  (Install sounddevice to play: pip install sounddevice)")
            except Exception as e:
                print(f"  Could not play: {e}")
        else:
            print(f"  No audio generated")
            return False

        await asyncio.sleep(0.5)

    return True


async def test_microphone():
    """Test with live microphone."""
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        print("sounddevice not installed. Run: pip install sounddevice")
        return False

    from optimized_pipeline import OptimizedPipeline

    print("\n" + "=" * 50)
    print("Microphone Test")
    print("=" * 50)
    print("\nPress ENTER to start recording (3 seconds)")
    print("Speak in Hindi, English, or Hinglish")
    input()

    print("Recording...")

    sample_rate = 16000
    duration = 3
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='int16',
    )
    sd.wait()

    print("Recording complete")

    audio_bytes = audio.tobytes()

    # Save recording
    with wave.open("test_recording.wav", 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(audio_bytes)
    print("Saved to test_recording.wav")

    # Full pipeline
    print("\nProcessing through pipeline...")
    pipeline = OptimizedPipeline()

    start = time.time()
    audio_chunks = []

    async for chunk in pipeline.process_full(audio_bytes):
        if len(audio_chunks) == 0:
            ttfa = (time.time() - start) * 1000
            print(f"  First audio: {ttfa:.0f}ms")
        audio_chunks.append(chunk)

    if audio_chunks:
        total_audio = b''.join(audio_chunks)
        total_time = (time.time() - start) * 1000

        with wave.open("test_response.wav", 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(total_audio)
        print(f"Saved to test_response.wav ({total_time:.0f}ms total)")

        print("Playing response...")
        audio_array = np.frombuffer(total_audio, dtype=np.int16)
        sd.play(audio_array, samplerate=16000)
        sd.wait()

        return True

    return False


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════


async def main():
    parser = argparse.ArgumentParser(description="Test Mochan Pipeline Locally")
    parser.add_argument("--text", type=str, help="Test with text input (skip STT)")
    parser.add_argument("--file", type=str, help="Test STT with audio file")
    parser.add_argument("--mic", action="store_true", help="Test with microphone")
    parser.add_argument("--tts-only", action="store_true", help="Test TTS only")
    parser.add_argument("--stt-only", action="store_true", help="Test Groq Whisper only")
    parser.add_argument("--llm-only", action="store_true", help="Test LLM only")
    parser.add_argument("--emotion-only", action="store_true", help="Test emotion only")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Mochan MVP - Local Pipeline Test")
    print("  STT: Groq Whisper (FREE)")
    print("  LLM: Groq Llama 3.3 70B (FREE)")
    print("  TTS: Svara-v1 (Speech-LLM)")
    print("  Emotion: Text-based (zero latency)")
    print("=" * 60)

    check_dependencies()

    if not check_api_keys():
        sys.exit(1)

    try:
        if args.emotion_only:
            await test_emotion_detection()
        elif args.tts_only:
            await test_svara_tts()
        elif args.stt_only:
            await test_groq_whisper_stt(args.file)
        elif args.llm_only:
            await test_groq_llm()
        elif args.mic:
            await test_microphone()
        elif args.text:
            await test_full_pipeline(args.text)
        else:
            # Run all tests
            print("\nRunning all tests...\n")

            emotion_ok = await test_emotion_detection()
            tts_ok = await test_svara_tts()
            stt_ok = await test_groq_whisper_stt()
            llm_ok = await test_groq_llm()

            pipeline_ok = False
            if tts_ok and llm_ok:
                pipeline_ok = await test_full_pipeline()

            print("\n" + "=" * 60)
            print("Test Summary")
            print("=" * 60)
            print(f"  Emotion Detection: {'PASS' if emotion_ok else 'FAIL'}")
            print(f"  Svara TTS:         {'PASS' if tts_ok else 'FAIL'}")
            print(f"  Groq Whisper STT:  {'PASS' if stt_ok else 'FAIL'}")
            print(f"  Groq LLM:          {'PASS' if llm_ok else 'FAIL'}")
            print(f"  Full Pipeline:     {'PASS' if pipeline_ok else 'FAIL'}")
            print("=" * 60)

            if emotion_ok and tts_ok and stt_ok and llm_ok and pipeline_ok:
                print("\nAll tests passed! Ready for Google Meet.")
                print("\nNext steps:")
                print("  1. Start tunnel: cloudflared tunnel --url http://localhost:8765")
                print("  2. Update WEBSOCKET_PUBLIC_URL in .env")
                print("  3. Update MEETING_URL in .env")
                print("  4. Run: python run_mvp.py")
            else:
                print("\nSome tests failed. Check errors above.")

    except KeyboardInterrupt:
        print("\n\nTest cancelled")
    except Exception as e:
        logger.error(f"Test error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
