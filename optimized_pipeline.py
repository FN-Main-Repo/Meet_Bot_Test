"""
Optimized Voice Pipeline - Groq Whisper STT + Groq LLM + Edge TTS
with Emotion-Safe Architecture

ALL FREE:
  STT: Groq Whisper large-v3-turbo (free tier)
  LLM: Groq Llama 3.3 70B (free tier)
  TTS: Edge TTS (free, no API key, neural Hindi/English voices)

Architecture:
  Audio ──┬──> Groq Whisper STT ──> transcript ──┐
          │                                       │
          └──> EmotionDetector ──> emotion ────────┤
                                                   ▼
                                            Groq LLM (streaming)
                                            (plain text, NO tags)
                                                   │
                                                   ▼
                                            Edge TTS (hi-IN-MadhurNeural)
                                            MP3 -> PCM via ffmpeg
                                                   │
                                                   ▼
                                            Audio out (16kHz PCM)

Key design decision: The LLM NEVER generates emotion tags.
Emotion is detected from user audio/text and applied as a
CONTROL SIGNAL by our deterministic code.
"""

import asyncio
import logging
import os
import io
import wave
import time
import re
from typing import AsyncGenerator, Optional
from collections import deque
from dataclasses import dataclass

from dotenv import load_dotenv
from groq import AsyncGroq

from svara_tts_client import SvaraTTSClient
from emotion_detector import (
    EmotionDetector,
    strip_all_tags,
    DEFAULT_EMOTION,
    TONE_GUIDES,
)

load_dotenv()
load_dotenv(".env.local", override=True)

logger = logging.getLogger("optimized-pipeline")

# Helper to resample 24kHz (Svara) to 16kHz (Pipeline standard)
def resample_24to16(audio_bytes: bytes) -> bytes:
    import numpy as np
    from scipy.signal import resample
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    num_samples = int(len(audio_np) * 16000 / 24000)
    resampled_np = resample(audio_np, num_samples).astype(np.int16)
    return resampled_np.tobytes()

@dataclass
class LatencyMetrics:
    """Track latency at each stage. All times are absolute (time.time())."""
    stt_start: float = 0.0
    stt_end: float = 0.0
    emotion_start: float = 0.0
    emotion_end: float = 0.0
    llm_start: float = 0.0
    llm_first_token: float = 0.0
    llm_end: float = 0.0
    tts_start: float = 0.0
    tts_first_audio: float = 0.0
    tts_end: float = 0.0
    total_start: float = 0.0
    total_first_audio: float = 0.0
    total_end: float = 0.0

    def _safe_diff(self, end: float, start: float) -> float:
        """Safe time diff — returns 0 if either timestamp is unset."""
        if end > 0 and start > 0 and end >= start:
            return (end - start) * 1000
        return 0.0

    def log_summary(self):
        """Log latency summary."""
        stt_time = self._safe_diff(self.stt_end, self.stt_start)
        emo_time = self._safe_diff(self.emotion_end, self.emotion_start)
        llm_ttft = self._safe_diff(self.llm_first_token, self.llm_start)
        llm_total = self._safe_diff(self.llm_end, self.llm_start)
        tts_ttfa = self._safe_diff(self.tts_first_audio, self.tts_start)
        tts_total = self._safe_diff(self.tts_end, self.tts_start)
        total_ttfa = self._safe_diff(self.total_first_audio, self.total_start)
        total_time = self._safe_diff(self.total_end, self.total_start)

        logger.info(
            f"\n"
            f"+------------------------------------------------------------+\n"
            f"|                    LATENCY BREAKDOWN                       |\n"
            f"+------------------------------------------------------------+\n"
            f"|  STT Processing:          {stt_time:>7.0f} ms                    |\n"
            f"|  Emotion Detection:       {emo_time:>7.0f} ms                    |\n"
            f"|  LLM Time-to-First-Token: {llm_ttft:>7.0f} ms                    |\n"
            f"|  LLM Total:               {llm_total:>7.0f} ms                    |\n"
            f"|  TTS Time-to-First-Audio: {tts_ttfa:>7.0f} ms                    |\n"
            f"|  TTS Total:               {tts_total:>7.0f} ms                    |\n"
            f"+------------------------------------------------------------+\n"
            f"|  TIME TO FIRST AUDIO:     {total_ttfa:>7.0f} ms                    |\n"
            f"|  TOTAL PROCESSING:        {total_time:>7.0f} ms                    |\n"
            f"+------------------------------------------------------------+"
        )


class OptimizedPipeline:
    """
    Emotion-aware voice pipeline: Groq Whisper STT + Groq LLM + Edge TTS.

    Completely FREE stack. No paid APIs required.
    """

    SYSTEM_PROMPT_TEMPLATE = """You are Mochan, a voice AI assistant in a Google Meet call.

YOUR NAME: Mochan (also Mochandas or Mo-chan).

{emotion_instruction}

LANGUAGE: Match the user's language. Hindi → Hindi. English → English. Hinglish → Hinglish.

CRITICAL STYLE RULES:
- KEEP IT SHORT! Maximum 1-2 sentences. You are in a VOICE call — nobody wants to hear long paragraphs.
- Be direct and to the point. Answer the question, don't ramble.
- Sound natural and conversational, like a friend talking.
- Use colloquial Hindi/Hinglish when the user does (bhai, haan, thoda, etc.)
- NEVER repeat back what the user said. Just answer directly.
- NEVER say "main samajh gaya" or "main aapki baat sun raha hoon" — just answer!

FORMATTING:
- NO XML tags, angle brackets, or special formatting
- NO emotion markers like <happy> or <sad>
- NO URLs, links, or markdown
- NO asterisks or bold text
- Plain spoken text only

You are speaking out loud. Be brief."""

    # TTS-friendly chunking
    MIN_CHUNK_LENGTH = 30
    MAX_CHUNK_LENGTH = 150

    def __init__(self, use_audio_emotion: bool = False):
        """
        Initialize the pipeline. All services are free tier.

        Args:
            use_audio_emotion: Use wav2vec2 audio emotion (default: text-only)
        """
        # API keys
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment")

        # Groq client (STT + LLM)
        self.groq_client = AsyncGroq(api_key=self.groq_api_key)

        # Svara TTS client (Advanced Indic Speech-LLM)
        self.tts = SvaraTTSClient()
        self.voice_id = os.getenv("TTS_VOICE_ID", "hindi_female")

        # Sarvam REST as fallback (optional, only if key present)
        self._sarvam_rest = None
        sarvam_key = os.getenv("SARVAM_API_KEY")
        if sarvam_key:
            try:
                from sarvam_streaming import SarvamRESTClient
                self._sarvam_rest = SarvamRESTClient(sarvam_key)
            except Exception:
                pass

        # Emotion detector
        self.emotion_detector = EmotionDetector(use_audio=use_audio_emotion)

        # Conversation history (last 6 messages)
        self.messages: deque = deque(maxlen=6)

        # Current turn emotion
        self._current_emotion = DEFAULT_EMOTION

        # Metrics
        self.current_metrics = LatencyMetrics()

        # STT config
        self.stt_model = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
        self.stt_language = os.getenv("STT_LANGUAGE", "hi")

        logger.info("Pipeline initialized (ALL FREE):")
        logger.info(f"  STT: Groq {self.stt_model}")
        logger.info(f"  LLM: Groq Llama 3.3 70B")
        logger.info(f"  TTS: Svara-v1 ({self.voice_id})")
        logger.info(f"  Emotion: {'audio+text' if use_audio_emotion else 'text-only'}")

    # ═══════════════════════════════════════════════════════════
    # Text Cleaning
    # ═══════════════════════════════════════════════════════════

    def _clean_for_tts(self, text: str) -> str:
        """Clean LLM output for TTS."""
        text = strip_all_tags(text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _find_split_point(self, text: str) -> Optional[int]:
        """Find optimal split point for TTS chunking."""
        length = len(text)
        if length < self.MIN_CHUNK_LENGTH:
            return None

        sentence_endings = '.?!;'
        for i in range(min(length, self.MAX_CHUNK_LENGTH) - 1, self.MIN_CHUNK_LENGTH - 1, -1):
            if text[i] in sentence_endings:
                return i + 1

        if length >= self.MAX_CHUNK_LENGTH:
            for marker in [',', ':', ';', ' - ', ' — ']:
                idx = text.rfind(marker, self.MIN_CHUNK_LENGTH, self.MAX_CHUNK_LENGTH)
                if idx > 0:
                    return idx + len(marker)
            idx = text.rfind(' ', self.MIN_CHUNK_LENGTH, self.MAX_CHUNK_LENGTH)
            if idx > 0:
                return idx + 1
            return self.MAX_CHUNK_LENGTH

        return None

    # ═══════════════════════════════════════════════════════════
    # STT: Groq Whisper (FREE)
    # ═══════════════════════════════════════════════════════════

    async def transcribe_audio(self, audio_pcm: bytes) -> Optional[str]:
        """Transcribe audio using Groq Whisper (free)."""
        self.current_metrics.stt_start = time.time()

        try:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(audio_pcm)
            wav_buffer.seek(0)
            wav_buffer.name = "audio.wav"

            # Build STT params — omit language for auto-detect (better for Hinglish)
            stt_params = {
                "file": wav_buffer,
                "model": self.stt_model,
                "response_format": "text",
            }
            if self.stt_language:  # Only set language if explicitly configured
                stt_params["language"] = self.stt_language

            transcription = await self.groq_client.audio.transcriptions.create(**stt_params)

            self.current_metrics.stt_end = time.time()
            transcript = transcription.strip() if transcription else None

            if transcript:
                stt_ms = (self.current_metrics.stt_end - self.current_metrics.stt_start) * 1000
                logger.info(f"STT [{stt_ms:.0f}ms]: '{transcript}'")
                return transcript
            return None

        except Exception as e:
            logger.error(f"STT error: {e}")
            self.current_metrics.stt_end = time.time()
            return None

    # ═══════════════════════════════════════════════════════════
    # Emotion Detection
    # ═══════════════════════════════════════════════════════════

    async def detect_emotion(
        self, transcript: str, audio_pcm: Optional[bytes] = None
    ) -> str:
        """Detect emotion from user input. Resets smoothing per turn."""
        self.current_metrics.emotion_start = time.time()

        # Reset emotion state each turn so previous turns don't bleed in
        self.emotion_detector.reset()

        response_emotion = await self.emotion_detector.detect(
            transcript=transcript,
            audio_pcm=audio_pcm,
        )

        self.current_metrics.emotion_end = time.time()
        self._current_emotion = response_emotion
        return response_emotion

    # ═══════════════════════════════════════════════════════════
    # LLM: Groq Llama 3.3 70B (streaming, FREE)
    # ═══════════════════════════════════════════════════════════

    def _get_system_prompt(self, emotion: str) -> str:
        """Build system prompt with emotion-adapted tone."""
        tone_guide = TONE_GUIDES.get(emotion, TONE_GUIDES["neutral"])
        return self.SYSTEM_PROMPT_TEMPLATE.format(emotion_instruction=tone_guide)

    async def generate_response_streaming(
        self, user_text: str, emotion: str = DEFAULT_EMOTION
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response. Produces PLAIN TEXT only."""
        self.current_metrics.llm_start = time.time()

        try:
            self.messages.append({"role": "user", "content": user_text})

            system_prompt = self._get_system_prompt(emotion)
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(list(self.messages))

            stream = await self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=100,      # Keep responses SHORT for voice
                temperature=0.7,
                stream=True,
            )

            first_token = True
            full_response = ""

            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    if first_token:
                        self.current_metrics.llm_first_token = time.time()
                        first_token = False
                    full_response += delta
                    yield delta

            self.current_metrics.llm_end = time.time()

            clean_response = self._clean_for_tts(full_response)
            self.messages.append({"role": "assistant", "content": clean_response})
            logger.info(f"LLM: '{clean_response[:100]}...'")

        except Exception as e:
            logger.error(f"LLM error: {e}")
            self.current_metrics.llm_end = time.time()
            yield "Maaf kijiye, kuch gadbad ho gayi."

    # ═══════════════════════════════════════════════════════════
    # TTS: Edge TTS (FREE, no API key)
    # ═══════════════════════════════════════════════════════════

    async def synthesize_speech(self, text: str, emotion: str = DEFAULT_EMOTION) -> Optional[bytes]:
        """
        Synthesize speech using Svara-v1.

        Args:
            text: Clean text to synthesize
            emotion: Response emotion (e.g. happy, sad, anger)

        Returns:
            Raw PCM audio bytes (Resampled to 16kHz, 16-bit, mono)
        """
        try:
            tts_text = self._clean_for_tts(text)
            if not tts_text:
                return None

            # Svara returns 24kHz audio
            audio_24k = await self.tts.synthesize(tts_text, voice_id=self.voice_id, emotion=emotion)

            if audio_24k:
                # Resample to 16kHz to match pipeline/agent standard
                audio_16k = resample_24to16(audio_24k)
                logger.debug(f"TTS: {len(audio_16k)} bytes for '{tts_text[:30]}...' (emotion: {emotion})")
                return audio_16k

            return None

        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

    # ═══════════════════════════════════════════════════════════
    # Full Streaming Pipeline
    # ═══════════════════════════════════════════════════════════

    async def process_streaming(self, user_text: str) -> AsyncGenerator[bytes, None]:
        """
        ULTRA-LOW LATENCY: Streams directly from Svara chunks.
        """
        self.current_metrics = LatencyMetrics()
        self.current_metrics.total_start = time.time()

        # 1. Detect emotion
        emotion = await self.detect_emotion(user_text)
        
        # 2. Get LLM response text (Fastest way is to get full text first for Llama 3.3 on Groq)
        # Groq is so fast (~200ms for full text) that waiting for full text is better than
        # token-by-token processing which adds overhead.
        full_llm_text = ""
        async for token in self.generate_response_streaming(user_text, emotion):
            full_llm_text += token

        # 3. Stream from Svara TTS Client
        self.current_metrics.tts_start = time.time()
        first_chunk = True
        
        async for audio_chunk_24k in self.tts.synthesize_streaming(full_llm_text, voice_id=self.voice_id, emotion=emotion):
            if first_chunk:
                self.current_metrics.tts_first_audio = time.time()
                self.current_metrics.total_first_audio = time.time()
                first_chunk = False
            
            # Resample and yield immediately
            yield resample_24to16(audio_chunk_24k)

        self.current_metrics.total_end = time.time()
        self.current_metrics.log_summary()

    async def process_full(self, audio_pcm: bytes) -> AsyncGenerator[bytes, None]:
        """
        Full pipeline: Audio -> STT -> Emotion -> LLM -> TTS -> Audio

        Args:
            audio_pcm: Raw PCM audio bytes (16-bit, mono, 16kHz)

        Yields:
            Raw PCM audio chunks
        """
        self.current_metrics = LatencyMetrics()
        self.current_metrics.total_start = time.time()

        transcript = await self.transcribe_audio(audio_pcm)

        if not transcript or len(transcript.strip()) < 2:
            logger.warning("No valid transcript")
            return

        async for audio_chunk in self.process_streaming(transcript):
            yield audio_chunk


# ═══════════════════════════════════════════════════════════════
# Standalone test
# ═══════════════════════════════════════════════════════════════

async def test_pipeline():
    """Test the pipeline with different emotional inputs."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "=" * 60)
    print("Testing Pipeline (Groq + Edge TTS + Emotion)")
    print("=" * 60 + "\n")

    pipeline = OptimizedPipeline()

    test_cases = [
        ("Mochan, aaj ka weather kaisa hai?", "neutral query"),
        ("Yeh kaam nahi ho raha hai, help karo!", "frustrated user"),
        ("Wow, amazing! Thank you so much Mochan!", "happy user"),
        ("Main bahut pareshan hoon, kya karoon?", "sad/worried user"),
    ]

    for text, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Input: '{text}'")

        audio_chunks = []
        start = time.time()

        async for chunk in pipeline.process_streaming(text):
            audio_chunks.append(chunk)
            if len(audio_chunks) == 1:
                ttfa = (time.time() - start) * 1000
                print(f"  First audio: {ttfa:.0f}ms")
            print(f"  Audio chunk: {len(chunk)} bytes")

        if audio_chunks:
            total_audio = b''.join(audio_chunks)
            duration_sec = len(total_audio) / (16000 * 2)
            total_ms = (time.time() - start) * 1000
            print(f"  Total: {len(total_audio)} bytes ({duration_sec:.1f}s audio) in {total_ms:.0f}ms")

            import wave as wave_mod
            filename = f"test_emotion_{description.replace(' ', '_').replace('/', '_')}.wav"
            with wave_mod.open(filename, 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(total_audio)
            print(f"  Saved to {filename}")
        else:
            print("  No audio generated")

        await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(test_pipeline())
