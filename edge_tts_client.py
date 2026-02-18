"""
Edge TTS Client — Free, no API key, human-quality Hindi/Hinglish/English voices.

Voices:
  hi-IN-MadhurNeural   — Hindi male (best for Hinglish too)
  hi-IN-SwaraNeural    — Hindi female
  en-IN-PrabhatNeural  — Indian English male
  en-IN-NeerjaNeural   — Indian English female

Output: MP3 from Edge → converted to 16kHz 16-bit PCM for MeetStream.
"""

import asyncio
import io
import logging
import subprocess
import time
from typing import Optional

import edge_tts

logger = logging.getLogger("edge-tts-client")

# Available Indian voices
INDIAN_VOICES = {
    "madhur": "hi-IN-MadhurNeural",       # Hindi male — recommended for Hinglish
    "swara": "hi-IN-SwaraNeural",          # Hindi female
    "prabhat": "en-IN-PrabhatNeural",      # Indian English male
    "neerja": "en-IN-NeerjaNeural",        # Indian English female
    "neerja_exp": "en-IN-NeerjaExpressiveNeural",  # Expressive female
}

DEFAULT_VOICE = "hi-IN-MadhurNeural"


def mp3_to_pcm(mp3_data: bytes, target_sample_rate: int = 16000) -> Optional[bytes]:
    """
    Convert MP3 bytes to raw PCM (16-bit, mono, 16kHz) using ffmpeg.

    Args:
        mp3_data: Raw MP3 audio bytes
        target_sample_rate: Output sample rate (16000 for MeetStream)

    Returns:
        Raw PCM bytes (16-bit, mono) or None on error
    """
    if not mp3_data:
        return None

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", "pipe:0",        # Read from stdin
                "-f", "s16le",          # Output format: signed 16-bit little-endian
                "-acodec", "pcm_s16le", # PCM codec
                "-ac", "1",             # Mono
                "-ar", str(target_sample_rate),  # Sample rate
                "pipe:1",              # Write to stdout
            ],
            input=mp3_data,
            capture_output=True,
            timeout=10,
        )

        if result.returncode != 0:
            logger.error(f"ffmpeg error: {result.stderr.decode()[:200]}")
            return None

        pcm_data = result.stdout
        if len(pcm_data) < 100:
            logger.warning(f"ffmpeg produced very short output: {len(pcm_data)} bytes")
            return None

        return pcm_data

    except FileNotFoundError:
        logger.error("ffmpeg not found. Install: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)")
        return None
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timed out")
        return None
    except Exception as e:
        logger.error(f"MP3 to PCM conversion error: {e}")
        return None


class EdgeTTSClient:
    """
    Async Edge TTS client for synthesizing speech.

    Usage:
        client = EdgeTTSClient(voice="madhur")  # Hindi male
        pcm = await client.synthesize("Namaste bhai!")
    """

    def __init__(
        self,
        voice: str = "madhur",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        sample_rate: int = 16000,
    ):
        """
        Args:
            voice: Voice name (short name like "madhur" or full like "hi-IN-MadhurNeural")
            rate: Speech rate adjustment (e.g., "+10%", "-5%")
            volume: Volume adjustment
            pitch: Pitch adjustment
            sample_rate: Output PCM sample rate
        """
        # Resolve short name to full name
        self.voice = INDIAN_VOICES.get(voice, voice)
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self.sample_rate = sample_rate

        logger.info(f"Edge TTS initialized: voice={self.voice}, rate={rate}")

    async def synthesize(self, text: str) -> Optional[bytes]:
        """
        Synthesize text to PCM audio.

        Args:
            text: Text to synthesize (Hindi, Hinglish, or English)

        Returns:
            Raw PCM bytes (16-bit, mono, 16kHz) or None on error
        """
        if not text or not text.strip():
            return None

        start = time.time()

        try:
            # Create communicator
            comm = edge_tts.Communicate(
                text.strip(),
                self.voice,
                rate=self.rate,
                volume=self.volume,
                pitch=self.pitch,
            )

            # Collect MP3 chunks
            mp3_data = b""
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    mp3_data += chunk["data"]

            if not mp3_data:
                logger.warning(f"Edge TTS returned no audio for: '{text[:50]}...'")
                return None

            # Convert MP3 to PCM
            pcm_data = await asyncio.get_event_loop().run_in_executor(
                None, mp3_to_pcm, mp3_data, self.sample_rate
            )

            elapsed_ms = (time.time() - start) * 1000

            if pcm_data:
                duration_sec = len(pcm_data) / (self.sample_rate * 2)
                logger.info(
                    f"TTS [{elapsed_ms:.0f}ms]: {len(pcm_data)} bytes "
                    f"({duration_sec:.1f}s) for '{text[:40]}...'"
                )

            return pcm_data

        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            return None

    async def synthesize_streaming(self, text: str):
        """
        Synthesize text and yield PCM chunks as they become available.
        Edge TTS streams MP3, so we buffer and convert in chunks.

        For now, this collects all MP3 first then converts (Edge TTS
        doesn't emit cleanly splittable MP3 chunks). True streaming
        would require an MP3 decoder that handles partial frames.

        Args:
            text: Text to synthesize

        Yields:
            PCM audio chunks
        """
        pcm = await self.synthesize(text)
        if pcm:
            yield pcm


# ═══════════════════════════════════════════════════════════════
# Standalone test
# ═══════════════════════════════════════════════════════════════

async def test_edge_tts():
    """Test Edge TTS with different voices and languages."""
    import wave

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("\n" + "=" * 50)
    print("Testing Edge TTS (FREE, no API key)")
    print("=" * 50)

    client = EdgeTTSClient(voice="madhur")

    tests = [
        ("Hindi", "Namaste, main Mochan hoon. Aap kaise hain bhai?"),
        ("Hinglish", "Arre bhai, yeh problem solve ho jayegi. Let me help you with this."),
        ("English", "Hello, I am Mochan, your AI meeting assistant. How can I help?"),
        ("Emotional", "Koi baat nahi bhai, main samajh sakta hoon. Hum yeh solve kar lenge."),
    ]

    for lang, text in tests:
        print(f"\n  {lang}: '{text}'")

        start = time.time()
        pcm = await client.synthesize(text)
        elapsed = (time.time() - start) * 1000

        if pcm:
            duration = len(pcm) / (16000 * 2)
            print(f"  -> {len(pcm)} bytes ({duration:.1f}s audio) in {elapsed:.0f}ms")

            filename = f"test_edge_{lang.lower()}.wav"
            with wave.open(filename, 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(pcm)
            print(f"  -> Saved to {filename}")
        else:
            print(f"  -> FAILED")

    # Test different voices
    print("\n  --- Voice comparison ---")
    text = "Mochan yahan hai, aapki seva mein."
    for name, voice_id in INDIAN_VOICES.items():
        c = EdgeTTSClient(voice=name)
        pcm = await c.synthesize(text)
        if pcm:
            duration = len(pcm) / (16000 * 2)
            print(f"  {name:15s} ({voice_id}): {duration:.1f}s audio")

    print("\n  Done!")


if __name__ == "__main__":
    asyncio.run(test_edge_tts())
