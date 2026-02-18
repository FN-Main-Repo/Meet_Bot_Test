"""
Sarvam Streaming Client - Parallel STT + TTS via WebSocket
Optimized for low-latency voice pipeline.

Features:
- Streaming STT with built-in VAD
- Streaming TTS for real-time audio generation
- Parallel processing support
"""

import asyncio
import json
import base64
import logging
import time
import struct
import numpy as np
from typing import AsyncGenerator, Optional, Callable
import websockets
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger("sarvam-streaming")


def parse_wav_to_pcm(wav_bytes: bytes, target_sample_rate: int = 16000) -> tuple[bytes, int]:
    """
    Parse WAV file and extract PCM data, resampling if necessary.
    
    Args:
        wav_bytes: Raw WAV file bytes
        target_sample_rate: Desired output sample rate
        
    Returns:
        Tuple of (pcm_bytes, actual_sample_rate)
    """
    if wav_bytes[:4] != b'RIFF':
        # Not a WAV file, assume raw PCM
        return wav_bytes, target_sample_rate
    
    # Parse WAV header
    # Bytes 20-21: Audio format (1 = PCM)
    audio_format = struct.unpack('<H', wav_bytes[20:22])[0]
    
    # Bytes 22-23: Number of channels
    num_channels = struct.unpack('<H', wav_bytes[22:24])[0]
    
    # Bytes 24-27: Sample rate
    source_sample_rate = struct.unpack('<I', wav_bytes[24:28])[0]
    
    # Bytes 34-35: Bits per sample
    bits_per_sample = struct.unpack('<H', wav_bytes[34:36])[0]
    
    logger.info(f"WAV format: {source_sample_rate}Hz, {num_channels}ch, {bits_per_sample}bit")
    
    # Find 'data' chunk - it may not be at byte 36
    pos = 36
    data_start = 44  # Default
    data_size = len(wav_bytes) - 44
    
    while pos < len(wav_bytes) - 8:
        chunk_id = wav_bytes[pos:pos+4]
        chunk_size = struct.unpack('<I', wav_bytes[pos+4:pos+8])[0]
        
        if chunk_id == b'data':
            data_start = pos + 8
            data_size = chunk_size
            break
        
        pos += 8 + chunk_size
        # WAV chunks are word-aligned
        if chunk_size % 2 == 1:
            pos += 1
    
    # Extract PCM data
    pcm_data = wav_bytes[data_start:data_start + data_size]
    
    # Convert stereo to mono if needed
    if num_channels == 2:
        audio_array = np.frombuffer(pcm_data, dtype=np.int16)
        # Average left and right channels
        mono_array = ((audio_array[0::2].astype(np.int32) + audio_array[1::2].astype(np.int32)) // 2).astype(np.int16)
        pcm_data = mono_array.tobytes()
    
    # Resample if sample rates don't match
    if source_sample_rate != target_sample_rate:
        logger.info(f"Resampling audio: {source_sample_rate}Hz -> {target_sample_rate}Hz")
        audio_array = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Calculate new length
        ratio = target_sample_rate / source_sample_rate
        new_length = int(len(audio_array) * ratio)
        
        # Resample using linear interpolation
        resampled = np.interp(
            np.linspace(0, len(audio_array) - 1, new_length),
            np.arange(len(audio_array)),
            audio_array.astype(np.float32)
        ).astype(np.int16)
        
        pcm_data = resampled.tobytes()
    
    return pcm_data, target_sample_rate


class SarvamStreamingSTT:
    """
    Streaming Speech-to-Text with Sarvam's WebSocket API.
    Supports real-time transcription with built-in VAD.
    """
    
    WS_URL = "wss://api.sarvam.ai/speech-to-text/streaming"
    
    def __init__(
        self,
        api_key: str,
        language: str = "hi-IN",
        model: str = "saarika:v2.5",
        sample_rate: int = 16000
    ):
        self.api_key = api_key
        self.language = language
        self.model = model
        self.sample_rate = sample_rate
        self.ws: Optional[WebSocketClientProtocol] = None
        self.is_connected = False
        self._receive_task: Optional[asyncio.Task] = None
        self._transcript_callback: Optional[Callable] = None
        self._final_transcript: str = ""
        self._interim_transcript: str = ""
        
    async def connect(self):
        """Connect to Sarvam streaming STT WebSocket."""
        try:
            headers = {
                "api-subscription-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            self.ws = await websockets.connect(
                self.WS_URL,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10
            )
            
            # Send configuration
            config = {
                "type": "config",
                "model": self.model,
                "language_code": self.language,
                "sample_rate": self.sample_rate,
                "input_audio_codec": "pcm_s16le",
                "vad_signals": True,  # Enable Sarvam's VAD
                "high_vad_sensitivity": False
            }
            await self.ws.send(json.dumps(config))
            
            # Wait for config acknowledgment
            response = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
            data = json.loads(response)
            
            if data.get("type") == "config_ack" or data.get("status") == "ready":
                self.is_connected = True
                logger.info(f"✓ Sarvam STT connected (model: {self.model})")
                return True
            else:
                logger.warning(f"Unexpected config response: {data}")
                self.is_connected = True  # Try anyway
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to Sarvam STT: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Sarvam STT."""
        self.is_connected = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self.ws:
            await self.ws.close()
            self.ws = None
        logger.info("Sarvam STT disconnected")
    
    async def send_audio(self, audio_chunk: bytes):
        """
        Stream audio chunk to Sarvam for transcription.
        Call this continuously as audio comes in.
        
        Args:
            audio_chunk: Raw PCM audio bytes (16-bit, mono, 16kHz)
        """
        if not self.is_connected or not self.ws:
            return
        
        try:
            # Send as binary frame
            await self.ws.send(audio_chunk)
        except Exception as e:
            logger.error(f"Error sending audio to Sarvam STT: {e}")
    
    async def flush(self):
        """Signal end of audio input and get final transcript."""
        if not self.is_connected or not self.ws:
            return
        
        try:
            # Send flush signal
            await self.ws.send(json.dumps({"type": "flush"}))
        except Exception as e:
            logger.error(f"Error flushing Sarvam STT: {e}")
    
    async def receive_transcripts(self) -> AsyncGenerator[dict, None]:
        """
        Receive streaming transcription results.
        
        Yields:
            dict with keys:
                - type: "interim" or "final"
                - transcript: The transcribed text
                - is_final: Boolean indicating if this is final
        """
        if not self.ws:
            return
        
        try:
            while self.is_connected:
                try:
                    msg = await asyncio.wait_for(self.ws.recv(), timeout=0.1)
                    data = json.loads(msg)
                    
                    msg_type = data.get("type", "")
                    
                    if msg_type == "transcript" or "transcript" in data:
                        transcript = data.get("transcript", "")
                        is_final = data.get("is_final", False) or data.get("type") == "final"
                        
                        if is_final:
                            self._final_transcript = transcript
                        else:
                            self._interim_transcript = transcript
                        
                        yield {
                            "type": "final" if is_final else "interim",
                            "transcript": transcript,
                            "is_final": is_final
                        }
                        
                        if is_final:
                            return
                    
                    elif msg_type == "vad_event":
                        # VAD events from Sarvam
                        vad_type = data.get("vad_type", "")
                        yield {
                            "type": "vad",
                            "vad_type": vad_type
                        }
                    
                    elif msg_type == "error":
                        logger.error(f"Sarvam STT error: {data}")
                        return
                        
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Sarvam STT connection closed")
                    return
                    
        except Exception as e:
            logger.error(f"Error receiving transcripts: {e}")
    
    def get_final_transcript(self) -> str:
        """Get the final transcript accumulated so far."""
        return self._final_transcript or self._interim_transcript


class SarvamStreamingTTS:
    """
    Streaming Text-to-Speech with Sarvam's WebSocket API.
    Generates audio in real-time as text is streamed.
    """
    
    WS_URL = "wss://api.sarvam.ai/text-to-speech/streaming"
    
    def __init__(
        self,
        api_key: str,
        speaker: str = "rahul",
        model: str = "bulbul:v3",
        sample_rate: int = 16000,
        language: str = "hi-IN"
    ):
        self.api_key = api_key
        self.speaker = speaker
        self.model = model
        self.sample_rate = sample_rate
        self.language = language
        self.ws: Optional[WebSocketClientProtocol] = None
        self.is_connected = False
        
    async def connect(self):
        """Connect to Sarvam streaming TTS WebSocket."""
        try:
            headers = {
                "api-subscription-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            self.ws = await websockets.connect(
                self.WS_URL,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10
            )
            
            # Send configuration
            config = {
                "type": "config",
                "model": self.model,
                "speaker": self.speaker,
                "language_code": self.language,
                "sample_rate": self.sample_rate,
                "audio_format": "pcm",  # Raw PCM for low latency
                "min_buffer_size": 30,  # Flush after 30 chars for low latency
                "pace": 1.0,
                "loudness": 1.0
            }
            await self.ws.send(json.dumps(config))
            
            # Wait for config acknowledgment
            response = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
            data = json.loads(response)
            
            if data.get("type") == "config_ack" or data.get("status") == "ready":
                self.is_connected = True
                logger.info(f"✓ Sarvam TTS connected (speaker: {self.speaker})")
                return True
            else:
                logger.warning(f"Unexpected TTS config response: {data}")
                self.is_connected = True
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to Sarvam TTS: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Sarvam TTS."""
        self.is_connected = False
        if self.ws:
            await self.ws.close()
            self.ws = None
        logger.info("Sarvam TTS disconnected")
    
    async def send_text(self, text: str):
        """
        Stream text chunk to TTS for synthesis.
        Call this as LLM tokens arrive.
        
        Args:
            text: Text chunk to synthesize
        """
        if not self.is_connected or not self.ws:
            return
        
        try:
            await self.ws.send(json.dumps({
                "type": "text",
                "text": text
            }))
        except Exception as e:
            logger.error(f"Error sending text to Sarvam TTS: {e}")
    
    async def flush(self):
        """Signal end of text input."""
        if not self.is_connected or not self.ws:
            return
        
        try:
            await self.ws.send(json.dumps({"type": "flush"}))
        except Exception as e:
            logger.error(f"Error flushing Sarvam TTS: {e}")
    
    async def receive_audio(self) -> AsyncGenerator[bytes, None]:
        """
        Receive streaming audio chunks.
        
        Yields:
            Raw PCM audio bytes (16-bit, mono)
        """
        if not self.ws:
            return
        
        try:
            while self.is_connected:
                try:
                    msg = await asyncio.wait_for(self.ws.recv(), timeout=0.1)
                    
                    # Check if binary (raw audio) or JSON
                    if isinstance(msg, bytes):
                        yield msg
                    else:
                        data = json.loads(msg)
                        msg_type = data.get("type", "")
                        
                        if msg_type == "audio":
                            # Base64 encoded audio
                            audio_b64 = data.get("audio", "")
                            if audio_b64:
                                yield base64.b64decode(audio_b64)
                        
                        elif msg_type == "done" or msg_type == "end":
                            logger.debug("TTS synthesis complete")
                            return
                        
                        elif msg_type == "error":
                            logger.error(f"Sarvam TTS error: {data}")
                            return
                            
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Sarvam TTS connection closed")
                    return
                    
        except Exception as e:
            logger.error(f"Error receiving audio: {e}")


class SarvamRESTClient:
    """
    REST client for Sarvam APIs (fallback when WebSocket is unavailable).
    """
    
    STT_URL = "https://api.sarvam.ai/speech-to-text"
    TTS_URL = "https://api.sarvam.ai/text-to-speech"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "api-subscription-key": api_key,
        }
    
    async def transcribe(
        self,
        audio_data: bytes,
        language: str = "hi-IN",
        model: str = "saarika:v2.5"
    ) -> Optional[str]:
        """
        Transcribe audio using REST API (non-streaming fallback).
        
        Args:
            audio_data: Raw PCM audio bytes
            language: Language code
            model: STT model to use
            
        Returns:
            Transcribed text or None
        """
        import aiohttp
        import io
        import wave
        
        try:
            # Convert PCM to WAV
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_data)
            wav_buffer.seek(0)
            
            # Create form data
            form = aiohttp.FormData()
            form.add_field('file', wav_buffer, filename='audio.wav', content_type='audio/wav')
            form.add_field('model', model)
            form.add_field('language_code', language)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.STT_URL,
                    headers={"api-subscription-key": self.api_key},
                    data=form,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("transcript", "")
                    else:
                        error = await response.text()
                        logger.error(f"Sarvam STT REST error: {response.status} - {error}")
                        return None
                        
        except Exception as e:
            logger.error(f"Sarvam STT REST error: {e}")
            return None
    
    async def synthesize(
        self,
        text: str,
        speaker: str = "rahul",
        language: str = "hi-IN",
        model: str = "bulbul:v3"
    ) -> Optional[bytes]:
        """
        Synthesize speech using REST API (non-streaming fallback).
        
        Args:
            text: Text to synthesize
            speaker: Voice to use
            language: Language code
            model: TTS model to use
            
        Returns:
            Raw PCM audio bytes or None
        """
        import aiohttp
        
        try:
            payload = {
                "text": text,
                "model": model,
                "speaker": speaker,
                "language_code": language,
                "sample_rate": 16000,
                "audio_format": "wav"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.TTS_URL,
                    headers={
                        "api-subscription-key": self.api_key,
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Sarvam returns "audios" array, not "audio"
                        audios = data.get("audios", [])
                        audio_b64 = audios[0] if audios else ""
                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            # Properly parse WAV and resample to 16kHz for Attendee
                            pcm_data, _ = parse_wav_to_pcm(audio_bytes, target_sample_rate=16000)
                            return pcm_data
                        return None
                    else:
                        error = await response.text()
                        logger.error(f"Sarvam TTS REST error: {response.status} - {error}")
                        return None
                        
        except Exception as e:
            logger.error(f"Sarvam TTS REST error: {e}")
            return None


# Test function
async def test_sarvam_streaming():
    """Test Sarvam streaming capabilities."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv(".env.local")
    api_key = os.getenv("SARVAM_API_KEY")
    
    if not api_key:
        print("❌ SARVAM_API_KEY not found in environment")
        return
    
    print("Testing Sarvam Streaming APIs...")
    
    # Test TTS
    print("\n1. Testing Streaming TTS...")
    tts = SarvamStreamingTTS(api_key, speaker="rahul")
    
    if await tts.connect():
        await tts.send_text("नमस्ते, मैं जार्विस हूं। ")
        await tts.send_text("आप कैसे हैं भाई?")
        await tts.flush()
        
        audio_chunks = []
        async for chunk in tts.receive_audio():
            audio_chunks.append(chunk)
            print(f"  Received audio chunk: {len(chunk)} bytes")
        
        if audio_chunks:
            total_audio = b''.join(audio_chunks)
            print(f"  ✓ Total audio: {len(total_audio)} bytes")
        
        await tts.disconnect()
    else:
        print("  ❌ Failed to connect to TTS")
    
    print("\n✓ Test complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(test_sarvam_streaming())
