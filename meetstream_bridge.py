"""
MeetStream.ai Bridge for Mochan Voice Assistant
Complete rewrite with proper streaming, VAD, and emotion-aware pipeline.

API Domain: https://api.meetstream.ai/api/v1
Auth: Authorization: Token <key>
Flow: We start WS server -> create bot -> bot connects TO us -> ready event -> bidirectional audio

IMPORTANT: MeetStream opens MULTIPLE WebSocket connections to our server:
  1. Main control socket (JSON events: ready, participant.joined, etc.)
  2. Live audio socket (audio data — may be binary or JSON)
  3. Possibly a transcript socket
We must handle ALL of them without overwriting each other.
"""

import asyncio
import json
import base64
import logging
import os
import time
import re
import struct
from typing import Optional

import aiohttp
import websockets
import numpy as np
from dotenv import load_dotenv

try:
    from scipy.signal import resample_poly
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from optimized_pipeline import OptimizedPipeline

load_dotenv()
load_dotenv(".env.local", override=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("meetstream-bridge")

# Silence noisy loggers
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════
# Audio Resampling (16kHz -> 48kHz for MeetStream output)
# ═══════════════════════════════════════════════════════════════

# MeetStream expects 48kHz audio output (their official agent uses 48000)
OUTGOING_AUDIO_RATE = int(os.getenv("MEETSTREAM_OUT_RATE", "48000"))


def resample_pcm16(pcm_bytes: bytes, src_hz: int, dst_hz: int) -> bytes:
    """
    Resample PCM16 audio from src_hz to dst_hz.
    Uses scipy polyphase resampling if available, else linear interpolation.
    """
    if src_hz == dst_hz:
        return pcm_bytes
    if len(pcm_bytes) < 2:
        return pcm_bytes

    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)

    if _HAS_SCIPY:
        from math import gcd
        g = gcd(src_hz, dst_hz)
        up, down = dst_hz // g, src_hz // g
        y = resample_poly(x, up=up, down=down)
    else:
        # Linear interpolation fallback
        n_out = int(len(x) * (dst_hz / src_hz))
        t_src = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
        t_dst = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        y = np.interp(t_dst, t_src, x)

    y = np.clip(y, -32768, 32767).astype(np.int16)
    return y.tobytes()


# ═══════════════════════════════════════════════════════════════
# Simple Energy-Based VAD
# ═══════════════════════════════════════════════════════════════

class SimpleVAD:
    """
    Energy-based Voice Activity Detector.

    IMPORTANT: MeetStream sends audio in LARGE chunks (~3s / ~98KB each)
    with LONG gaps between chunks (3-10 seconds).

    This VAD:
    1. Sub-divides large chunks into 20ms frames for speech/silence detection
    2. Tracks wall-clock time between chunks for inter-chunk silence detection
    3. Flushes buffered speech if no new audio arrives within gap_timeout
    """

    FRAME_MS = 20  # Analyze in 20ms frames

    def __init__(
        self,
        sample_rate: int = 16000,
        energy_threshold: float = 200,
        min_speech_ms: int = 200,
        min_silence_ms: int = 600,
        max_utterance_sec: float = 15.0,
        gap_timeout_sec: float = 2.0,  # Flush speech if no audio for this long
    ):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.frame_size = int(self.FRAME_MS / 1000 * sample_rate) * 2  # bytes per frame
        self.min_speech_frames = int(min_speech_ms / self.FRAME_MS)
        self.min_silence_frames = int(min_silence_ms / self.FRAME_MS)
        self.max_utterance_bytes = int(max_utterance_sec * sample_rate * 2)
        self.gap_timeout_sec = gap_timeout_sec

        self.is_speaking = False
        self.speech_buffer = bytearray()
        self.silence_frames = 0
        self.speech_frames = 0
        self._leftover = bytearray()  # Buffer for incomplete frames
        self._last_audio_time = 0.0  # Wall-clock time of last audio chunk

    def check_gap_timeout(self) -> Optional[bytes]:
        """
        Check if speech should be flushed due to gap between audio chunks.
        Call this periodically (e.g., every second) to handle cases where
        MeetStream stops sending audio while user is mid-sentence.

        Returns:
            Complete utterance bytes if flushed, None otherwise
        """
        if not self.is_speaking or not self.speech_buffer:
            return None

        if self._last_audio_time <= 0:
            return None

        gap = time.time() - self._last_audio_time
        if gap >= self.gap_timeout_sec:
            duration = len(self.speech_buffer) / (self.sample_rate * 2)
            logger.info(f"VAD: Gap timeout ({gap:.1f}s gap) — flushing {duration:.1f}s utterance")
            utterance = bytes(self.speech_buffer)
            self.reset()
            return utterance

        return None

    def process(self, audio_chunk: bytes) -> list:
        """
        Feed audio chunk (can be any size). Returns list of complete utterances.

        MeetStream sends ~3s chunks. We sub-divide into 20ms frames for
        proper VAD analysis. May return 0, 1, or multiple utterances.

        Args:
            audio_chunk: Raw PCM audio (16-bit, mono, 16kHz)

        Returns:
            List of complete utterance bytes (may be empty)
        """
        utterances = []
        self._last_audio_time = time.time()

        if len(audio_chunk) < 2:
            return utterances

        # Prepend any leftover bytes from last chunk
        data = bytes(self._leftover) + audio_chunk
        self._leftover = bytearray()

        # Process in small frames
        offset = 0
        while offset + self.frame_size <= len(data):
            frame = data[offset:offset + self.frame_size]
            offset += self.frame_size

            frame_array = np.frombuffer(frame, dtype=np.int16)
            energy = np.sqrt(np.mean(frame_array.astype(np.float32) ** 2))
            is_speech = energy > self.energy_threshold

            if is_speech:
                self.silence_frames = 0
                self.speech_frames += 1

                if not self.is_speaking and self.speech_frames >= self.min_speech_frames:
                    self.is_speaking = True
                    logger.info(f"VAD: Speech started (energy={energy:.0f})")

                if self.is_speaking:
                    self.speech_buffer.extend(frame)

                    # Safety cap
                    if len(self.speech_buffer) >= self.max_utterance_bytes:
                        duration = len(self.speech_buffer) / (self.sample_rate * 2)
                        logger.info(f"VAD: Max utterance reached ({duration:.1f}s)")
                        utterances.append(bytes(self.speech_buffer))
                        self.reset()
                        self._last_audio_time = time.time()
            else:
                if self.is_speaking:
                    self.silence_frames += 1
                    self.speech_buffer.extend(frame)

                    if self.silence_frames >= self.min_silence_frames:
                        duration = len(self.speech_buffer) / (self.sample_rate * 2)
                        logger.info(f"VAD: Speech ended ({duration:.1f}s, silence_frames={self.silence_frames})")
                        utterances.append(bytes(self.speech_buffer))
                        self.reset()
                        self._last_audio_time = time.time()
                else:
                    self.speech_frames = 0

        # Save leftover bytes for next chunk
        if offset < len(data):
            self._leftover = bytearray(data[offset:])

        return utterances

    def reset(self):
        """Reset VAD state."""
        self.is_speaking = False
        self.speech_buffer = bytearray()
        self.silence_frames = 0
        self.speech_frames = 0


# ═══════════════════════════════════════════════════════════════
# MeetStream API Bridge
# ═══════════════════════════════════════════════════════════════

class MeetStreamBridge:
    """
    Handles MeetStream.ai bot lifecycle and WebSocket communication.

    IMPORTANT: MeetStream makes MULTIPLE WebSocket connections:
    - Control socket: JSON events (ready, participant events, etc.)
    - Audio socket: Live audio data (binary PCM or JSON with audio)
    - Transcript socket: Live transcription data

    We handle all of them on the same server, differentiating by
    the first message received on each connection.

    Flow:
    1. start_socket_server() — we listen for bot connections
    2. create_bot() — tells MeetStream to join meeting and connect to us
    3. Bot sends "ready" event over control socket
    4. Audio starts flowing on audio socket
    5. We send "sendaudio" commands on control socket
    6. Bot plays audio in the meeting
    """

    def __init__(
        self,
        api_key: str,
        bot_name: str = "Mochan",
        on_audio_received=None,
        on_transcript_received=None,
    ):
        self.api_key = api_key
        self.bot_name = bot_name
        self.on_audio_received = on_audio_received
        self.on_transcript_received = on_transcript_received

        self.base_url = "https://api.meetstream.ai/api/v1"
        self.bot_id: Optional[str] = None
        self.bot_socket = None  # Main control socket (for sending commands)
        self.incoming_sample_rate = int(os.getenv("MEETSTREAM_IN_RATE", "48000"))  # MeetStream sends 48kHz
        self.internal_sample_rate = 16000  # Our VAD/STT works at 16kHz
        self.sample_rate = self.internal_sample_rate  # For backward compatibility
        self._running = False
        self._ready = asyncio.Event()

        # Track all connections
        self._connections = {}  # id -> {"type": "control"|"audio"|"unknown", "ws": websocket}
        self._conn_counter = 0
        self._audio_chunks_received = 0
        self._audio_bytes_received = 0

    def _headers(self) -> dict:
        return {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

    async def create_bot(
        self,
        meeting_link: str,
        socket_connection_url: str,
        live_audio_ws_url: Optional[str] = None,
    ) -> str:
        """Create MeetStream bot and join the meeting."""
        payload = {
            "meeting_link": meeting_link,
            "audio_required": True,
            "video_required": False,
            "bot_name": self.bot_name,
            "bot_message": f"Hi, I'm {self.bot_name}, your AI meeting assistant!",
            "socket_connection_url": socket_connection_url,
            "automatic_leave": {
                "waiting_room_timeout": 300,
                "everyone_left_timeout": 60,
            },
        }

        if live_audio_ws_url:
            payload["live_audio_required"] = {
                "websocket_url": live_audio_ws_url,
            }

        logger.info(f"Creating bot for: {meeting_link}")
        logger.info(f"Control socket URL: {socket_connection_url}")
        if live_audio_ws_url:
            logger.info(f"Audio socket URL: {live_audio_ws_url}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/bots/create_bot",
                headers=self._headers(),
                json=payload,
            ) as resp:
                response_text = await resp.text()

                if resp.status not in [200, 201]:
                    raise Exception(
                        f"Failed to create bot: {resp.status} - {response_text}"
                    )

                data = json.loads(response_text)
                self.bot_id = data.get("id") or data.get("bot_id")
                logger.info(f"Bot created: {self.bot_id}")
                logger.info(f"Create response: {response_text[:500]}")
                return self.bot_id

    async def send_audio(self, pcm_data: bytes, chunk_duration_ms: int = 100):
        """
        Send audio to the meeting. Resamples from 16kHz to 48kHz (MeetStream requirement).

        MeetStream's official agent uses:
        - 48kHz sample rate
        - PCM16 (signed 16-bit little-endian)
        - Mono (1 channel)
        - Fields: command, audiochunk, bot_id, sample_rate, encoding, channels, endianness

        Args:
            pcm_data: Raw PCM audio (16-bit, mono, 16kHz from Edge TTS)
            chunk_duration_ms: Size of each chunk in ms (100ms = good balance)
        """
        if not self.bot_socket:
            logger.warning("Bot socket not connected")
            return

        if not self._ready.is_set():
            logger.warning("Bot not ready, waiting...")
            try:
                await asyncio.wait_for(self._ready.wait(), timeout=10)
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for bot ready")
                return

        # Step 1: Resample 16kHz -> 48kHz (MeetStream expects 48kHz)
        pcm_48k = resample_pcm16(pcm_data, src_hz=self.sample_rate, dst_hz=OUTGOING_AUDIO_RATE)

        # Step 2: Send in chunks for smooth playback
        # Chunk size based on OUTGOING rate (48kHz * 2 bytes * chunk_ms/1000)
        chunk_size = int((chunk_duration_ms / 1000.0) * OUTGOING_AUDIO_RATE * 2)
        chunks_sent = 0

        for i in range(0, len(pcm_48k), chunk_size):
            chunk = pcm_48k[i : i + chunk_size]
            if not chunk:
                continue

            audio_b64 = base64.b64encode(chunk).decode("utf-8")

            # Full message format matching MeetStream's official agent
            message = {
                "command": "sendaudio",
                "audiochunk": audio_b64,
                "bot_id": self.bot_id,
                "sample_rate": OUTGOING_AUDIO_RATE,
                "encoding": "pcm16",
                "channels": 1,
                "endianness": "little",
            }

            try:
                await self.bot_socket.send(json.dumps(message))
                chunks_sent += 1
                # Small yield to prevent flooding the socket
                if chunks_sent % 10 == 0:
                    await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error sending audio chunk: {e}")
                break

        duration_sec = len(pcm_data) / (self.sample_rate * 2)
        logger.info(
            f"Sent {chunks_sent} chunks ({duration_sec:.1f}s audio) to meeting "
            f"[resampled 16kHz→{OUTGOING_AUDIO_RATE}Hz, {len(pcm_48k)} bytes]"
        )

    async def send_chat_message(self, message: str):
        """Send a chat message in the meeting."""
        if not self.bot_socket or not self._ready.is_set():
            return

        msg = {
            "command": "sendmsg",
            "message": message,
            "bot_id": self.bot_id,
        }
        try:
            await self.bot_socket.send(json.dumps(msg))
            logger.info(f"Chat: {message}")
        except Exception as e:
            logger.error(f"Error sending chat: {e}")

    async def start_socket_server(self, host: str = "0.0.0.0", port: int = 8765):
        """Start WebSocket server that the MeetStream bot connects TO."""

        async def handle_bot_connection(websocket, path="/"):
            """
            Handle ANY incoming WebSocket connection from MeetStream.
            MeetStream may open multiple connections (control, audio, transcript).
            We identify each by the first message it sends.
            """
            self._conn_counter += 1
            conn_id = self._conn_counter
            remote = websocket.remote_address
            request_path = getattr(websocket, 'path', path)

            logger.info(f"[Conn #{conn_id}] New WebSocket connection from {remote}, path={request_path}")
            logger.info(f"[Conn #{conn_id}] Headers: {dict(websocket.request_headers) if hasattr(websocket, 'request_headers') else 'N/A'}")

            self._connections[conn_id] = {"type": "unknown", "ws": websocket}

            # The first connection that sends a "ready" event is the control socket
            # Connections that send binary data or audio events are audio sockets
            conn_type = "unknown"
            is_first_message = True

            try:
                async for message in websocket:
                    # ─── Handle binary messages (likely raw audio) ───
                    if isinstance(message, bytes):
                        if is_first_message:
                            conn_type = "audio_binary"
                            self._connections[conn_id]["type"] = conn_type
                            logger.info(f"[Conn #{conn_id}] Identified as BINARY AUDIO socket (first msg: {len(message)} bytes)")
                            is_first_message = False

                        self._audio_chunks_received += 1
                        self._audio_bytes_received += len(message)

                        # Log periodically
                        if self._audio_chunks_received % 500 == 1:
                            logger.info(
                                f"[Audio] Received {self._audio_chunks_received} chunks, "
                                f"{self._audio_bytes_received / 1024:.1f} KB total"
                            )

                        # Downsample 48kHz -> 16kHz, then feed to audio handler
                        if self.on_audio_received:
                            if self.incoming_sample_rate != self.internal_sample_rate:
                                pcm_16k = resample_pcm16(
                                    message, src_hz=self.incoming_sample_rate, dst_hz=self.internal_sample_rate
                                )
                            else:
                                pcm_16k = message
                            await self.on_audio_received(pcm_16k, self.internal_sample_rate)
                        continue

                    # ─── Handle text/JSON messages ───
                    if is_first_message:
                        logger.info(f"[Conn #{conn_id}] First message (text, {len(message)} chars): {message[:500]}")
                        is_first_message = False

                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        logger.warning(f"[Conn #{conn_id}] Non-JSON text message ({len(message)} chars): {message[:200]}")
                        # Could be raw audio as text? Try base64 decode
                        if len(message) > 100 and self.on_audio_received:
                            try:
                                pcm = base64.b64decode(message)
                                if len(pcm) > 0:
                                    conn_type = "audio_b64_raw"
                                    self._connections[conn_id]["type"] = conn_type
                                    logger.info(f"[Conn #{conn_id}] Detected raw base64 audio stream")
                                    self._audio_chunks_received += 1
                                    self._audio_bytes_received += len(pcm)
                                    await self.on_audio_received(pcm, self.sample_rate)
                            except Exception:
                                pass
                        continue

                    event_type = data.get("type")
                    command = data.get("command")

                    # ─── Identify connection type by first meaningful event ───
                    if event_type == "ready":
                        ready_msg = data.get("message", "")
                        self.bot_id = data.get("bot_id", self.bot_id)

                        if "receive" in ready_msg.lower():
                            # "Ready to receive messages" = command socket (we SEND to this)
                            conn_type = "command"
                            self._connections[conn_id]["type"] = conn_type
                            self.bot_socket = websocket
                            self._running = True
                            logger.info(f"[Conn #{conn_id}] ★ COMMAND SOCKET — '{ready_msg}' — Bot ID: {self.bot_id}")
                            self._ready.set()
                        elif "send" in ready_msg.lower() or "audio" in ready_msg.lower():
                            # "Ready to send audio chunks" = audio source socket (we RECEIVE from this)
                            conn_type = "audio_source"
                            self._connections[conn_id]["type"] = conn_type
                            logger.info(f"[Conn #{conn_id}] ★ AUDIO SOURCE SOCKET — '{ready_msg}' — Bot ID: {self.bot_id}")
                            # Don't set _ready here — we need the COMMAND socket to send audio/chat
                            self._running = True
                        else:
                            # Unknown ready message — treat as control if we don't have one
                            conn_type = "control"
                            self._connections[conn_id]["type"] = conn_type
                            if not self.bot_socket:
                                self.bot_socket = websocket
                                self._running = True
                            logger.info(f"[Conn #{conn_id}] ★ CONTROL SOCKET — '{ready_msg}' — Bot ID: {self.bot_id}")
                            self._ready.set()

                    elif event_type == "PCMChunk":
                        # MeetStream's actual audio format: {"type":"PCMChunk","audioData":"base64..."}
                        # Audio arrives at 48kHz, we downsample to 16kHz for VAD/STT
                        if conn_type not in ("audio_source", "audio_pcmchunk"):
                            conn_type = "audio_pcmchunk"
                            self._connections[conn_id]["type"] = conn_type
                            logger.info(f"[Conn #{conn_id}] Identified as PCMChunk AUDIO socket (incoming={self.incoming_sample_rate}Hz)")

                        audio_b64 = data.get("audioData") or data.get("audio_data") or data.get("audio")
                        if audio_b64 and self.on_audio_received:
                            try:
                                pcm_bytes = base64.b64decode(audio_b64)
                                self._audio_chunks_received += 1
                                self._audio_bytes_received += len(pcm_bytes)

                                # Downsample 48kHz -> 16kHz for our VAD/STT
                                if self.incoming_sample_rate != self.internal_sample_rate:
                                    pcm_16k = resample_pcm16(
                                        pcm_bytes,
                                        src_hz=self.incoming_sample_rate,
                                        dst_hz=self.internal_sample_rate
                                    )
                                else:
                                    pcm_16k = pcm_bytes

                                if self._audio_chunks_received % 200 == 1:
                                    if len(pcm_16k) >= 2:
                                        arr = np.frombuffer(pcm_16k, dtype=np.int16)
                                        energy = np.sqrt(np.mean(arr.astype(np.float32) ** 2))
                                    else:
                                        energy = 0
                                    logger.info(
                                        f"[PCMChunk] #{self._audio_chunks_received}: "
                                        f"{len(pcm_bytes)}→{len(pcm_16k)} bytes ({self.incoming_sample_rate}→{self.internal_sample_rate}Hz), "
                                        f"energy={energy:.0f}, total={self._audio_bytes_received / 1024:.1f} KB"
                                    )

                                await self.on_audio_received(pcm_16k, self.internal_sample_rate)
                            except Exception as e:
                                logger.error(f"[Conn #{conn_id}] Error decoding PCMChunk: {e}")

                    elif event_type in ("audio.data", "audio"):
                        if conn_type == "unknown":
                            conn_type = "audio_json"
                            self._connections[conn_id]["type"] = conn_type
                            logger.info(f"[Conn #{conn_id}] Identified as JSON AUDIO socket")

                        # Extract audio data — handle multiple possible formats
                        audio_b64 = None
                        sample_rate = self.sample_rate

                        # Format 1: {"type": "audio.data", "data": {"audio": "base64...", "sample_rate": 16000}}
                        audio_data = data.get("data", {})
                        if isinstance(audio_data, dict):
                            audio_b64 = audio_data.get("audio") or audio_data.get("buffer") or audio_data.get("chunk")
                            sample_rate = audio_data.get("sample_rate", audio_data.get("sampleRate", self.sample_rate))

                        # Format 2: {"type": "audio", "audio": "base64..."}
                        if not audio_b64:
                            audio_b64 = data.get("audio") or data.get("buffer") or data.get("chunk") or data.get("audiochunk")

                        # Format 3: {"type": "audio.data", "audio_data": "base64..."}
                        if not audio_b64:
                            audio_b64 = data.get("audio_data")

                        if audio_b64 and self.on_audio_received:
                            try:
                                pcm_bytes = base64.b64decode(audio_b64)
                                self._audio_chunks_received += 1
                                self._audio_bytes_received += len(pcm_bytes)

                                if self._audio_chunks_received % 500 == 1:
                                    logger.info(
                                        f"[Audio JSON] Received {self._audio_chunks_received} chunks, "
                                        f"{self._audio_bytes_received / 1024:.1f} KB total, "
                                        f"sample_rate={sample_rate}"
                                    )

                                await self.on_audio_received(pcm_bytes, sample_rate)
                            except Exception as e:
                                logger.error(f"[Conn #{conn_id}] Error decoding audio: {e}")
                        elif self._audio_chunks_received == 0:
                            # Log first audio message structure for debugging
                            logger.warning(f"[Conn #{conn_id}] Audio event but no audio data found. Keys: {list(data.keys())}")
                            if isinstance(audio_data, dict):
                                logger.warning(f"  data sub-keys: {list(audio_data.keys())}")

                    elif event_type == "transcript":
                        text = data.get("text") or data.get("transcript", "")
                        speaker = data.get("speaker") or data.get("speakerName", "Unknown")
                        if text and self.on_transcript_received:
                            logger.info(f"[Transcript] [{speaker}]: {text}")
                            await self.on_transcript_received(text, speaker)

                    elif event_type == "participant.joined":
                        name = data.get("name") or data.get("participant", {}).get("name", "Unknown")
                        logger.info(f"★ Participant joined: {name}")

                    elif event_type == "participant.left":
                        name = data.get("name") or data.get("participant", {}).get("name", "Unknown")
                        logger.info(f"★ Participant left: {name}")

                    elif event_type == "bot.joined":
                        logger.info("★ Bot joined the meeting!")

                    elif event_type == "bot.left":
                        logger.info("★ Bot left the meeting")
                        self._running = False

                    elif event_type == "error":
                        logger.error(f"Bot error: {data.get('message', data)}")

                    elif command in ("sendaudio", "sendmsg"):
                        # Silently ignore MeetStream acknowledgments for our sent audio/messages
                        # These come back as: {"command":"sendaudio","status":"sent","bot_id":"...","timestamp":"..."}
                        # They flood the logs — hundreds per response. Just count them.
                        if not hasattr(self, '_ack_count'):
                            self._ack_count = 0
                        self._ack_count += 1
                        if self._ack_count % 200 == 1:
                            status = data.get("status", "unknown")
                            logger.info(f"[Conn #{conn_id}] Ack #{self._ack_count}: {command} status={status}")

                    else:
                        # LOG EVERYTHING we don't recognize — critical for debugging
                        msg_preview = message[:300] if len(message) > 300 else message
                        logger.info(f"[Conn #{conn_id}] UNHANDLED event type='{event_type}', command='{command}': {msg_preview}")

            except websockets.exceptions.ConnectionClosed as e:
                logger.info(f"[Conn #{conn_id}] Disconnected ({conn_type}): {e}")
            except Exception as e:
                logger.error(f"[Conn #{conn_id}] Error ({conn_type}): {e}", exc_info=True)
            finally:
                # Only clear control socket state if THIS was the command socket
                if conn_type in ("command", "control"):
                    self._running = False
                    self.bot_socket = None
                    self._ready.clear()
                    logger.info(f"[Conn #{conn_id}] Command/control socket closed")

                if conn_id in self._connections:
                    del self._connections[conn_id]

                logger.info(
                    f"[Conn #{conn_id}] Closed. Type was: {conn_type}. "
                    f"Active connections: {len(self._connections)}"
                )

        server = await websockets.serve(
            handle_bot_connection,
            host,
            port,
            max_size=10 * 1024 * 1024,  # 10MB max message size for audio
            ping_interval=20,
            ping_timeout=60,
        )
        logger.info(f"Socket server listening on ws://{host}:{port}")
        return server

    async def leave_meeting(self):
        """Remove bot from meeting."""
        if not self.bot_id:
            return

        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.base_url}/bots/{self.bot_id}",
                    headers=self._headers(),
                ) as resp:
                    if resp.status in [200, 204]:
                        logger.info("Bot left meeting")
                    else:
                        error = await resp.text()
                        logger.warning(f"Leave response: {resp.status} - {error}")
        except Exception as e:
            logger.error(f"Error leaving meeting: {e}")

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()


# ═══════════════════════════════════════════════════════════════
# Mochan Agent — Ties Pipeline + MeetStream + VAD Together
# ═══════════════════════════════════════════════════════════════

class MochanAgent:
    """
    Mochan Voice Assistant for Google Meet via MeetStream.

    Architecture:
    Google Meet <-> MeetStream Bot <-> WebSocket <-> This Agent
                                                        |
                                                    SimpleVAD
                                                        |
                                                  Groq Whisper STT
                                                  + Emotion Detect
                                                        |
                                                  Groq LLM (streaming)
                                                        |
                                                  Edge TTS
                                                        |
                                                  Audio -> Meet
    """

    def __init__(self, meetstream_api_key: str, bot_name: str = "Mochan"):
        # Pipeline
        self.pipeline = OptimizedPipeline()

        # VAD — tuned for MeetStream's sparse audio delivery
        # MeetStream sends ~3s chunks every 3-10s with gaps in between
        self.vad = SimpleVAD(
            sample_rate=16000,
            energy_threshold=300,
            min_speech_ms=200,       # Require 200ms of speech before triggering
            min_silence_ms=600,      # 600ms silence within a chunk ends utterance
            gap_timeout_sec=5.0,     # If no audio chunk for 5s while speaking, flush utterance
                                     # MeetStream gap between chunks is typically 3-5s, so 5s
                                     # means we flush only after a real pause, not between chunks
        )

        # MeetStream bridge
        self.bridge = MeetStreamBridge(
            api_key=meetstream_api_key,
            bot_name=bot_name,
            on_audio_received=self._on_audio_received,
            on_transcript_received=self._on_transcript_received,
        )

        # Wake word patterns
        self.wake_patterns = [
            r"mochan", r"mo[\s\-]?chan", r"mochandas",
            r"\u092e\u094b\u091a\u0928",         # मोचन
            r"\u092e\u094b\u091a\u0928\u0926\u093e\u0938",  # मोचनदास
            r"mohan", r"\u092e\u094b\u0939\u0928",  # मोहन
        ]

        # Stop patterns
        self.stop_patterns = [
            r"stop\s+mochan", r"mochan\s+stop",
            r"band\s+karo", r"ruk\s+jao",
            r"chup", r"bas",
        ]

        # State — start ACTIVE for testing (no wake word needed)
        self.is_active = True
        self.active_timeout = 300  # 5 min timeout for testing
        self.last_interaction_time = time.time()  # Start now since active
        self.cooldown_seconds = 2.0
        self.last_response_time = 0.0
        self._running = False
        self._processing_lock = asyncio.Lock()

        # Echo suppression: mute incoming audio while bot is speaking
        # MeetStream plays our audio in the meeting and sends it back to us as echo.
        # We must mute for the ENTIRE duration of the audio being played + buffer.
        self._is_speaking = False
        self._speaking_done_time = 0.0
        self._echo_guard_sec = 0.0  # Dynamically set based on audio length sent

        # Metrics
        self.metrics = {
            "utterances": 0,
            "responses": 0,
            "wake_activations": 0,
            "audio_callbacks": 0,
            "latency_samples": [],
        }

    # ─── Audio Handling ───────────────────────────────────────

    async def _on_audio_received(self, pcm_bytes: bytes, sample_rate: int):
        """Handle incoming audio from the meeting — feed to VAD."""
        self.metrics["audio_callbacks"] += 1

        if not self._running:
            return

        # ── ECHO SUPPRESSION ──
        # When the bot is speaking (sending TTS audio), mute incoming audio.
        # The bot's own TTS audio gets picked up by MeetStream and sent back to us,
        # producing garbage transcriptions like 'अवावी।', 'Júna, sjálik evol positives', etc.
        # Also keep a guard period after speaking finishes to catch echo tail.
        if self._is_speaking:
            # Bot is actively sending audio — drop all incoming audio, reset VAD
            if self.metrics["audio_callbacks"] % 100 == 1:
                logger.info("[Agent] Echo suppression: MUTED (bot is speaking)")
            self.vad.reset()
            return

        if self._speaking_done_time > 0:
            time_since_done = time.time() - self._speaking_done_time
            if time_since_done < self._echo_guard_sec:
                if self.metrics["audio_callbacks"] % 100 == 1:
                    logger.info(f"[Agent] Echo guard: MUTED ({time_since_done:.1f}s / {self._echo_guard_sec}s)")
                self.vad.reset()
                return

        # Log periodically so we know audio is flowing
        if self.metrics["audio_callbacks"] % 50 == 1:
            logger.info(
                f"[Agent] Audio callback #{self.metrics['audio_callbacks']}: "
                f"{len(pcm_bytes)} bytes, rate={sample_rate}"
            )

        # VAD now returns a LIST of utterances (may be 0, 1, or more)
        utterances = self.vad.process(pcm_bytes)

        for utterance in utterances:
            duration_sec = len(utterance) / (16000 * 2)
            logger.info(f"[Agent] Utterance detected! {duration_sec:.1f}s, {len(utterance)} bytes")
            asyncio.create_task(self._process_utterance(utterance))

    async def _on_transcript_received(self, text: str, speaker: str):
        """
        Handle transcript from MeetStream (if using their STT).
        This is a secondary path — primary is our own STT via audio.
        """
        logger.info(f"[Transcript] [{speaker}]: {text}")

        # Check cooldown
        now = time.time()
        if now - self.last_response_time < self.cooldown_seconds:
            return

        text_lower = text.lower()

        # Check wake word
        if any(re.search(p, text_lower) for p in self.wake_patterns):
            self._activate()
            asyncio.create_task(self._respond_to_text(text))

        elif self.is_active and now < self.last_interaction_time + self.active_timeout:
            self._activate()  # Refresh timeout
            asyncio.create_task(self._respond_to_text(text))

    # ─── Utterance Processing ─────────────────────────────────

    async def _process_utterance(self, audio_pcm: bytes):
        """Process a complete utterance through the full pipeline."""
        duration_sec = len(audio_pcm) / (16000 * 2)

        # Filter noise (min 0.5s)
        if duration_sec < 0.5:
            logger.info(f"[Agent] Utterance too short ({duration_sec:.1f}s), skipping")
            return

        # Prevent overlapping processing
        if self._processing_lock.locked():
            logger.info("Already processing, skipping utterance")
            return

        async with self._processing_lock:
            self.metrics["utterances"] += 1
            start_time = time.time()

            try:
                # 1. Transcribe with Groq Whisper
                logger.info(f"[Agent] Transcribing {duration_sec:.1f}s audio...")
                transcript = await self.pipeline.transcribe_audio(audio_pcm)

                if not transcript or len(transcript.strip()) < 2:
                    logger.info(f"[Agent] Empty transcript, skipping")
                    return

                logger.info(f"[Agent] Transcript: '{transcript}'")

                # 2. Check for stop command
                text_lower = transcript.lower()
                if any(re.search(p, text_lower) for p in self.stop_patterns):
                    logger.info(f"Stop command: '{transcript}'")
                    self._deactivate()
                    return

                # 3. Check wake word / active state
                has_wake_word = any(
                    re.search(p, text_lower) for p in self.wake_patterns
                )

                if has_wake_word:
                    self._activate()
                elif not self.is_active:
                    logger.info(f"Ignored (not active): '{transcript}'")
                    return
                else:
                    self._activate()  # Refresh timeout

                # 4. Generate response, collect ALL TTS audio, then send seamlessly
                logger.info(f"[Agent] Processing: '{transcript}'")
                self.metrics["responses"] += 1
                first_audio_time = None

                # Collect all TTS chunks first for seamless playback
                all_audio_chunks = []

                async for audio_chunk in self.pipeline.process_streaming(transcript):
                    if first_audio_time is None:
                        first_audio_time = time.time()
                        logger.info(f"[Agent] First audio chunk ready ({(first_audio_time - start_time)*1000:.0f}ms)")
                    all_audio_chunks.append(audio_chunk)

                # Concatenate all audio for seamless playback
                if all_audio_chunks:
                    full_audio = b''.join(all_audio_chunks)
                    total_duration = len(full_audio) / (16000 * 2)
                    logger.info(f"[Agent] Sending {total_duration:.1f}s audio seamlessly ({len(all_audio_chunks)} chunks joined)")

                    # Set speaking flag BEFORE sending (echo suppression)
                    self._is_speaking = True
                    self._speaking_done_time = 0.0

                    # Send all audio as one continuous stream with 100ms chunks
                    await self.bridge.send_audio(full_audio, chunk_duration_ms=100)

                    # Dynamic echo guard: audio plays in real-time in the meeting,
                    # so we need to mute for at least the audio duration + 2s buffer.
                    # We sent the data instantly, but MeetStream plays it over total_duration.
                    self._is_speaking = False
                    self._echo_guard_sec = total_duration + 2.0
                    self._speaking_done_time = time.time()
                    logger.info(f"[Agent] Done speaking, echo guard active for {self._echo_guard_sec:.1f}s (audio={total_duration:.1f}s)")

                # Track latency
                if first_audio_time:
                    latency_ms = (first_audio_time - start_time) * 1000
                    self.metrics["latency_samples"].append(latency_ms)
                    avg = sum(self.metrics["latency_samples"]) / len(
                        self.metrics["latency_samples"]
                    )
                    logger.info(
                        f"TTFA: {latency_ms:.0f}ms (avg: {avg:.0f}ms)"
                    )

                self.last_response_time = time.time()

            except asyncio.CancelledError:
                logger.info("Processing cancelled")
            except Exception as e:
                logger.error(f"Error processing utterance: {e}", exc_info=True)
            finally:
                # Always clear speaking flag on any exit path
                self._is_speaking = False

    async def _respond_to_text(self, text: str):
        """Generate response from text (MeetStream transcript path)."""
        if self._processing_lock.locked():
            return

        async with self._processing_lock:
            try:
                self.metrics["responses"] += 1
                logger.info(f"[Agent] Responding to text: '{text}'")

                all_audio_chunks = []
                async for audio_chunk in self.pipeline.process_streaming(text):
                    all_audio_chunks.append(audio_chunk)

                if all_audio_chunks:
                    full_audio = b''.join(all_audio_chunks)
                    total_duration = len(full_audio) / (16000 * 2)
                    self._is_speaking = True
                    self._speaking_done_time = 0.0
                    await self.bridge.send_audio(full_audio, chunk_duration_ms=100)
                    self._is_speaking = False
                    self._echo_guard_sec = total_duration + 2.0
                    self._speaking_done_time = time.time()

                self.last_response_time = time.time()

            except Exception as e:
                logger.error(f"Error responding: {e}", exc_info=True)
            finally:
                self._is_speaking = False

    # ─── Activation ───────────────────────────────────────────

    def _activate(self):
        was_inactive = not self.is_active
        self.is_active = True
        self.last_interaction_time = time.time()
        if was_inactive:
            self.metrics["wake_activations"] += 1
            logger.info("★ MOCHAN ACTIVATED ★")

    def _deactivate(self):
        if self.is_active:
            self.is_active = False
            logger.info("MOCHAN DEACTIVATED — say 'Mochan' to wake")

    async def _activity_monitor(self):
        """Deactivate after timeout. Also periodic status logging and VAD gap check."""
        status_interval = 10  # Log status every 10s
        last_status = time.time()

        while self._running:
            await asyncio.sleep(0.5)  # Check every 500ms for responsive gap detection

            # ── VAD gap timeout check ──
            # If MeetStream stops sending audio while user is mid-sentence,
            # the VAD will never see silence frames. This check flushes the
            # buffered speech after gap_timeout_sec of no new audio.
            if not self._is_speaking:  # Don't check during our own speech
                gap_utterance = self.vad.check_gap_timeout()
                if gap_utterance:
                    duration_sec = len(gap_utterance) / (16000 * 2)
                    logger.info(f"[Agent] Gap-timeout utterance: {duration_sec:.1f}s, {len(gap_utterance)} bytes")
                    asyncio.create_task(self._process_utterance(gap_utterance))

            if self.is_active:
                elapsed = time.time() - self.last_interaction_time
                if elapsed >= self.active_timeout:
                    self._deactivate()

            # Periodic status log
            now = time.time()
            if now - last_status >= status_interval:
                last_status = now
                vad_state = f"speaking={self.vad.is_speaking}, buf={len(self.vad.speech_buffer)/1024:.1f}KB"
                logger.info(
                    f"[Status] Active: {self.is_active} | "
                    f"Audio callbacks: {self.metrics['audio_callbacks']} | "
                    f"Bridge audio chunks: {self.bridge._audio_chunks_received} | "
                    f"Utterances: {self.metrics['utterances']} | "
                    f"Responses: {self.metrics['responses']} | "
                    f"Connections: {len(self.bridge._connections)} | "
                    f"VAD: {vad_state}"
                )

    # ─── Main Entry ───────────────────────────────────────────

    async def run(self, meeting_url: str, websocket_public_url: str, port: int = 8765):
        """
        Start Mochan agent and join a Google Meet.

        Args:
            meeting_url: Google Meet URL
            websocket_public_url: Public WSS URL (cloudflare tunnel)
            port: Local WebSocket server port
        """
        self._running = True

        # 1. Start socket server
        server = await self.bridge.start_socket_server(port=port)

        # 2. Create bot (it will connect to our socket server)
        try:
            await self.bridge.create_bot(
                meeting_link=meeting_url,
                socket_connection_url=websocket_public_url,
                live_audio_ws_url=websocket_public_url,
            )
        except Exception as e:
            logger.error(f"Failed to create bot: {e}")
            server.close()
            return

        logger.info("Waiting for bot to connect...")

        # 3. Wait for ready
        try:
            await asyncio.wait_for(self.bridge._ready.wait(), timeout=120)
            logger.info("Bot connected and ready!")
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for bot")
            await self.bridge.leave_meeting()
            server.close()
            return

        # 4. Start activity monitor
        monitor_task = asyncio.create_task(self._activity_monitor())

        # 5. Send greeting
        await self.bridge.send_chat_message(
            "Mochan is now active! Say 'Mochan' followed by your question."
        )

        logger.info("=" * 60)
        logger.info("MOCHAN IS LIVE!")
        logger.info(f"  Audio callbacks so far: {self.metrics['audio_callbacks']}")
        logger.info(f"  Bridge connections: {len(self.bridge._connections)}")
        logger.info("=" * 60)
        logger.info("Say 'Mochan' followed by your question")
        logger.info("Press Ctrl+C to stop")

        # 6. Run until disconnected
        try:
            while self.bridge._running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self._running = False
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            await self.bridge.leave_meeting()
            server.close()
            self._log_metrics()

    def _log_metrics(self):
        logger.info("\n" + "=" * 60)
        logger.info("Session Metrics:")
        logger.info(f"  Audio callbacks: {self.metrics['audio_callbacks']}")
        logger.info(f"  Audio chunks (bridge): {self.bridge._audio_chunks_received}")
        logger.info(f"  Audio bytes (bridge): {self.bridge._audio_bytes_received / 1024:.1f} KB")
        logger.info(f"  Utterances: {self.metrics['utterances']}")
        logger.info(f"  Responses: {self.metrics['responses']}")
        logger.info(f"  Wake activations: {self.metrics['wake_activations']}")
        if self.metrics["latency_samples"]:
            samples = self.metrics["latency_samples"]
            logger.info(f"  Avg TTFA: {sum(samples)/len(samples):.0f}ms")
            logger.info(f"  Min TTFA: {min(samples):.0f}ms")
            logger.info(f"  Max TTFA: {max(samples):.0f}ms")
        logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════
# Entry Points
# ═══════════════════════════════════════════════════════════════

async def test_connection():
    """Test MeetStream connection without full pipeline."""
    api_key = os.getenv("MEETSTREAM_API_KEY")
    if not api_key:
        logger.error("MEETSTREAM_API_KEY not set")
        return

    meeting_url = os.getenv("MEETING_URL", "")
    websocket_url = os.getenv("WEBSOCKET_PUBLIC_URL", "")

    if not meeting_url or "xxx" in meeting_url:
        logger.error("MEETING_URL not set in .env")
        return

    if not websocket_url or "your-tunnel" in websocket_url:
        logger.error("WEBSOCKET_PUBLIC_URL not set")
        logger.error("Run: cloudflared tunnel --url http://localhost:8765")
        return

    bridge = MeetStreamBridge(api_key=api_key, bot_name="Mochan")
    server = await bridge.start_socket_server(port=8765)

    try:
        await bridge.create_bot(
            meeting_link=meeting_url,
            socket_connection_url=websocket_url,
            live_audio_ws_url=websocket_url,
        )

        logger.info("Waiting for bot to connect...")
        await asyncio.wait_for(bridge._ready.wait(), timeout=120)

        logger.info("Connection test successful!")
        await bridge.send_chat_message("Hello! Mochan connection test passed.")

        # Monitor for 60 seconds to see what events come in
        for i in range(60):
            await asyncio.sleep(1)
            if i % 10 == 0:
                logger.info(
                    f"[Test] {i}s - Audio chunks: {bridge._audio_chunks_received}, "
                    f"Audio bytes: {bridge._audio_bytes_received / 1024:.1f} KB, "
                    f"Connections: {len(bridge._connections)}"
                )

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    finally:
        await bridge.leave_meeting()
        server.close()


async def main():
    """Run full Mochan agent."""
    api_key = os.getenv("MEETSTREAM_API_KEY")
    meeting_url = os.getenv("MEETING_URL", "")
    websocket_url = os.getenv("WEBSOCKET_PUBLIC_URL", "")
    port = int(os.getenv("WEBSOCKET_SERVER_PORT", "8765"))

    if not api_key:
        logger.error("MEETSTREAM_API_KEY not set!")
        logger.error("Add to .env: MEETSTREAM_API_KEY=your_key_here")
        return

    if not meeting_url or "xxx" in meeting_url:
        logger.error("MEETING_URL not set!")
        logger.error("Add to .env: MEETING_URL=https://meet.google.com/abc-defg-hij")
        return

    if not websocket_url or "your-tunnel" in websocket_url:
        logger.error("WEBSOCKET_PUBLIC_URL not set!")
        logger.error("1. Run: cloudflared tunnel --url http://localhost:8765")
        logger.error("2. Add to .env: WEBSOCKET_PUBLIC_URL=wss://xxx.trycloudflare.com")
        return

    # Ensure wss://
    if not websocket_url.startswith("wss://"):
        websocket_url = websocket_url.replace("https://", "wss://").replace(
            "http://", "wss://"
        )
        if not websocket_url.startswith("wss://"):
            websocket_url = f"wss://{websocket_url}"

    agent = MochanAgent(meetstream_api_key=api_key, bot_name="Mochan")
    await agent.run(meeting_url, websocket_url, port=port)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(test_connection())
    else:
        asyncio.run(main())
