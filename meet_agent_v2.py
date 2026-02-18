"""
Optimized Google Meet Voice Agent - Jarvis V2
Uses Sarvam STT/TTS + Groq LLM with reduced VAD for low latency.

Target: ~800ms end-to-end latency
"""

import asyncio
import logging
import os
import re
import sys
import time
from typing import Optional
from enum import Enum
from dataclasses import dataclass

from dotenv import load_dotenv
import numpy as np

# Import from local modules
from sarvam_streaming import SarvamStreamingSTT, SarvamRESTClient
from optimized_pipeline import OptimizedPipeline

load_dotenv(".env.local")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("jarvis-v2")

# Silence noisy loggers
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class AgentState(Enum):
    """Conversation state machine states."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


@dataclass
class VADConfig:
    """VAD configuration for reduced latency."""
    min_speech_duration: float = 0.1      # 100ms - catch speech quickly
    min_silence_duration: float = 0.5     # 500ms - reduced from 1500ms!
    prefix_padding_duration: float = 0.3  # 300ms - capture first words
    max_buffered_speech: float = 30.0     # 30s max utterance
    activation_threshold: float = 0.4     # Sensitivity
    sample_rate: int = 16000


class SimpleVAD:
    """
    Simple energy-based VAD for reduced latency.
    Silero VAD adds overhead - this is faster for real-time use.
    """
    
    def __init__(self, config: VADConfig):
        self.config = config
        self.is_speaking = False
        self.speech_buffer = bytearray()
        self.silence_samples = 0
        self.speech_samples = 0
        
        # Thresholds
        self.energy_threshold = 500  # Adjust based on testing
        self.samples_per_chunk = int(config.sample_rate * 0.02)  # 20ms chunks
        self.min_speech_samples = int(config.min_speech_duration * config.sample_rate)
        self.min_silence_samples = int(config.min_silence_duration * config.sample_rate)
        
        # Callbacks
        self.on_speech_start = None
        self.on_speech_end = None
    
    def process(self, audio_chunk: bytes) -> Optional[bytes]:
        """
        Process audio chunk and detect speech boundaries.
        
        Args:
            audio_chunk: Raw PCM audio (16-bit, mono)
            
        Returns:
            Complete utterance if speech ended, None otherwise
        """
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
        is_speech = energy > self.energy_threshold
        
        if is_speech:
            self.silence_samples = 0
            self.speech_samples += len(audio_array)
            
            if not self.is_speaking and self.speech_samples >= self.min_speech_samples:
                self.is_speaking = True
                logger.info("🎤 Speech started")
                if self.on_speech_start:
                    self.on_speech_start()
            
            if self.is_speaking:
                self.speech_buffer.extend(audio_chunk)
        else:
            if self.is_speaking:
                self.silence_samples += len(audio_array)
                self.speech_buffer.extend(audio_chunk)  # Include trailing silence
                
                if self.silence_samples >= self.min_silence_samples:
                    # Speech ended
                    self.is_speaking = False
                    utterance = bytes(self.speech_buffer)
                    self.speech_buffer = bytearray()
                    self.speech_samples = 0
                    self.silence_samples = 0
                    
                    duration_sec = len(utterance) / (self.config.sample_rate * 2)
                    logger.info(f"🔇 Speech ended ({duration_sec:.1f}s)")
                    
                    if self.on_speech_end:
                        self.on_speech_end()
                    
                    return utterance
            else:
                self.speech_samples = 0
        
        return None
    
    def reset(self):
        """Reset VAD state."""
        self.is_speaking = False
        self.speech_buffer = bytearray()
        self.silence_samples = 0
        self.speech_samples = 0


class JarvisV2:
    """
    Optimized Jarvis voice agent for Google Meet.
    
    Architecture:
    Google Meet ←→ Attendee.dev ←→ WebSocket ←→ This Agent
                                                    ↓
                                         SimpleVAD (500ms silence)
                                                    ↓
                                         Sarvam STT (parallel possible)
                                                    ↓
                                         Groq LLM (streaming)
                                                    ↓
                                         Sarvam TTS (streaming)
                                                    ↓
                                         Audio back to Meet
    """
    
    def __init__(self):
        """Initialize Jarvis V2."""
        self.sample_rate = 16000
        self.state = AgentState.IDLE
        
        # Pipeline
        self.pipeline = OptimizedPipeline()
        
        # VAD with reduced timeout
        vad_config = VADConfig(
            min_silence_duration=0.5,  # 500ms - key optimization!
            min_speech_duration=0.1,
            prefix_padding_duration=0.3
        )
        self.vad = SimpleVAD(vad_config)
        
        # Wake word patterns - "Mochan" variations
        self.jarvis_patterns = [
            r"mochan",
            r"mo[\s\-]?chan",      # mo-chan, mo chan
            r"mochan[\s\-]?d",     # mochan d, mochan-d
            r"mochandas",
            r"मोचन",               # Hindi
            r"मोचनदास",            # Hindi full
            r"मो\s*चन",            # Hindi with space
            r"mohan",              # Common mishearing
            r"मोहन",               # Hindi mohan
        ]
        
        # Stop patterns
        self.stop_patterns = [
            r"stop\s+mochan",
            r"mochan\s+stop",
            r"band\s+karo",
            r"ruk\s+jao",
            r"stop",
            r"bas",
        ]
        
        # Active listening mode
        self.is_active = False
        self.active_timeout = 60  # seconds
        self.last_interaction_time = 0.0
        
        # Bridge and session (set externally)
        self.bridge = None
        self.meet_session = None
        
        # Task management
        self.is_running = False
        self.current_task: Optional[asyncio.Task] = None
        self.activity_monitor_task: Optional[asyncio.Task] = None
        
        # Parallel STT (experimental)
        self.parallel_stt: Optional[SarvamStreamingSTT] = None
        self.parallel_transcript = ""
        
        # Metrics
        self.metrics = {
            "utterances": 0,
            "responses": 0,
            "interruptions": 0,
            "wake_activations": 0,
            "avg_latency_ms": 0,
            "latency_samples": []
        }
        
        logger.info("🤖 Jarvis V2 initialized")
        logger.info(f"   VAD silence timeout: {vad_config.min_silence_duration}s")
    
    def _activate(self):
        """Activate Jarvis for listening."""
        was_inactive = not self.is_active
        self.is_active = True
        self.last_interaction_time = time.time()
        
        if was_inactive:
            self.metrics["wake_activations"] += 1
            logger.info(f"🟢 JARVIS ACTIVATED - Listening for {self.active_timeout}s")
    
    def _deactivate(self):
        """Deactivate Jarvis."""
        if self.is_active:
            self.is_active = False
            logger.info("🔴 MOCHAN DEACTIVATED - Say 'Mochan' to wake")
    
    async def _activity_monitor(self):
        """Monitor activity and deactivate after timeout."""
        while self.is_running:
            await asyncio.sleep(1.0)
            
            if self.is_active:
                elapsed = time.time() - self.last_interaction_time
                if elapsed >= self.active_timeout:
                    self._deactivate()
    
    async def on_audio_received(self, pcm_data: bytes, sample_rate: int):
        """
        Called when audio is received from Google Meet.
        
        Args:
            pcm_data: Raw PCM audio (16-bit, mono)
            sample_rate: Sample rate (should be 16000)
        """
        if not self.is_running:
            return
        
        # Process through VAD
        utterance = self.vad.process(pcm_data)
        
        if utterance:
            # Complete utterance detected
            await self._process_utterance(utterance)
    
    async def _process_utterance(self, audio_pcm: bytes):
        """Process a complete utterance."""
        duration_sec = len(audio_pcm) / (self.sample_rate * 2)
        
        # Filter noise (min 200ms)
        if duration_sec < 0.2:
            logger.debug(f"Ignoring short audio ({duration_sec:.2f}s)")
            return
        
        self.metrics["utterances"] += 1
        self.state = AgentState.PROCESSING
        
        try:
            # 1. Transcribe
            start_time = time.time()
            transcript = await self.pipeline.transcribe_audio(audio_pcm)
            
            if not transcript or len(transcript.strip()) < 2:
                self.state = AgentState.IDLE
                return
            
            # 2. Check for stop command
            if any(re.search(p, transcript, re.IGNORECASE) for p in self.stop_patterns):
                logger.info(f"🛑 Stop command: '{transcript}'")
                self._deactivate()
                self.state = AgentState.IDLE
                return
            
            # 3. Check for wake word
            has_wake_word = any(
                re.search(p, transcript, re.IGNORECASE) 
                for p in self.jarvis_patterns
            )
            
            should_respond = has_wake_word or self.is_active
            
            if has_wake_word:
                self._activate()
            elif self.is_active:
                self._activate()  # Refresh timeout
            
            if not should_respond:
                logger.info(f"💤 Ignored (not active): '{transcript}'")
                self.state = AgentState.IDLE
                return
            
            # 4. Generate and stream response
            logger.info(f"🎯 Processing: '{transcript}'")
            self.state = AgentState.SPEAKING
            self.metrics["responses"] += 1
            
            chunks_sent = 0
            first_audio_time = None
            
            async for audio_chunk in self.pipeline.process_streaming(transcript):
                if chunks_sent == 0:
                    first_audio_time = time.time()
                
                # Send audio to Meet
                await self._send_audio(audio_chunk)
                chunks_sent += 1
            
            # Track latency
            if first_audio_time:
                latency_ms = (first_audio_time - start_time) * 1000
                self.metrics["latency_samples"].append(latency_ms)
                self.metrics["avg_latency_ms"] = sum(self.metrics["latency_samples"]) / len(self.metrics["latency_samples"])
                logger.info(f"⚡ Time to first audio: {latency_ms:.0f}ms (avg: {self.metrics['avg_latency_ms']:.0f}ms)")
            
            self._activate()  # Refresh timeout after response
            
        except asyncio.CancelledError:
            logger.info("⏹️ Utterance processing cancelled")
        except Exception as e:
            logger.error(f"Error processing utterance: {e}", exc_info=True)
        finally:
            self.state = AgentState.IDLE
    
    async def _send_audio(self, audio_chunk: bytes):
        """Send audio chunk to Google Meet via Attendee bridge."""
        if self.bridge:
            await self.bridge.send_audio(audio_chunk, self.sample_rate)
        else:
            logger.warning("No bridge available to send audio")
    
    async def start(self, bridge, meet_session):
        """
        Start the agent.
        
        Args:
            bridge: AttendeeBridge instance
            meet_session: MeetSession instance
        """
        self.bridge = bridge
        self.meet_session = meet_session
        self.is_running = True
        
        # Start activity monitor
        self.activity_monitor_task = asyncio.create_task(self._activity_monitor())
        
        logger.info("🚀 Mochan started")
        logger.info(f"   Say 'Mochan' to activate")
        logger.info(f"   Active timeout: {self.active_timeout}s")
    
    async def stop(self):
        """Stop the agent."""
        self.is_running = False
        
        if self.activity_monitor_task:
            self.activity_monitor_task.cancel()
            try:
                await self.activity_monitor_task
            except asyncio.CancelledError:
                pass
        
        if self.current_task:
            self.current_task.cancel()
            try:
                await self.current_task
            except asyncio.CancelledError:
                pass
        
        self.vad.reset()
        
        # Log metrics
        logger.info("\n📊 Session Metrics:")
        logger.info(f"   Utterances processed: {self.metrics['utterances']}")
        logger.info(f"   Responses generated: {self.metrics['responses']}")
        logger.info(f"   Wake activations: {self.metrics['wake_activations']}")
        if self.metrics['latency_samples']:
            logger.info(f"   Avg latency: {self.metrics['avg_latency_ms']:.0f}ms")
            logger.info(f"   Min latency: {min(self.metrics['latency_samples']):.0f}ms")
            logger.info(f"   Max latency: {max(self.metrics['latency_samples']):.0f}ms")
        
        logger.info("🛑 Jarvis V2 stopped")


# =============================================================================
# Main Entry Point with Attendee Integration
# =============================================================================

async def main():
    """Main entry point with Google Meet integration."""
    import websockets
    import json
    import base64
    import aiohttp
    
    if len(sys.argv) < 2:
        print("\n" + "="*60)
        print("Jarvis V2 - Optimized Google Meet Voice Agent")
        print("="*60)
        print("\nUsage:")
        print("  python meet_agent_v2.py <google_meet_url>")
        print("\nExample:")
        print("  python meet_agent_v2.py https://meet.google.com/abc-defg-hij")
        print("\nPrerequisites:")
        print("  1. Set SARVAM_API_KEY in .env.local")
        print("  2. Set GROQ_API_KEY in .env.local")
        print("  3. Set ATTENDEE_API_KEY in .env.local")
        print("  4. Run cloudflared tunnel or ngrok")
        print("  5. Set WEBSOCKET_PUBLIC_URL in .env.local")
        print("="*60 + "\n")
        sys.exit(1)
    
    meeting_url = sys.argv[1]
    
    # Load environment
    api_key = os.getenv("ATTENDEE_API_KEY")
    ws_public_url = os.getenv("WEBSOCKET_PUBLIC_URL")
    ws_port = int(os.getenv("WEBSOCKET_SERVER_PORT", "8765"))
    
    if not api_key:
        print("❌ ATTENDEE_API_KEY not found")
        sys.exit(1)
    
    if not ws_public_url:
        print("❌ WEBSOCKET_PUBLIC_URL not found")
        print("   Run: cloudflared tunnel --url http://localhost:8765")
        print("   Then set WEBSOCKET_PUBLIC_URL to the tunnel URL")
        sys.exit(1)
    
    # Ensure wss://
    if not ws_public_url.startswith("wss://"):
        ws_public_url = ws_public_url.replace("https://", "wss://").replace("http://", "wss://")
        if not ws_public_url.startswith("wss://"):
            ws_public_url = f"wss://{ws_public_url}"
    
    # Initialize agent
    agent = JarvisV2()
    
    # WebSocket handler for Attendee
    async def handle_attendee_ws(websocket):
        """Handle WebSocket connection from Attendee bot."""
        logger.info(f"📥 Attendee bot connected")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    trigger = data.get("trigger", "")
                    
                    if trigger == "realtime_audio.mixed":
                        # Audio from meeting
                        audio_b64 = data["data"]["chunk"]
                        sample_rate = data["data"].get("sample_rate", 16000)
                        pcm_bytes = base64.b64decode(audio_b64)
                        
                        # Process through agent
                        await agent.on_audio_received(pcm_bytes, sample_rate)
                    
                    elif trigger == "bot.joined":
                        logger.info("✅ Bot joined meeting")
                    
                    elif trigger == "bot.left":
                        logger.warning("⚠️ Bot left meeting")
                        break
                        
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("📤 Attendee bot disconnected")
    
    # Simple bridge for sending audio back - WITH PROPER CHUNKING
    class SimpleBridge:
        def __init__(self):
            self.ws = None
            self.sample_rate = 16000
        
        async def send_audio(self, pcm_data: bytes, sample_rate: int = 16000, chunk_duration_ms: int = 20):
            """
            Send audio to Attendee in small chunks for smooth playback.
            
            Args:
                pcm_data: Raw PCM audio bytes (16-bit, mono)
                sample_rate: Sample rate (must be 16000 for Attendee)
                chunk_duration_ms: Chunk size in ms (20ms recommended)
            """
            if not self.ws:
                logger.warning("No WebSocket connection")
                return
            
            try:
                # Calculate chunk size (16-bit = 2 bytes per sample)
                chunk_size = int((chunk_duration_ms / 1000.0) * sample_rate * 2)
                
                total_bytes = len(pcm_data)
                chunks_sent = 0
                
                for i in range(0, total_bytes, chunk_size):
                    chunk = pcm_data[i:i + chunk_size]
                    
                    if len(chunk) == 0:
                        continue
                    
                    # Create Attendee message
                    payload = {
                        "trigger": "realtime_audio.bot_output",
                        "data": {
                            "chunk": base64.b64encode(chunk).decode('utf-8'),
                            "sample_rate": sample_rate
                        }
                    }
                    
                    await self.ws.send(json.dumps(payload))
                    chunks_sent += 1
                    
                    # Small delay for real-time playback (95% of chunk duration)
                
                duration_sec = total_bytes / (sample_rate * 2)
                logger.debug(f"📤 Sent {chunks_sent} chunks ({duration_sec:.1f}s audio)")
                
            except Exception as e:
                logger.error(f"Error sending audio: {e}")
    
    bridge = SimpleBridge()
    
    # Modified handler to share websocket with bridge
    async def handle_ws_with_bridge(websocket):
        bridge.ws = websocket
        await handle_attendee_ws(websocket)
    
    try:
        # Start WebSocket server
        logger.info(f"\n{'='*60}")
        logger.info("Starting Jarvis V2")
        logger.info(f"{'='*60}\n")
        
        server = await websockets.serve(
            handle_ws_with_bridge,
            "0.0.0.0",
            ws_port,
            ping_interval=20,
            ping_timeout=10
        )
        
        logger.info(f"✓ WebSocket server running on port {ws_port}")
        logger.info(f"✓ Public URL: {ws_public_url}")
        
        # Create Attendee bot
        logger.info(f"\nCreating bot for meeting: {meeting_url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://app.attendee.dev/api/v1/bots",
                headers={
                    "Authorization": f"Token {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "meeting_url": meeting_url,
                    "bot_name": "Mochan",
                    "websocket_settings": {
                        "audio": {
                            "url": ws_public_url,
                            "sample_rate": 16000
                        }
                    }
                }
            ) as response:
                if response.status not in [200, 201]:
                    error = await response.text()
                    logger.error(f"Failed to create bot: {error}")
                    sys.exit(1)
                
                result = await response.json()
                bot_id = result.get("id")
                logger.info(f"✓ Bot created: {bot_id}")
        
        # Start agent
        await agent.start(bridge, None)
        
        logger.info(f"\n{'='*60}")
        logger.info("🎤 Mochan is LIVE!")
        logger.info(f"{'='*60}")
        logger.info("\n💡 Say 'Mochan' followed by your question")
        logger.info("💡 Example: 'Mochan, aaj ka weather kaisa hai?'")
        logger.info("\nPress Ctrl+C to stop\n")
        
        # Run forever
        await asyncio.Future()
        
    except KeyboardInterrupt:
        logger.info("\n\n⏹️ Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        await agent.stop()
        logger.info("✓ Cleanup complete")


if __name__ == "__main__":
    asyncio.run(main())
