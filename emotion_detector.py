"""
Lightweight Emotion Detector for Real-Time Voice Pipeline

Detects user emotion from speech audio and maps it to an appropriate
response emotion. Runs PARALLEL to STT with zero additional latency.

Architecture:
  User audio ──┬──> STT (Whisper/Sarvam)  ──> transcript
               │
               └──> EmotionDetector        ──> response_emotion
                    (this module, ~80ms CPU)

The detected emotion is used as a CONTROL SIGNAL for TTS, NOT as inline
text tags. This eliminates the risk of LLM-generated tag malformation.
"""

import asyncio
import logging
import time
import re
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("emotion-detector")


# ═══════════════════════════════════════════════════════════════
# Emotion Mappings
# ═══════════════════════════════════════════════════════════════

# Map detected USER emotion → appropriate RESPONSE emotion
USER_TO_RESPONSE_MAP = {
    "angry": "calm",
    "sad": "empathetic",
    "happy": "happy",
    "excited": "excited",
    "frustrated": "calm",
    "fearful": "calm",
    "surprised": "neutral",
    "disgusted": "calm",
    "neutral": "neutral",
}

# Valid emotions our TTS can render
VALID_TTS_EMOTIONS = {
    "happy", "sad", "angry", "calm",
    "empathetic", "excited", "neutral",
}

# Tone guidance for LLM system prompt
TONE_GUIDES = {
    "calm": "The user seems frustrated or upset. Be patient, reassuring, and solution-oriented.",
    "empathetic": "The user seems sad or down. Be warm, understanding, and gently supportive.",
    "happy": "The user seems happy and positive. Match their energy and be cheerful.",
    "excited": "The user is excited and enthusiastic. Be upbeat and energetic.",
    "neutral": "Respond naturally and conversationally.",
}

DEFAULT_EMOTION = "neutral"


@dataclass
class EmotionState:
    """
    Tracks emotion with smoothing to prevent flip-flopping.

    Only changes emotion when 2 of the last 3 detections agree,
    preventing jarring tone shifts on noisy single-turn detections.
    Decays to neutral after 60s of silence.
    """
    current: str = DEFAULT_EMOTION
    confidence: float = 0.0
    last_update: float = 0.0
    history: list = field(default_factory=list)
    max_history: int = 3
    decay_seconds: float = 60.0

    def update(self, detected_emotion: str, confidence: float = 0.5) -> str:
        """
        Update emotion state with new detection.
        Returns the current response emotion (may not change on every call).
        """
        self.history.append(detected_emotion)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # First detection with confidence: apply immediately
        # Subsequent: require 2 of last 3 to match (smoothing)
        if len(self.history) == 1 and confidence >= 0.3:
            self.current = detected_emotion
            self.confidence = confidence
            self.last_update = time.time()
        elif self.history.count(detected_emotion) >= 2:
            self.current = detected_emotion
            self.confidence = confidence
            self.last_update = time.time()

        return self.get_response_emotion()

    def get_response_emotion(self) -> str:
        """Get the mapped response emotion, with decay."""
        # Decay to neutral after silence
        if self.last_update > 0 and (time.time() - self.last_update) > self.decay_seconds:
            return DEFAULT_EMOTION

        return USER_TO_RESPONSE_MAP.get(self.current, DEFAULT_EMOTION)

    def reset(self):
        """Reset emotion state."""
        self.current = DEFAULT_EMOTION
        self.confidence = 0.0
        self.last_update = 0.0
        self.history.clear()


class TextEmotionDetector:
    """
    Text-based emotion detection using keyword heuristics.

    This is the MVP approach — no ML model needed, runs in <1ms.
    Detects emotion from the TRANSCRIPT text as a lightweight fallback
    when audio-based detection is not available.

    For production, replace with wav2vec2-based audio classifier.
    """

    # Hindi/Hinglish/English emotion keywords
    EMOTION_KEYWORDS = {
        "angry": [
            "gussa", "angry", "naraz", "irritate", "pagal", "bakwas",
            "kya bakwas", "nonsense", "stupid", "idiot", "bewakoof",
            "chup", "shut up", "hate", "nafrat",
        ],
        "sad": [
            "sad", "dukhi", "udaas", "cry", "ro", "miss", "yaad",
            "bura", "kharab", "problem", "mushkil", "pareshan",
            "help me", "madad", "sorry", "maaf",
        ],
        "happy": [
            "khushi", "happy", "great", "awesome", "amazing", "maza",
            "fantastic", "wonderful", "accha", "bahut accha", "sahi",
            "perfect", "love", "pyaar", "dhanyavaad", "thanks",
            "thank you", "shukriya",
        ],
        "excited": [
            "wow", "yay", "excited", "josh", "awesome", "incredible",
            "unbelievable", "kamaal", "zabardast", "oh my god",
            "omg", "seriously", "are you kidding",
        ],
        "frustrated": [
            "nahi ho raha", "not working", "kaam nahi", "error",
            "problem", "issue", "bug", "fix", "why", "kyu", "kyun",
            "phir se", "again", "still", "abhi bhi", "samajh nahi",
            "confused", "frustrat",
        ],
        "fearful": [
            "dar", "scared", "afraid", "fear", "worry", "tension",
            "kya hoga", "what will happen", "risky", "dangerous",
        ],
    }

    def detect_from_text(self, text: str) -> tuple[str, float]:
        """
        Detect emotion from text using keyword matching.

        Args:
            text: Transcript text (Hindi/Hinglish/English)

        Returns:
            Tuple of (emotion, confidence)
        """
        if not text:
            return DEFAULT_EMOTION, 0.0

        text_lower = text.lower().strip()

        # Count keyword matches per emotion
        scores = {}
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > 0:
                scores[emotion] = count

        if not scores:
            return DEFAULT_EMOTION, 0.0

        # Return highest scoring emotion
        best_emotion = max(scores, key=scores.get)
        # Confidence: clamp between 0.3 (1 match) and 0.8 (3+ matches)
        confidence = min(0.3 + (scores[best_emotion] - 1) * 0.25, 0.8)

        return best_emotion, confidence


class EmotionDetector:
    """
    Main emotion detector that combines text-based detection
    with optional audio-based detection.

    MVP uses text-only. Audio classifier can be added later by
    implementing detect_from_audio().
    """

    def __init__(self, use_audio: bool = False):
        """
        Args:
            use_audio: Whether to use audio-based emotion detection.
                       Requires transformers + wav2vec2 model (~300MB).
                       Default False for MVP (text-only).
        """
        self.text_detector = TextEmotionDetector()
        self.state = EmotionState()
        self.use_audio = use_audio
        self._audio_classifier = None

        if use_audio:
            self._init_audio_classifier()

        logger.info(f"EmotionDetector initialized (audio={use_audio})")

    def _init_audio_classifier(self):
        """Initialize wav2vec2 audio emotion classifier (optional)."""
        try:
            from transformers import pipeline as hf_pipeline
            self._audio_classifier = hf_pipeline(
                "audio-classification",
                model="superb/wav2vec2-base-superb-er",
                device="cpu",  # CPU is fine — model is tiny (29MB)
            )
            logger.info("Audio emotion classifier loaded (wav2vec2)")
        except Exception as e:
            logger.warning(f"Could not load audio classifier: {e}")
            logger.warning("Falling back to text-only emotion detection")
            self.use_audio = False

    async def detect_from_audio(self, audio_pcm: bytes) -> tuple[str, float]:
        """
        Detect emotion from audio using wav2vec2 classifier.

        Args:
            audio_pcm: Raw PCM audio bytes (16-bit, mono, 16kHz)

        Returns:
            Tuple of (emotion, confidence)
        """
        if not self._audio_classifier:
            return DEFAULT_EMOTION, 0.0

        try:
            import numpy as np

            # Convert PCM bytes to float32 array
            audio_np = np.frombuffer(audio_pcm, dtype=np.int16).astype(np.float32) / 32768.0

            # Run classifier in thread pool (CPU-bound, don't block async loop)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._audio_classifier(audio_np, sampling_rate=16000, top_k=1)
            )

            if result:
                label = result[0]["label"].lower()
                score = result[0]["score"]

                # Map wav2vec2 labels to our emotion set
                label_map = {
                    "ang": "angry",
                    "hap": "happy",
                    "sad": "sad",
                    "neu": "neutral",
                    "exc": "excited",
                    "fru": "frustrated",
                    "fea": "fearful",
                    "dis": "disgusted",
                    "sur": "surprised",
                }
                emotion = label_map.get(label, DEFAULT_EMOTION)
                return emotion, score

            return DEFAULT_EMOTION, 0.0

        except Exception as e:
            logger.error(f"Audio emotion detection error: {e}")
            return DEFAULT_EMOTION, 0.0

    async def detect(
        self,
        transcript: str,
        audio_pcm: Optional[bytes] = None,
    ) -> str:
        """
        Detect emotion and return the appropriate RESPONSE emotion.

        This is the main entry point. It combines text and optional
        audio signals, updates the smoothed state, and returns the
        emotion tag that should be applied to TTS output.

        Args:
            transcript: The user's transcribed text
            audio_pcm: Optional raw audio for audio-based detection

        Returns:
            Response emotion string (e.g., "calm", "happy", "empathetic")
        """
        start = time.time()

        # Text-based detection (always available, <1ms)
        text_emotion, text_confidence = self.text_detector.detect_from_text(transcript)

        # Audio-based detection (optional, ~80ms on CPU)
        audio_emotion = DEFAULT_EMOTION
        audio_confidence = 0.0

        if self.use_audio and audio_pcm:
            audio_emotion, audio_confidence = await self.detect_from_audio(audio_pcm)

        # Combine signals: prefer audio if confident, else use text
        if audio_confidence > 0.6:
            final_emotion = audio_emotion
            final_confidence = audio_confidence
        elif text_confidence >= 0.3:
            final_emotion = text_emotion
            final_confidence = text_confidence
        else:
            final_emotion = DEFAULT_EMOTION
            final_confidence = 0.0

        # Update smoothed state
        response_emotion = self.state.update(final_emotion, final_confidence)

        elapsed_ms = (time.time() - start) * 1000
        if final_emotion != DEFAULT_EMOTION:
            logger.info(
                f"Emotion: user={final_emotion}({final_confidence:.2f}) "
                f"-> response={response_emotion} [{elapsed_ms:.1f}ms]"
            )

        return response_emotion

    def get_tone_guide(self, response_emotion: str) -> str:
        """Get tone guidance string for LLM system prompt."""
        return TONE_GUIDES.get(response_emotion, TONE_GUIDES["neutral"])

    def get_current_emotion(self) -> str:
        """Get current response emotion without new detection."""
        return self.state.get_response_emotion()

    def reset(self):
        """Reset emotion state."""
        self.state.reset()


def apply_emotion_tag(text: str, emotion: str) -> str:
    """
    Safely wrap text with a validated emotion tag for TTS.

    This is the ONLY place emotion tags are created in the entire pipeline.
    The LLM never generates tags. This function is deterministic.

    Args:
        text: Clean text (no tags)
        emotion: Validated emotion string

    Returns:
        Text wrapped with emotion tag, or clean text if emotion is neutral
    """
    # Strip any tags the LLM might have accidentally generated
    clean = re.sub(r'<[^>]*>', '', text).strip()

    if not clean:
        return ""

    # Validate emotion
    if emotion not in VALID_TTS_EMOTIONS:
        emotion = DEFAULT_EMOTION

    # Neutral doesn't need a tag
    if emotion == "neutral":
        return clean

    return f"<{emotion}>{clean}</{emotion}>"


def strip_all_tags(text: str) -> str:
    """Strip ALL XML-like tags from text. Safety net for LLM output."""
    return re.sub(r'<[^>]*>', '', text).strip()
