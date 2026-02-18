# Generated from: svara_tts_kaggle.ipynb
# Converted at: 2026-02-18T06:17:58.372Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Svara-TTS — Kaggle Test Notebook
# **Model:** `kenpath/svara-tts-v1`  
# **Languages:** 19 Indian languages + Indian English  
# **Platform:** Kaggle T4 GPU  
# 
# **Important notes before running:**
# - Run cells **one by one**, top to bottom
# - Cell 4 downloads ~6GB model — wait for it fully
# - Svara uses same Orpheus architecture (Llama 3B + SNAC codec)
# - BUT unlike broken Orpheus Hindi — Svara was trained correctly with extended tokenizer
# 
# **Prompt format for Svara (different from Orpheus English):**
# ```
# Hindi (Female): आपका टेक्स्ट यहाँ। <happy>
# ```
# Voice = `Language (Gender)` — emotion tag at END of sentence


# ── CELL 1: Install dependencies ──
!pip install -q transformers torch torchaudio snac soundfile huggingface_hub

# ── CELL 2: Imports + GPU check ──
import os
import re
import torch
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM
from snac import SNAC
from huggingface_hub import login
from IPython.display import Audio, display

print("✅ Imports done!")
print(f"   PyTorch  : {torch.__version__}")
print(f"   CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU      : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── CELL 3: HuggingFace Login ──
# Get token from: https://huggingface.co/settings/tokens (READ token is enough)
# svara-tts-v1 is PUBLIC so no special access needed — login is just good practice

HF_TOKEN = "hf_xxxxxxxxxxxxxxxx"  # <── PASTE YOUR TOKEN HERE

if not HF_TOKEN.startswith("hf_") or len(HF_TOKEN) < 15:
    raise ValueError("Invalid HF token. Get one at https://huggingface.co/settings/tokens")

login(token=HF_TOKEN)
print("✅ HuggingFace login successful!")

# ── CELL 4: Load Svara model + SNAC decoder ──
# First run downloads ~6GB — subsequent runs load from cache (fast)

MODEL_NAME = "kenpath/svara-tts-v1"
SNAC_NAME  = "hubertsiuzdak/snac_24khz"

# Orpheus-style special token IDs (Svara uses same architecture)
START_OF_SPEECH   = 128257
END_OF_SPEECH     = 128258
START_OF_HUMAN    = 128259
END_OF_HUMAN      = 128260
START_OF_AI       = 128261
AUDIO_CODE_OFFSET = 128266
PAD_TOKEN         = 128263

print(f"📦 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
print(f"   Vocab size: {tokenizer.vocab_size}")

# Check vocab size — if it is ~128000 the tokenizer is the broken one
# For Svara it should be larger because they trained with extended tokenizer
if tokenizer.vocab_size <= 128000:
    print(f"   ⚠️  WARNING: Vocab size is only {tokenizer.vocab_size}")
    print(f"      This may indicate same tokenizer bug as Orpheus Hindi.")
    print(f"      We will detect this after generation and apply offset fix if needed.")
else:
    print(f"   ✅ Extended vocab confirmed — tokenizer is correct!")

print(f"\n📦 Loading model (float16)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
device = next(model.parameters()).device
print(f"   ✅ Model loaded on: {device}")

print(f"\n📦 Loading SNAC decoder...")
snac_model = SNAC.from_pretrained(SNAC_NAME).eval().to(device)
print(f"   ✅ SNAC loaded on: {device}")
print("\n🎉 All models ready!")

# ── CELL 5: Detect tokenizer type + set correct decode mode ──
# Svara may or may not have the same tokenizer bug as Orpheus Hindi
# We probe it the same way we did before

# Check one audio token to see what range we expect
# Audio tokens in a correct model start at 128266 and go to 128266 + (7*4096)
EXPECTED_MAX_AUDIO_TOKEN = 128266 + (7 * 4096)  # = 157018
print(f"Expected max audio token ID : {EXPECTED_MAX_AUDIO_TOKEN}")
print(f"Model vocab size             : {tokenizer.vocab_size}")

if tokenizer.vocab_size >= EXPECTED_MAX_AUDIO_TOKEN:
    NEEDS_OFFSET_FIX = False
    print("\n✅ Tokenizer vocab covers audio token range — standard decode mode")
else:
    NEEDS_OFFSET_FIX = True
    print(f"\n⚠️  Vocab ({tokenizer.vocab_size}) < expected ({EXPECTED_MAX_AUDIO_TOKEN})")
    print("   Will apply offset fix during decode (same fix we used for Orpheus Hindi)")

# Svara voice ID format: "Language (Gender)"
# Full list from official model card
SVARA_VOICES = {
    # Hindi
    "hindi_female" : "Hindi (Female)",
    "hindi_male"   : "Hindi (Male)",
    # Bengali
    "bengali_female": "Bengali (Female)",
    "bengali_male"  : "Bengali (Male)",
    # Marathi
    "marathi_female": "Marathi (Female)",
    "marathi_male"  : "Marathi (Male)",
    # Telugu
    "telugu_female" : "Telugu (Female)",
    "telugu_male"   : "Telugu (Male)",
    # Tamil
    "tamil_female"  : "Tamil (Female)",
    "tamil_male"    : "Tamil (Male)",
    # Kannada
    "kannada_female": "Kannada (Female)",
    "kannada_male"  : "Kannada (Male)",
    # Malayalam
    "malayalam_female": "Malayalam (Female)",
    "malayalam_male"  : "Malayalam (Male)",
    # Gujarati
    "gujarati_female": "Gujarati (Female)",
    "gujarati_male"  : "Gujarati (Male)",
    # Punjabi
    "punjabi_female": "Punjabi (Female)",
    "punjabi_male"  : "Punjabi (Male)",
    # Indian English
    "english_female": "Indian English (Female)",
    "english_male"  : "Indian English (Male)",
    # Others
    "nepali_female" : "Nepali (Female)",
    "nepali_male"   : "Nepali (Male)",
    "sanskrit_female": "Sanskrit (Female)",
    "assamese_female": "Assamese (Female)",
    "bhojpuri_female": "Bhojpuri (Female)",
    "maithili_female": "Maithili (Female)",
}

# Emotion tags — go at END of sentence (confirmed from model card)
EMOTION_TAGS = ["<happy>", "<sad>", "<anger>", "<fear>", "<neutral>"]

print(f"\n📋 Available voice shortcuts: {len(SVARA_VOICES)}")
for key, val in list(SVARA_VOICES.items())[:6]:
    print(f"   {key:20s} → '{val}'")
print(f"   ... and {len(SVARA_VOICES)-6} more")
print(f"\n🎭 Emotion tags: {EMOTION_TAGS}")

# ── CELL 6: Helper functions ──

def build_prompt(voice_id: str, text: str) -> torch.Tensor:
    """
    Build Svara prompt token sequence.
    Format: 'Voice ID: text <emotion_tag>'
    Emotion tag MUST be at end — confirmed from Svara model card.
    """
    # Resolve shortcut if used
    voice = SVARA_VOICES.get(voice_id, voice_id)
    prompt_text = f"{voice}: {text}"
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    tokens = [START_OF_HUMAN] + input_ids + [END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)


def generate_audio_tokens(
    text: str,
    voice_id: str,
    max_new_tokens: int = 1200,
    temperature: float = 0.6,
    top_p: float = 0.8,
    repetition_penalty: float = 1.3,
) -> torch.Tensor:
    """Run model inference, return raw audio token IDs."""
    input_ids = build_prompt(voice_id, text).to(device)
    print(f"   Prompt tokens : {input_ids.shape[1]}")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            eos_token_id=END_OF_SPEECH,
            pad_token_id=PAD_TOKEN,
        )

    generated = output[0, input_ids.shape[1]:]
    audio_tokens = generated[generated >= AUDIO_CODE_OFFSET] - AUDIO_CODE_OFFSET
    print(f"   Generated     : {len(generated)} tokens")
    print(f"   Audio tokens  : {len(audio_tokens)}")
    if len(audio_tokens) > 0:
        print(f"   Max token val : {audio_tokens.max().item()} (SNAC limit: 4095)")
    return audio_tokens


def decode_to_waveform(audio_tokens: torch.Tensor) -> np.ndarray:
    """
    Decode audio tokens to waveform.
    Applies per-position offset fix automatically if NEEDS_OFFSET_FIX is True.
    Uses correct Orpheus-style interleaving: pos0→c0, pos1→c1, pos2→c2,
    pos3→c2, pos4→c1, pos5→c2, pos6→c2
    """
    n = len(audio_tokens)
    n_frames = n // 7
    if n_frames == 0:
        print("⚠️  Too few audio tokens to decode.")
        return np.zeros(24000, dtype=np.float32)

    tokens = audio_tokens[:n_frames * 7].cpu().tolist()
    layer_1, layer_2, layer_3 = [], [], []

    # Offsets only applied if tokenizer is broken (same fix as Orpheus Hindi)
    offsets = [0, 4096, 8192, 12288, 16384, 20480, 24576] if NEEDS_OFFSET_FIX else [0]*7

    for i in range(n_frames):
        base = i * 7
        # Correct interleaving order — discovered from Orpheus codebase
        layer_1.append(tokens[base + 0] - offsets[0])
        layer_2.append(tokens[base + 1] - offsets[1])
        layer_3.append(tokens[base + 2] - offsets[2])
        layer_3.append(tokens[base + 3] - offsets[3])
        layer_2.append(tokens[base + 4] - offsets[4])
        layer_3.append(tokens[base + 5] - offsets[5])
        layer_3.append(tokens[base + 6] - offsets[6])

    c0 = torch.tensor(layer_1, dtype=torch.long).unsqueeze(0).to(device).clamp(0, 4095)
    c1 = torch.tensor(layer_2, dtype=torch.long).unsqueeze(0).to(device).clamp(0, 4095)
    c2 = torch.tensor(layer_3, dtype=torch.long).unsqueeze(0).to(device).clamp(0, 4095)

    with torch.no_grad():
        audio = snac_model.decode([c0, c1, c2])

    return audio.squeeze().cpu().float().numpy()


def save_and_play(waveform: np.ndarray, filename: str, sample_rate: int = 24000):
    """Normalize, save WAV, play inline."""
    if waveform.max() > 0:
        waveform = waveform / np.abs(waveform).max() * 0.95
    waveform_int16 = (waveform * 32767).astype(np.int16)
    sf.write(filename, waveform_int16, sample_rate, subtype="PCM_16")
    duration = len(waveform) / sample_rate
    print(f"   💾 Saved: {filename}  ({duration:.2f}s)")
    display(Audio(filename))


def tts(text: str, voice_id: str = "hindi_female", filename: str = "output.wav", **kwargs):
    """
    Full pipeline: text + voice → WAV file + inline playback.
    voice_id: use shortcut (e.g. 'hindi_female') or full ID (e.g. 'Hindi (Female)')
    Emotion tag tip: add at END of text e.g. 'नमस्ते! <happy>'
    """
    voice = SVARA_VOICES.get(voice_id, voice_id)
    print(f"\n🎤 voice='{voice}'")
    print(f"   text='{text[:80]}{'...' if len(text)>80 else ''}'")

    tokens = generate_audio_tokens(text, voice_id, **kwargs)
    if len(tokens) < 7:
        print("❌ Not enough audio tokens — check voice ID or text.")
        return

    waveform = decode_to_waveform(tokens)
    save_and_play(waveform, filename)


def tts_long(text: str, voice_id: str = "hindi_female", filename: str = "output_long.wav", **kwargs):
    """
    For long texts — splits on sentence boundaries and joins audio.
    Use this for paragraphs, not short sentences.
    """
    # Split on Hindi (।) and standard punctuation
    sentences = re.split(r'(?<=[।!?\.]\s)|(?<=[।!?\.]$)', text.strip())
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]

    print(f"📝 Split into {len(sentences)} sentences")
    for i, s in enumerate(sentences):
        print(f"   {i+1}: {s[:60]}{'...' if len(s)>60 else ''}")

    all_waveforms = []
    silence_200ms = np.zeros(int(24000 * 0.2), dtype=np.float32)

    for i, sentence in enumerate(sentences):
        print(f"\n--- Sentence {i+1}/{len(sentences)} ---")
        tokens = generate_audio_tokens(sentence, voice_id, **kwargs)
        if len(tokens) < 7:
            print("⚠️  Skipping sentence — too few tokens")
            continue
        waveform = decode_to_waveform(tokens)
        all_waveforms.append(waveform)
        all_waveforms.append(silence_200ms)

    if not all_waveforms:
        print("❌ No audio generated")
        return

    full_waveform = np.concatenate(all_waveforms)
    save_and_play(full_waveform, filename)


print("✅ All helper functions ready!")
print("\nUsage:")
print("  tts('नमस्ते! <happy>', voice_id='hindi_female')")
print("  tts('Hello there!', voice_id='english_female')")
print("  tts_long('long paragraph...', voice_id='hindi_male')")

# ── CELL 7: Test 1 — Hindi Female, Neutral ──
tts(
    text="नमस्ते! मेरा नाम स्वरा है और मैं हिंदी में बोल सकती हूँ।",
    voice_id="hindi_female",
    filename="test1_hindi_female.wav"
)

# ── CELL 8: Test 2 — Hindi Female, Happy emotion ──
# Emotion tag goes at END of sentence — this is how Svara was trained
tts(
    text="आज का दिन बहुत खास है, सच में बहुत अच्छा लग रहा है! <happy>",
    voice_id="hindi_female",
    filename="test2_hindi_happy.wav"
)

# ── CELL 9: Test 3 — Hindi Male ──
tts(
    text="भारत एक विविधताओं से भरा देश है।",
    voice_id="hindi_male",
    filename="test3_hindi_male.wav"
)

# ── CELL 10: Test 4 — Indian English Female ──
tts(
    text="Hello! I am Svara, a multilingual text to speech model for India.",
    voice_id="english_female",
    filename="test4_english_female.wav"
)

# ── CELL 11: Test 5 — Emotion comparison ──
# Same sentence, different emotions — hear the difference
base_text = "मुझे नहीं पता क्या होगा"
for emotion in ["<neutral>", "<sad>", "<fear>", "<anger>"]:
    tts(
        text=f"{base_text} {emotion}",
        voice_id="hindi_female",
        filename=f"test5_emotion_{emotion.strip('<>')}.wav",
        max_new_tokens=400
    )

# ── CELL 12: Test 6 — Hinglish (code-mix) ──
tts(
    text="Yaar, aaj ka din toh bohot amazing tha, seriously!",
    voice_id="hindi_female",
    filename="test6_hinglish.wav"
)

# ── CELL 13: Test 7 — Long paragraph with tts_long ──
tts_long(
    text="धीरे-धीरे ढलता हुआ सूरज आसमान में अपनी नारंगी आभा बिखेर रहा था। ठंडी हवा के झोंके जब चेहरे को छूकर गुजरते हैं, तो ऐसा महसूस होता है जैसे प्रकृति खुद हमें याद दिला रही है। यह वक्त हमें सिखाता है कि चाहे रात कितनी भी गहरी क्यों न हो, सुबह की किरण एक नई उम्मीद लेकर जरूर आती है।",
    voice_id="hindi_female",
    filename="test7_long_paragraph.wav"
)

# ── CELL 14: Debug — check token distribution (run if audio sounds wrong) ──
# This tells us if Svara has the same tokenizer bug as Orpheus Hindi
# If max token value >> 4095, the offset fix will auto-apply

print("Running token distribution check...")
debug_tokens = generate_audio_tokens(
    "नमस्ते!",
    voice_id="hindi_female",
    max_new_tokens=150
)

if len(debug_tokens) >= 7:
    n_frames = len(debug_tokens) // 7
    t = debug_tokens[:n_frames * 7].cpu().view(n_frames, 7)
    print("\nPer-position ranges (raw token values before offset):")
    for i in range(7):
        print(f"   Position {i}: min={t[:,i].min().item():6d}  max={t[:,i].max().item():6d}")
    print(f"\nNEEDS_OFFSET_FIX = {NEEDS_OFFSET_FIX}")
    if t.max().item() <= 4095:
        print("✅ Token range is clean — standard SNAC decode")
    else:
        print(f"⚠️  Max token {t.max().item()} > 4095 — offset fix is active")
else:
    print("⚠️  Too few tokens generated for debug check")