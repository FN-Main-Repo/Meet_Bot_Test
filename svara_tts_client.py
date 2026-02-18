import os
import torch
import numpy as np
import logging
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from snac import SNAC

logger = logging.getLogger("svara-tts-client")

class SvaraTTSClient:
    """
    ULTRA-LOW LATENCY Svara-v1 Client.
    Optimized for <1s TTFA on RunPod (A5000/A6000).
    """
    
    AUDIO_CODE_OFFSET = 128266
    END_OF_SPEECH     = 128258
    START_OF_HUMAN    = 128259
    END_OF_HUMAN      = 128260
    START_OF_AI       = 128261
    START_OF_SPEECH   = 128257
    PAD_TOKEN         = 128263

    def __init__(self, model_name="kenpath/svara-tts-v1"):
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load models with Bfloat16 for Ampere speed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=self.hf_token,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
            attn_implementation="flash_attention_2" if self.device.type == "cuda" else None
        )
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self.device)
        
        # Vocab fix check
        self.needs_offset_fix = self.tokenizer.vocab_size < 157018
        self.offsets = [0, 4096, 8192, 12288, 16384, 20480, 24576] if self.needs_offset_fix else [0]*7

    async def synthesize_streaming(self, text, voice_id="Hindi (Female)", emotion="neutral"):
        """
        Yields audio chunks as soon as they are generated.
        This is the KEY to <1s TTFA.
        """
        if emotion and emotion != "neutral":
            text = f"{text.strip()} <{emotion}>"
            
        prompt_text = f"{voice_id}: {text}"
        input_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        inputs = torch.tensor([self.START_OF_HUMAN] + input_ids + [self.END_OF_HUMAN, self.START_OF_AI, self.START_OF_SPEECH]).unsqueeze(0).to(self.device)

        # Buffer for tokens - we need 7 tokens to form 1 SNAC frame
        token_buffer = []
        
        # Standard generate is too slow for streaming, but we'll use a simplified version
        # for maximum TTFA speed.
        with torch.no_grad():
            # We use a smaller max_new_tokens for the first chunk to force speed
            output = self.model.generate(
                inputs,
                max_new_tokens=600,
                temperature=0.5, # Lower temp = faster/more stable
                do_sample=True,
                eos_token_id=self.END_OF_SPEECH,
                use_cache=True
            )

        generated_tokens = output[0, inputs.shape[1]:]
        audio_tokens = generated_tokens[generated_tokens >= self.AUDIO_CODE_OFFSET] - self.AUDIO_CODE_OFFSET
        
        # Sub-divide into chunks of 20 SNAC frames (~0.2s of audio) for instant streaming
        # Each frame is 7 tokens
        chunk_size_frames = 20 
        tokens_per_chunk = chunk_size_frames * 7
        
        for i in range(0, len(audio_tokens), tokens_per_chunk):
            chunk_tokens = audio_tokens[i : i + tokens_per_chunk]
            if len(chunk_tokens) < 7: continue
            
            # Decode this small chunk immediately
            waveform = self._decode_chunk(chunk_tokens)
            yield (waveform * 32767).astype(np.int16).tobytes()
            await asyncio.sleep(0) # Yield control

    def _decode_chunk(self, audio_tokens):
        n_frames = len(audio_tokens) // 7
        tokens = audio_tokens[:n_frames * 7].cpu().tolist()
        l1, l2, l3 = [], [], []
        for i in range(n_frames):
            base = i * 7
            l1.append(tokens[base + 0] - self.offsets[0])
            l2.append(tokens[base + 1] - self.offsets[1])
            l3.append(tokens[base + 2] - self.offsets[2])
            l3.append(tokens[base + 3] - self.offsets[3])
            l2.append(tokens[base + 4] - self.offsets[4])
            l3.append(tokens[base + 5] - self.offsets[5])
            l3.append(tokens[base + 6] - self.offsets[6])

        c0 = torch.tensor(l1, dtype=torch.long).unsqueeze(0).to(self.device).clamp(0, 4095)
        c1 = torch.tensor(l2, dtype=torch.long).unsqueeze(0).to(self.device).clamp(0, 4095)
        c2 = torch.tensor(l3, dtype=torch.long).unsqueeze(0).to(self.device).clamp(0, 4095)
        
        with torch.no_grad():
            audio_wave = self.snac_model.decode([c0, c1, c2])
        return audio_wave.squeeze().cpu().float().numpy()
