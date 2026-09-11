import os
import sys
from datetime import datetime
import numpy as np

# ==============================================================================
# 1. DIRECTORY ISOLATION & LOCAL PATH MAPPING
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
TEMP_DIR = os.path.join(SCRIPT_DIR, "cache", "temp")
PROMPT_FILE = os.path.join(SCRIPT_DIR, "prompt.txt")
MODEL_PATH = os.path.join(MODELS_DIR, "CosyVoice2-0.5B")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

os.environ["HF_HOME"] = MODELS_DIR
os.environ["MODELSCOPE_CACHE"] = MODELS_DIR
os.environ["TORCH_HOME"] = MODELS_DIR
os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR

sys.path.append(os.path.join(SCRIPT_DIR, "CosyVoice"))
sys.path.append(os.path.join(SCRIPT_DIR, "CosyVoice", "third_party", "Matcha-TTS"))

import torch
import torchaudio.transforms as T
import soundfile as sf
import cosyvoice.utils.file_utils

# ==============================================================================
# 2. TORCHCODEC-FREE AUDIO LOADERS
# ==============================================================================
def safe_load_wav(wav_path, target_sr=16000):
    """Loads and resamples audio, handling both file paths and pre-loaded tensors."""
    # If CosyVoice passes the pre-loaded 16k tensor back in, resample it to 24k directly
    if isinstance(wav_path, torch.Tensor):
        speech = wav_path
        if target_sr == 24000:
            resampler = T.Resample(orig_freq=16000, new_freq=24000)
            speech = resampler(speech)
        return speech

    # If it is a string file path, load it normally
    audio, sample_rate = sf.read(wav_path, dtype="float32")
    speech = torch.from_numpy(audio).float()

    if speech.ndim == 1:
        speech = speech.unsqueeze(0)
    elif speech.ndim == 2:
        speech = speech.t().mean(dim=0, keepdim=True)

    if sample_rate != target_sr:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr)
        speech = resampler(speech)

    return speech

# Monkeypatch the internal loader
cosyvoice.utils.file_utils.load_wav = safe_load_wav

from cosyvoice.cli.cosyvoice import AutoModel

# ==============================================================================
# 3. PROMPT PARSER
# ==============================================================================
def parse_prompt(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing prompt file at: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    sections = {}
    current_tag = None
    buffer = []

    for line in content.splitlines():
        line_strip = line.strip()
        if line_strip.startswith("[") and line_strip.endswith("]"):
            if current_tag:
                sections[current_tag] = "\n".join(buffer).strip()
                buffer = []
            current_tag = line_strip[1:-1].upper()
        else:
            buffer.append(line)

    if current_tag:
        sections[current_tag] = "\n".join(buffer).strip()

    return (
        sections.get("INSTRUCTION", ""),
        sections.get("REFERENCE_AUDIO", ""),
        sections.get("REFERENCE_TRANSCRIPT", ""),
        sections.get("DIALOGUE", "")
    )

# ==============================================================================
# 4. INFERENCE PIPELINE
# ==============================================================================
def main():
    print("=" * 60)
    print(" CosyVoice 2 Standalone Inference Engine")
    print("=" * 60)

    instruction, ref_audio_rel, ref_text, dialogue = parse_prompt(PROMPT_FILE)
    ref_audio_path = os.path.join(SCRIPT_DIR, ref_audio_rel) if ref_audio_rel else None

    print(f"[Directing] Instruction : {instruction or 'Zero-Shot Mimicry'}")
    print(f"[Reference] Voice Audio : {ref_audio_path}")
    print(f"[Dialogue]  Synthesis   : {dialogue}")
    print("-" * 60)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at: {MODEL_PATH}")

    print(f"[Loader] Loading model weights from: {MODEL_PATH}")
    cosyvoice = AutoModel(model_dir=MODEL_PATH, fp16=True)

    print("[Pipeline] Synthesizing audio...")
    if ref_audio_path and os.path.exists(ref_audio_path):
        prompt_speech_16k = safe_load_wav(ref_audio_path, 16000)
        if instruction:
            output = cosyvoice.inference_instruct2(dialogue, instruction, prompt_speech_16k)
        else:
            output = cosyvoice.inference_zero_shot(dialogue, ref_text, prompt_speech_16k)
    else:
        print("[Warning] No reference audio found. Running standard synthesis.")
        output = cosyvoice.inference_sft(dialogue, "中文女")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_file = os.path.join(OUTPUT_DIR, f"cosy_dialogue_{timestamp}.wav")

    for res in output:
        audio_np = res["tts_speech"].squeeze().detach().cpu().numpy()
        sf.write(out_file, audio_np, cosyvoice.sample_rate)
        break

    print("-" * 60)
    print(f"[SUCCESS] Audio generated and saved to:")
    print(f"          -> {out_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()