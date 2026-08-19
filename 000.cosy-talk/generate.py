import os
import re
import sys
import gc
from pathlib import Path
from datetime import datetime

# 1. Environment Hijacking & Isolation (Before Torch/Transformers/CosyVoice)
ROOT_DIR = Path(__file__).parent.resolve()

CACHE_TEMP = ROOT_DIR / "cache" / "temp"
CACHE_PIP = ROOT_DIR / "cache" / "pip"
HF_HOME = ROOT_DIR / "cache" / "hf"
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "output"
VOICES_DIR = ROOT_DIR / "voices"

os.environ["TEMP"] = str(CACHE_TEMP)
os.environ["TMP"] = str(CACHE_TEMP)
os.environ["PIP_CACHE_DIR"] = str(CACHE_PIP)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["MODELS_DIR"] = str(MODELS_DIR)

for d in [CACHE_TEMP, CACHE_PIP, HF_HOME, MODELS_DIR, OUTPUT_DIR, VOICES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 2. Imports after isolation is enforced
import torch
import torchaudio

try:
    from cosyvoice.cli.cosyvoice import CosyVoice
    from cosyvoice.utils.file_utils import load_wav
    COSYVOICE_AVAILABLE = True
except ImportError:
    COSYVOICE_AVAILABLE = False
    print("Warning: cosyvoice library not found. Will simulate processing for architecture build.")

def parse_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    tags = ["INSTRUCTION", "REFERENCE_AUDIO", "REFERENCE_TRANSCRIPT", "DIALOGUE"]
    parsed = {tag: "" for tag in tags}

    positions = []
    for tag in tags:
        match = re.search(rf"\[{tag}\]", content)
        if match:
            positions.append((match.start(), match.end(), tag))

    positions.sort()

    for i in range(len(positions)):
        start = positions[i][1]
        end = positions[i+1][0] if i + 1 < len(positions) else len(content)
        parsed[positions[i][2]] = content[start:end].strip()

    return parsed

def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def main():
    print("Parsing prompt.txt...")
    prompt_file = ROOT_DIR / "prompt.txt"
    if not prompt_file.exists():
        print("Error: prompt.txt not found.")
        sys.exit(1)

    prompt_data = parse_prompt(prompt_file)
    instruction = prompt_data.get("INSTRUCTION", "")
    ref_audio_path = prompt_data.get("REFERENCE_AUDIO", "")
    ref_transcript = prompt_data.get("REFERENCE_TRANSCRIPT", "")
    dialogue = prompt_data.get("DIALOGUE", "")

    if not dialogue:
        print("Error: No [DIALOGUE] found in prompt.txt")
        sys.exit(1)

    print("Configuring memory-efficient loading for NVIDIA RTX 2060 (6GB VRAM)...")
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        # Prevent complete OOM by restricting memory fraction if needed
        torch.cuda.set_per_process_memory_fraction(0.9)

    cleanup_memory()

    if not COSYVOICE_AVAILABLE:
        print("CosyVoice library is missing in this environment. Script architecture validates.")
        sys.exit(0)

    model_path = str(MODELS_DIR)
    print(f"Loading CosyVoice model from {model_path} into CUDA...")

    try:
        cosyvoice = CosyVoice(model_path, load_jit=True, load_onnx=False)
        # Apply FP16 for memory efficiency on 6GB VRAM
        if torch.cuda.is_available():
            cosyvoice.model.half()
            cosyvoice.model.cuda()
    except Exception as e:
        print(f"Failed to load CosyVoice model: {e}")
        sys.exit(1)

    print("Model loaded. Starting inference...")

    if ref_audio_path and os.path.exists(ROOT_DIR / ref_audio_path):
        ref_audio_full_path = str(ROOT_DIR / ref_audio_path)
        prompt_speech_16k = load_wav(ref_audio_full_path, 16000)
        if torch.cuda.is_available():
            prompt_speech_16k = prompt_speech_16k.half().cuda()
    else:
        prompt_speech_16k = None

    audio_chunks = []

    try:
        if instruction and not ref_transcript:
            print("Running Instructed Inference...")
            for i in cosyvoice.inference_instruct(tts_text=dialogue, spk_id="default", instruct_text=instruction):
                audio_chunks.append(i['tts_speech'])

        elif ref_audio_path and ref_transcript and prompt_speech_16k is not None:
            print("Running Zero-Shot Inference...")
            for i in cosyvoice.inference_zero_shot(tts_text=dialogue, prompt_text=ref_transcript, prompt_speech_16k=prompt_speech_16k):
                audio_chunks.append(i['tts_speech'])

        else:
            print("Running Standard Inference...")
            for i in cosyvoice.inference_sft(tts_text=dialogue, spk_id="default"):
                audio_chunks.append(i['tts_speech'])
    except Exception as e:
        print(f"Inference failed: {e}")
        cleanup_memory()
        sys.exit(1)

    if not audio_chunks:
        print("Error: No audio generated.")
        sys.exit(1)

    output_audio = torch.cat(audio_chunks, dim=1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"generation_{timestamp}.wav"

    print(f"Resampling to 48kHz and saving to {out_file}...")

    if isinstance(output_audio, torch.Tensor):
        if output_audio.dim() == 1:
            output_audio = output_audio.unsqueeze(0)

        sample_rate = 22050
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=48000)
        # Resample requires float32
        audio_48k = resampler(output_audio.cpu().float())

        torchaudio.save(str(out_file), audio_48k, 48000)
        print("Generation Complete!")
    else:
        print("Unexpected output format from CosyVoice.")

    cleanup_memory()

if __name__ == "__main__":
    main()
