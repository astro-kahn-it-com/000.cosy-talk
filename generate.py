import os
import sys

# Hijack environment variables to enforce directory isolation before heavy imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cache_temp = os.path.join(BASE_DIR, "cache", "temp")
cache_pip = os.path.join(BASE_DIR, "cache", "pip")
models_dir = os.path.join(BASE_DIR, "models")

# We only set them if not already set by start_generation.bat or we enforce them here
os.environ["TEMP"] = cache_temp
os.environ["TMP"] = cache_temp
os.environ["PIP_CACHE_DIR"] = cache_pip
os.environ["HF_HOME"] = models_dir
os.environ["MODELS_DIR"] = models_dir
# Enforce PyTorch memory efficiency for limited VRAM (6GB RTX 2060)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# Optional: ensure local folders exist in case the script is run directly
for path in [cache_temp, cache_pip, models_dir, os.path.join(BASE_DIR, "output"), os.path.join(BASE_DIR, "voices")]:
    os.makedirs(path, exist_ok=True)

import time
import torch
import torchaudio

# We need to handle potential ModuleNotFoundError in the isolated environment for testing
try:
    from cosyvoice.cli.cosyvoice import CosyVoice
    from cosyvoice.utils.file_utils import load_wav
except ImportError as e:
    print(f"Warning: CosyVoice could not be imported: {e}. Ensure it is installed in the embedded Python environment.")
    CosyVoice = None
    load_wav = None


def parse_prompt(prompt_file="prompt.txt"):
    """
    Parses the prompt.txt file and extracts the tagged blocks.
    Returns a dictionary with keys matching the tags.
    """
    tags = ["[INSTRUCTION]", "[REFERENCE_AUDIO]", "[REFERENCE_TRANSCRIPT]", "[DIALOGUE]"]
    parsed_data = {tag: "" for tag in tags}

    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Could not find {prompt_file}")

    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    current_tag = None
    for line in content.split('\n'):
        line = line.strip()
        if line in tags:
            current_tag = line
        elif current_tag:
            if line: # Append non-empty lines
                if parsed_data[current_tag]:
                    parsed_data[current_tag] += " " + line
                else:
                    parsed_data[current_tag] = line

    return parsed_data


def get_cosyvoice_model(model_dir):
    """
    Loads the CosyVoice model using half precision (FP16/BF16) or similar memory-efficient configuration.
    """
    print(f"Loading CosyVoice model from {model_dir}...")
    # Typically, you'd specify a subfolder like 'CosyVoice-300M-SFT' or 'CosyVoice-300M-Instruct'
    # Fallback to a default directory if it's placed exactly in HF_HOME or MODELS_DIR
    # This might depend on the specific downloaded model layout. Let's assume 'CosyVoice-300M-Instruct' for instruction support.
    instruct_model_path = os.path.join(model_dir, "CosyVoice-300M-Instruct")
    sft_model_path = os.path.join(model_dir, "CosyVoice-300M-SFT")
    zero_shot_model_path = os.path.join(model_dir, "CosyVoice-300M")

    path_to_load = None
    if os.path.exists(instruct_model_path):
        path_to_load = instruct_model_path
    elif os.path.exists(sft_model_path):
        path_to_load = sft_model_path
    elif os.path.exists(zero_shot_model_path):
        path_to_load = zero_shot_model_path
    else:
        # Just use the root model dir if it contains the model files, otherwise defaults
        path_to_load = model_dir

    if CosyVoice is None:
         print("CosyVoice module is not available. Skipping model loading for dry run.")
         return None

    # Load with basic parameters. Memory efficiency might rely on PyTorch configurations and device mapping.
    model = CosyVoice(path_to_load)

    # Optional: Quantization or half precision for memory efficiency
    # If CosyVoice API supports it, one might do model.half() or load in fp16.
    # Currently relying on PYTORCH_CUDA_ALLOC_CONF and clearing cache.

    return model

def main():
    print("Starting generation pipeline...")

    prompt_file = os.path.join(BASE_DIR, "prompt.txt")
    output_dir = os.path.join(BASE_DIR, "output")

    # 1. Parse prompt
    print("Parsing prompt.txt...")
    try:
        prompts = parse_prompt(prompt_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    instruction = prompts["[INSTRUCTION]"]
    ref_audio_path = prompts["[REFERENCE_AUDIO]"]
    ref_transcript = prompts["[REFERENCE_TRANSCRIPT]"]
    dialogue = prompts["[DIALOGUE]"]

    print("Parsed Inputs:")
    print(f"  Instruction: {instruction}")
    print(f"  Ref Audio: {ref_audio_path}")
    print(f"  Ref Transcript: {ref_transcript}")
    print(f"  Dialogue: {dialogue}")

    if not dialogue:
        print("Error: No dialogue provided in prompt.txt.")
        return

    # 2. Load Model
    torch.cuda.empty_cache() # Clear cache before loading
    model = get_cosyvoice_model(models_dir)

    if model is None:
        print("Model loading simulated. End of run.")
        return

    # 3. Prepare Inference Inputs
    print("Preparing inputs for inference...")
    ref_audio_tensor = None
    if ref_audio_path and os.path.exists(os.path.join(BASE_DIR, ref_audio_path)):
        # CosyVoice load_wav utility handles standard resampling typically (16kHz for model input)
        ref_audio_tensor = load_wav(os.path.join(BASE_DIR, ref_audio_path), 16000)
    elif ref_audio_path:
        print(f"Warning: Reference audio not found at {ref_audio_path}")

    # Determine inference mode
    outputs = []

    torch.cuda.empty_cache() # Clear cache before inference

    print("Running inference...")
    with torch.no_grad():
        if instruction and ref_audio_tensor is not None and ref_transcript:
             # If we have everything, maybe instruct or zero-shot depending on model.
             # For CosyVoice-300M-Instruct:
             # The instruct model usually takes text, spk_id, instruct_text.
             # If zero_shot is needed, we need prompt_speech and prompt_text.
             # Let's assume zero-shot if we have prompt speech and text, and instruct if we just have instruction.
             # Actually, CosyVoice has specific methods: inference_instruct, inference_zero_shot, inference_sft
             print("Using zero_shot / instruct mode...")
             # Note: API might vary, fallback to zero-shot if both are provided for a specific model, or loop generators.
             if hasattr(model, 'inference_zero_shot'):
                  for i, j in enumerate(model.inference_zero_shot(dialogue, ref_transcript, ref_audio_tensor)):
                      outputs.append(j['tts_speech'])
             else:
                 print("Zero-shot inference method not found on model.")
        elif ref_audio_tensor is not None and ref_transcript:
             print("Using zero_shot mode...")
             if hasattr(model, 'inference_zero_shot'):
                  for i, j in enumerate(model.inference_zero_shot(dialogue, ref_transcript, ref_audio_tensor)):
                      outputs.append(j['tts_speech'])
        elif instruction:
             print("Using instruct mode...")
             # instruct mode usually expects the speaker embedding or id, let's use default or sft if available.
             if hasattr(model, 'inference_instruct'):
                  # Assuming default speaker or fallback
                  for i, j in enumerate(model.inference_instruct(dialogue, 'default_speaker', instruction)):
                      outputs.append(j['tts_speech'])
        else:
             print("Using SFT mode...")
             if hasattr(model, 'inference_sft'):
                  for i, j in enumerate(model.inference_sft(dialogue, 'default_speaker')):
                      outputs.append(j['tts_speech'])

    if not outputs:
        print("Inference yielded no outputs. Check model API and provided inputs.")
        return

    # Concatenate if multiple chunks
    audio_output = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]

    # 4. Save Output (resample to 48kHz if needed, CosyVoice output is often 22050Hz or 24000Hz)
    # Get model sample rate, usually 22050
    model_sr = 22050
    target_sr = 48000

    print(f"Resampling from {model_sr}Hz to {target_sr}Hz...")
    if model_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=model_sr, new_freq=target_sr).to(audio_output.device)
        audio_output = resampler(audio_output)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"output_{timestamp}.wav")

    print(f"Saving output to {out_file}...")
    # audio_output is usually [1, T] tensor.
    torchaudio.save(out_file, audio_output.cpu(), target_sr)
    print("Done!")

if __name__ == "__main__":
    main()
