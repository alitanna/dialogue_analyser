import gc
import torch
import whisperx
from config import WHISPER_MODEL_SIZE, DEVICE, COMPUTE_TYPE

def transcribe_and_align(audio_path: str):
    """
    Transcribes and aligns timestamps. Returns (audio_array, result_dict).
    """
    print(f"Loading WhisperX ({WHISPER_MODEL_SIZE}) on {DEVICE}...")
    
    model = whisperx.load_model(WHISPER_MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=8) # Lower batch for 4GB VRAM
    
    print("Transcription complete. Freeing VRAM...")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading alignment model...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)

    print("Alignment complete. Freeing VRAM...")
    del model_a
    gc.collect()
    torch.cuda.empty_cache()

    full_text = " ".join([seg["text"].strip() for seg in aligned_result["segments"]])
    aligned_result["text"] = full_text.strip()

    # FIX: Return the array first to match run.py unpacking
    return audio, aligned_result