import gc
import torch
import pandas as pd
import whisperx
from pyannote.audio import Pipeline
from config import HF_TOKEN, DEVICE

def diarize_audio(audio_data, aligned_result):
    """
    Runs pyannote.audio diarization and converts output to a DataFrame 
    to prevent 'Annotation object has no attribute iterrows' errors.
    """
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is missing in .env")

    print("--- Step 3: Identifying Speakers (Pyannote 3.1) ---")
    
    # 1. Load the Model
    diarize_model = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN 
    ).to(torch.device(DEVICE))
    
    # 2. Prepare Waveform Tensor
    audio_tensor = torch.from_numpy(audio_data).to(DEVICE).unsqueeze(0)
    input_data = {"waveform": audio_tensor, "sample_rate": 16000}
    
    print("Running diarization inference...")
    outputs = diarize_model(input_data)

    # 3. Extract the Annotation
    diarization = getattr(outputs, "speaker_diarization", outputs)

    # --- THE FIX: Convert Annotation to Pandas DataFrame ---
    print("Converting Annotation to DataFrame for WhisperX compatibility...")
    segments = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": segment.start,
            "end": segment.end,
            "speaker": speaker
        })
    diarize_df = pd.DataFrame(segments)
    # -------------------------------------------------------

    # 4. Critical VRAM Cleanup
    print("Freeing VRAM after Diarization...")
    del diarize_model
    del audio_tensor
    gc.collect()
    torch.cuda.empty_cache() 

    print("Assigning speakers to word-level segments...")
    # FIX: Pass the DataFrame (diarize_df) instead of the Annotation object
    final_result = whisperx.assign_word_speakers(diarize_df, aligned_result)
    
    return final_result