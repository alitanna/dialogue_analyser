import os
import warnings
import gc
import torch

# 1. Immediate Heartbeat to confirm execution
print("--- [System] Initializing Pipeline Modules... ---")

# 2. Suppress noisy warnings for a clean dissertation demo
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torchcodec.*")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 3. Absolute Imports
from pipeline.audio import download_and_convert_audio
from pipeline.transcribe import transcribe_and_align
from pipeline.diarize import diarize_audio
from pipeline.segment import group_and_chunk_segments
from pipeline.sentiment import analyze_sentiment
from pipeline.summarize import generate_summaries

def run_full_analysis(youtube_url: str, progress_callback=None):
    """
    Executes the 6-module pipeline per Dissertation Section 3.2.7.
    """
    def update_progress(msg, p):
        print(f"--- [Progress {p}%]: {msg} ---")
        if progress_callback: progress_callback(msg, p)

    try:
        # Step 1 & 2: Acquisition and Transcription
        update_progress("Downloading audio...", 10)
        audio_path = download_and_convert_audio(youtube_url)

        update_progress("Transcribing (Whisper-small)...", 30)
        audio_data, whisper_results = transcribe_and_align(audio_path)

        # Step 3: Diarization & Speaker Assignment
        # This module now returns a WhisperX dictionary with speaker labels
        update_progress("Diarizing speakers (Pyannote 3.1)...", 50)
        final_diarized = diarize_audio(audio_data, whisper_results)

        # Step 4: Segmentation (Module 4)
        update_progress("Segmenting dialogue...", 65)
        # Note: Ensure segment.py extracts results.get("segments", [])
        speaker_turns = group_and_chunk_segments(final_diarized)

        # Step 5: Sentiment Analysis (Module 5)
        update_progress("Analyzing sentiment (Neutral Override)...", 80)
        analyzed_turns = analyze_sentiment(speaker_turns)

        # Step 6: Hierarchical Summarization (Module 6)
        update_progress("Generating summaries (BART)...", 95)
        summary_results = generate_summaries(analyzed_turns)

        update_progress("Analysis Complete!", 100)
        
        # Return structured analytical information
        return {
            "transcript": analyzed_turns, 
            "summary": summary_results['overall'],
            "speaker_summaries": summary_results['per_speaker']
        }

    except Exception as e:
        # Safety: Clear VRAM for 4GB RTX 3050
        torch.cuda.empty_cache()
        gc.collect()
        raise RuntimeError(f"Orchestrator Error: {str(e)}")

if __name__ == "__main__":
    test_url = "https://www.youtube.com/watch?v=bUvzds-p4oE" 
    print(f"\n🚀 STARTING ANALYSIS: {test_url}")
    try:
        results = run_full_analysis(test_url)
        print("\n" + "="*30)
        print("✅ SUCCESSFUL RUN")
        print("="*30)
        print(f"\nSUMMARY:\n{results['summary']}")
    except Exception as e:
        print(f"\n❌ PIPELINE CRASHED: {str(e)}")