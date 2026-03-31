import torch
import subprocess
import os
from dotenv import load_dotenv

def run_diagnostics():
    print("--- 🔍 Running System Diagnostics ---")
    
    # 1. Check PyTorch & CUDA
    print("\n1. Checking GPU capability...")
    if torch.cuda.is_available():
        print(f"✅ CUDA is available! Detected GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM capacity: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("❌ CUDA is NOT available. PyTorch is running on CPU. (Did you install the correct wheel?)")

    # 2. Check FFmpeg
    print("\n2. Checking FFmpeg installation...")
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ FFmpeg is installed and accessible in the system PATH.")
    except FileNotFoundError:
        print("❌ FFmpeg not found! You must install FFmpeg and add it to your Windows PATH.")

    # 3. Check HuggingFace Token
    print("\n3. Checking Environment Variables...")
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if token and token.startswith("hf_"):
        print("✅ HuggingFace token loaded successfully.")
    else:
        print("❌ HF_TOKEN is missing or invalid in your .env file. Pyannote diarization will fail.")

    print("\n--- Diagnostics Complete ---")

if __name__ == "__main__":
    run_diagnostics()