import os
import subprocess
import yt_dlp
from config import TEMP_DIR, AUDIO_SAMPLE_RATE

def get_video_duration(url):
    """Checks video length before downloading to prevent GPU OOM."""
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False)
        return info.get('duration', 0)

def download_and_convert_audio(youtube_url: str) -> str:
    """
    Downloads audio from YouTube and converts it to 16kHz mono WAV.
    Returns the path to the processed WAV file.
    """
    print(f"Downloading audio from: {youtube_url}")
    
    # We use a static filename for simplicity in the temp directory
    raw_audio_base = os.path.join(TEMP_DIR, "raw_download")
    final_wav_path = os.path.join(TEMP_DIR, "processed_audio.wav")
    
    # Clean up previous runs to avoid conflicts
    if os.path.exists(final_wav_path):
        os.remove(final_wav_path)

    # yt-dlp options to grab the best audio format
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': raw_audio_base,
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
    except Exception as e:
        raise RuntimeError(f"Failed to download from YouTube. Check URL or network: {e}")

    # Determine the actual downloaded file (yt-dlp might add extensions like .webm or .m4a)
    downloaded_files = [f for f in os.listdir(TEMP_DIR) if f.startswith("raw_download")]
    if not downloaded_files:
        raise FileNotFoundError("Audio download failed, no file found in temp directory.")
    
    actual_raw_path = os.path.join(TEMP_DIR, downloaded_files[0])

    print("Converting to 16kHz mono WAV using FFmpeg...")
    
    # FFmpeg command to enforce 16000 Hz and mono audio
    command = [
        "ffmpeg",
        "-y",                          # Overwrite output
        "-i", actual_raw_path,         # Input file
        "-ar", str(AUDIO_SAMPLE_RATE), # 16000 Hz
        "-ac", "1",                    # Mono channel
        "-c:a", "pcm_s16le",           # 16-bit WAV codec
        final_wav_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed. Ensure FFmpeg is installed and in your PATH. Error: {e}")
    finally:
        # Clean up the raw download to save disk space
        if os.path.exists(actual_raw_path):
            os.remove(actual_raw_path)

    print(f"Audio processing complete. Ready for transcription.")
    return final_wav_path