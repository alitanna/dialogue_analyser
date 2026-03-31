@echo off
echo ==========================================
echo   Dialogue Analyzer - Windows Setup
echo ==========================================

:: Check for FFmpeg (Required per Dissertation Section 3.1.1)
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] FFmpeg is not installed or not in PATH.
    echo Please install FFmpeg from https://ffmpeg.org/ before continuing.
    pause
    exit /b
)

echo [1/4] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo [2/4] Upgrading build tools...
python -m pip install --upgrade pip setuptools wheel

echo [3/4] Installing GPU-accelerated PyTorch (CUDA 12.1)...
:: Matches your RTX 3050 hardware requirement
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo [4/4] Installing remaining project requirements...
pip install -r requirements.txt

echo ==========================================
echo Setup Complete! 
echo 1. Create a .env file with your HF_TOKEN.
echo 2. Run: venv\Scripts\activate
echo 3. Run: streamlit run app.py
echo ==========================================
pause