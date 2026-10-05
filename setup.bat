@echo off
rem One-time setup: creates .venv and installs dependencies.
cd /d "%~dp0"
python -m venv .venv || (echo Python 3.10+ is required: https://www.python.org/downloads/ & exit /b 1)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt || exit /b 1
where nvidia-smi >nul 2>nul && (
    echo NVIDIA GPU found: installing CUDA libraries for GPU acceleration...
    ".venv\Scripts\python.exe" -m pip install -r requirements-gpu.txt
)
echo Setup complete.
