@echo off
REM Quick start script for LoRA Persona project

echo.
echo ================================================
echo LoRA Persona Project - Quick Start
echo ================================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Virtual environment not found. Creating it...
    uv venv --python 3.11
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check environment status
echo.
echo Checking environment status...
python setup_env.py check

echo.
echo ================================================
echo Environment ready! You can now:
echo 1. Run: jupyter lab
echo 2. Select "LoRA Persona (uv)" kernel
echo 3. Open LoRA-slack-persona-local.ipynb
echo ================================================
echo.

REM Keep window open
cmd /k