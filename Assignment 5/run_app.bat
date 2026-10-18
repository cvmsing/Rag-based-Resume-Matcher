@echo off
setlocal
cd /d "%~dp0"
title Assignment 5 - Resume Matcher

echo ================================================
echo      Assignment 5 - Resume Matcher UI
echo ================================================
echo.

if not exist ".env" (
    echo [WARNING] .env file not found in this folder.
    echo Create .env and add: COHERE_API_KEY=your_actual_api_key
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating Python virtual environment...
    py -m venv .venv
    if errorlevel 1 goto :error
)

echo [2/3] Checking/installing required packages...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :error

echo.
echo [3/3] Starting Streamlit UI...
echo The browser should open automatically. Keep this window open while using the app.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo [ERROR] The application could not start. Review the message above.
echo Confirm Python is installed and the .env file contains a valid Cohere API key.
pause
exit /b 1
