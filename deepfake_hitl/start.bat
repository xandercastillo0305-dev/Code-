@echo off
REM Double-click to start the prototype and open it in the browser.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo .venv not found. Follow the Setup steps in README.md first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python run.py
pause
