@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo StreetWatch has not been installed yet.
  echo Run INSTALL_AND_RUN.bat first.
  pause
  exit /b 1
)
start "StreetWatch" ".venv\Scripts\pythonw.exe" app\streetwatch.py
