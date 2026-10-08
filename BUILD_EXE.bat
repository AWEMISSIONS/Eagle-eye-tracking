@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Build StreetWatch Pro EXE
if not exist ".venv\Scripts\python.exe" (
  echo Run INSTALL_AND_RUN.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade pyinstaller
if errorlevel 1 goto :fail
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul

set "MODELARG="
if exist "yolo26n.pt" set MODELARG=--add-data "yolo26n.pt;."

".venv\Scripts\pyinstaller.exe" --noconfirm --windowed --name StreetWatch --collect-all ultralytics --collect-all torch --collect-all torchvision --collect-all pyttsx3 --hidden-import=PIL._tkinter_finder %MODELARG% app\streetwatch.py
if errorlevel 1 goto :fail

echo.
echo Finished. Your normal Windows app folder is:
echo   %CD%\dist\StreetWatch\
echo Double-click StreetWatch.exe inside that folder.
explorer "%CD%\dist\StreetWatch"
pause
exit /b 0

:fail
echo.
echo EXE build failed. Send a screenshot of this window if you want me to diagnose it.
pause
exit /b 1
