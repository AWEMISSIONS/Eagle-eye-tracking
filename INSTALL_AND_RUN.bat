@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title StreetWatch Pro Setup
color 0A

echo ============================================================
echo   StreetWatch Pro for Windows - First-Time Setup
 echo ============================================================
echo.

set "PYEXE="
where py >nul 2>nul && set "PYEXE=py -3.12"
if not defined PYEXE where python >nul 2>nul && set "PYEXE=python"

if not defined PYEXE (
  echo Python was not found. StreetWatch will try to install Python 3.12.
  where winget >nul 2>nul
  if errorlevel 1 (
    echo.
    echo Windows Package Manager ^(winget^) was not found.
    echo Install Python 3.12 from https://www.python.org/downloads/windows/
    echo and then double-click this file again.
    pause
    exit /b 1
  )
  winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
  if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
    set "PYEXE=%LocalAppData%\Programs\Python\Python312\python.exe"
  ) else (
    echo.
    echo Python installation finished, but this window cannot see it yet.
    echo Close this window and double-click INSTALL_AND_RUN.bat again.
    pause
    exit /b 0
  )
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating StreetWatch environment...
  %PYEXE% -m venv .venv
  if errorlevel 1 goto :fail
)

echo Updating installer...
".venv\Scripts\python.exe" -m pip install --upgrade pip wheel
if errorlevel 1 goto :fail

echo Installing StreetWatch Pro and local AI dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo Preparing the YOLO26 model. The first setup may download it once...
".venv\Scripts\python.exe" -c "from ultralytics import YOLO; YOLO('yolo26n.pt'); print('YOLO26 model ready.')"
if errorlevel 1 goto :fail

echo.
echo Setup finished successfully.
echo NEXT TIME: use RUN_STREETWATCH.bat -- you do NOT reinstall each time.
echo.
start "StreetWatch Pro" ".venv\Scripts\pythonw.exe" app\streetwatch.py
pause
exit /b 0

:fail
echo.
echo Setup failed. This window will stay open.
echo Send a screenshot of the error and I can diagnose it.
pause
exit /b 1
