@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
(
  echo @echo off
  echo cd /d "%CD%"
  echo call "%CD%\RUN_STREETWATCH.bat"
) > "%STARTUP%\StreetWatch-Pro.bat"
echo StreetWatch will now start when you sign in to Windows.
pause
