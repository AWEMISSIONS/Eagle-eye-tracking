@echo off
set "F=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\StreetWatch-Pro.bat"
if exist "%F%" del "%F%"
echo StreetWatch auto-start has been disabled.
pause
