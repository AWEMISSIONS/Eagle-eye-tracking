# Eagle Eye Tracking — StreetWatch Pro

Windows-only local AI street / driveway monitor.

Current version: **StreetWatch Pro 3.1 — Fast Start + Camera Picker**

## Main features
- Live selectable Windows camera feed
- Local YOLO vehicle, person, and animal detection
- Fast startup: camera opens while AI loads in the background
- Parked/stationary vehicle suppression
- Persistent vehicle IDs and repeat-vehicle matching
- Person event IDs and animal IDs
- Adaptive learning from missed detections and false alarms
- Night Assist and moving-headlight detection
- Automatic estimated speed plus optional calibrated speed gates
- Road / driveway / sidewalk / ignore zones
- Full-screen camera mode
- Best-frame snapshots
- Local SQLite history and CSV export
- Camera health / reconnect support
- No cloud video processing

## Quick start
1. Clone or download this repository to Windows.
2. Double-click `INSTALL_AND_RUN.bat` once.
3. After setup, use `RUN_STREETWATCH.bat`.
4. Click **FIND CAMERAS**, choose your camera, then **START LIVE**.

## Privacy
All normal processing and event storage are local to the Windows PC. Person events use visit IDs/photos and manual labels; the app does not automatically identify people by face.

## Generated/local files
The repo intentionally excludes the Python virtual environment, model weights, database, snapshots, build outputs, and Python cache files. They are created or downloaded locally when needed.
