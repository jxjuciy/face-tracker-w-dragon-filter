# Dragon Face Tracker

A webcam face-tracking demo that overlays a textured 3D dragon bust when the
user opens their mouth.

## Run on Windows

1. Install Python 3.11 and make sure the Python Launcher (`py`) is available.
2. Download or clone this repository.
3. Double-click `run_windows.bat`.
4. Allow camera access if Windows asks. The first launch needs internet access
   to download the MediaPipe face-landmarker model.

To run manually from Command Prompt in the project folder:

```bat
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

Press **Q** or **Esc** to quit. If the wrong camera opens, change `CAM_SOURCE`
near the top of `hand_ar.py` (`0` is the default camera; try `1` for another).

## Project contents

- `main.py` starts the tracker.
- `hand_ar.py` implements face tracking, 3D rendering, and compositing.
- `model/` contains the dragon GLTF mesh and its texture.

The supplied dragon is a head-and-shoulders bust, not a full-body model.

## Dragon model credit and license

“Dragon Head with Human Form” by Scorpion:

- Source: https://sketchfab.com/3d-models/dragon-head-with-human-form-43b8b08e096f4fd3a87a12f98a668d54
- License: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

The model requires attribution and is not licensed for commercial use. See
`model/license.txt` for the original attribution notice.
