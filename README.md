# RoadSafe (Windows setup)

1. Install Python 3.11 (tick "Add python.exe to PATH").
2. Extract this folder to C:\dev\roadsafe_project
3. In a terminal inside the folder:

       py -3.11 -m venv .venv
       .venv\Scripts\activate
       pip install -r requirements.txt
       python app.py

4. Open http://127.0.0.1:5000 and upload a road video.

The first run downloads yolov8n.pt automatically (needs internet).
