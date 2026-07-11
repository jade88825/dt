#!/bin/bash
cd cloud_platform
python3 -m venv /opt/venv
/opt/venv/bin/pip install --upgrade pip
/opt/venv/bin/pip install opencv-python-headless flask ultralytics numpy Pillow
/opt/venv/bin/python3 app.py
