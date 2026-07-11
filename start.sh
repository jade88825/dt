#!/bin/bash
cd cloud_platform
pip3 install --break-system-packages opencv-python-headless flask ultralytics numpy Pillow gunicorn
python3 -m gunicorn app:app --timeout 120 --workers 1 --bind 0.0.0.0:$PORT
