#!/bin/bash
cd cloud_platform
python3 -m gunicorn app:app --timeout 120 --workers 1 --bind 0.0.0.0:$PORT
