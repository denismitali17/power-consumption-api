#!/bin/bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 10