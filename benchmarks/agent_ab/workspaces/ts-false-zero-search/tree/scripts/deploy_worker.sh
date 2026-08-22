#!/usr/bin/env bash
PY="/home/sentiment/venv/bin/python"
exec "$PY" -m worker.main "$@"
