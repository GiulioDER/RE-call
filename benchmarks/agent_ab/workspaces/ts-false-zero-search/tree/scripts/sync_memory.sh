#!/usr/bin/env bash
set -euo pipefail
REMOTE_STORE="/home/sentiment/recall-repos/memory"
rsync -az "$REMOTE_STORE/" ./memory/
