#!/usr/bin/env sh
# POSIX launcher. Usage:
#   ./run.sh --url https://api.example.com/v1 --key sk-... --model my-model
set -e
cd "$(dirname "$0")"
exec python -m model_fingerprint "$@"
