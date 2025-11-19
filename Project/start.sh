#!/bin/sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD="docker-compose"
else
  echo "Neither 'docker compose' nor 'docker-compose' is available. Please install Docker Compose." >&2
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "'ollama' CLI is required on the host. Install it from https://ollama.ai and try again." >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "'curl' is required for health checks. Please install it and retry." >&2
  exit 1
fi

OLLAMA_LISTEN_ADDR="${OLLAMA_LISTEN_ADDR:-0.0.0.0:11434}"
OLLAMA_CLIENT_ADDR="${OLLAMA_CLIENT_ADDR:-127.0.0.1:11434}"
OLLAMA_HEALTH_URL="http://${OLLAMA_CLIENT_ADDR}/api/version"
OLLAMA_LOG="${OLLAMA_LOG:-/tmp/ollama-serve.log}"
OLLAMA_PID=""

cleanup() {
  if [ -n "${OLLAMA_PID}" ]; then
    echo "Stopping local Ollama process (pid ${OLLAMA_PID})..."
    kill "${OLLAMA_PID}" 2>/dev/null || true
    wait "${OLLAMA_PID}" 2>/dev/null || true
  fi
}

ensure_ollama_ready() {
  i=0
  while [ $i -lt 30 ]; do
    if curl -sf "${OLLAMA_HEALTH_URL}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    i=$((i + 1))
  done
  return 1
}

if ! curl -sf "${OLLAMA_HEALTH_URL}" >/dev/null 2>&1; then
  echo "Starting local Ollama server..."
  trap cleanup EXIT INT TERM
  OLLAMA_HOST="${OLLAMA_LISTEN_ADDR}" nohup ollama serve >"${OLLAMA_LOG}" 2>&1 &
  OLLAMA_PID=$!
  if ! ensure_ollama_ready; then
    echo "Ollama did not become ready. Check logs at ${OLLAMA_LOG}." >&2
    exit 1
  fi
  echo "Ollama is running (log: ${OLLAMA_LOG})."
else
  echo "Detected existing Ollama server at ${OLLAMA_CLIENT_ADDR}."
fi

prime_model() {
  model="$1"
  echo "Priming model ${model}..."
  if ! printf "%s" "Warmup request for Med pipeline readiness." | OLLAMA_HOST="${OLLAMA_CLIENT_ADDR}" ollama run "${model}" >/dev/null 2>&1; then
    echo "Failed to prime ${model}. Ensure the model name is correct and the server is running." >&2
    exit 1
  fi
}

prime_model "hf.co/unsloth/medgemma-27b-it-GGUF:Q4_K_M"
prime_model "hf.co/mradermacher/II-Medical-8B-GGUF:Q4_K_M"

echo "Starting Docker services..."
${COMPOSE_CMD} up --build "$@"
