#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-nvidia/diffusiongemma-26B-A4B-it-NVFP4}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export DATA_DIR="${DATA_DIR:-/workspace/open-webui}"
mkdir -p "$HF_HOME" "$DATA_DIR"

# GPU sanity check: NVFP4 weights alone need ~16.5GB free
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv || true
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 || echo 0)
    if [ "${FREE_MB:-0}" -lt "${MIN_FREE_VRAM_MB:-18000}" ]; then
        echo "[entrypoint] FATAL: only ${FREE_MB}MiB VRAM free — another process on this host is holding GPU memory." >&2
        echo "[entrypoint] This vast.ai machine is unusable for this model. Destroy the instance and rent a different machine." >&2
        sleep 60   # avoid tight restart loop billing
        exit 1
    fi
fi

# Diffusion sampler warmup allocates vocab(262k) x 256-token-block x num_seqs
# fp32 buffers outside the vLLM memory budget; keep both small on 24GB cards.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[entrypoint] starting vLLM for ${MODEL_ID} on :8000 (first boot downloads ~16.5GB weights)"
# shellcheck disable=SC2086
vllm serve "$MODEL_ID" \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code \
    --max-num-seqs "${MAX_NUM_SEQS:-2}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.84}" \
    --max-model-len "${MAX_MODEL_LEN:-32768}" \
    --attention-backend TRITON_ATTN \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --override-generation-config '{"max_new_tokens": null}' \
    --default-chat-template-kwargs '{"enable_thinking":true}' \
    ${VLLM_EXTRA_ARGS:-} &
VLLM_PID=$!

echo "[entrypoint] waiting for vLLM /health ..."
until curl -sf http://127.0.0.1:8000/health >/dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[entrypoint] vLLM exited before becoming healthy" >&2
        exit 1
    fi
    sleep 5
done
echo "[entrypoint] vLLM ready"

export OPENAI_API_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-vllm-local}"
export ENABLE_OLLAMA_API=false

echo "[entrypoint] starting Open WebUI on :8080"
/opt/openwebui/bin/open-webui serve --host 0.0.0.0 --port 8080 &
WEBUI_PID=$!

trap 'kill "$VLLM_PID" "$WEBUI_PID" 2>/dev/null || true' TERM INT

# If either process dies, bring the container down so vast.ai surfaces it
wait -n "$VLLM_PID" "$WEBUI_PID"
STATUS=$?
echo "[entrypoint] a service exited (status ${STATUS}), shutting down" >&2
kill "$VLLM_PID" "$WEBUI_PID" 2>/dev/null || true
exit "$STATUS"
