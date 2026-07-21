# DiffusionGemma-26B-A4B-it-NVFP4 serving image for vast.ai
# Base: official vLLM image with diffusion_gemma support built in
FROM vllm/vllm-openai:gemma

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Open WebUI in an isolated venv so its deps never touch vLLM's
RUN pip install --no-cache-dir uv \
    && uv venv /opt/openwebui --python 3.11 \
    && uv pip install --python /opt/openwebui/bin/python --no-cache open-webui

ENV MODEL_ID=nvidia/diffusiongemma-26B-A4B-it-NVFP4 \
    HF_HOME=/workspace/hf_cache \
    DATA_DIR=/workspace/open-webui \
    VLLM_USE_V2_MODEL_RUNNER=1

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000 8080
ENTRYPOINT ["/entrypoint.sh"]
