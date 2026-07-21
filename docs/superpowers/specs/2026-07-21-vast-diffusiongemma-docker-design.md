# vast.ai DiffusionGemma NVFP4 서빙 도커 이미지 — 설계

날짜: 2026-07-21
상태: 승인됨

## 목적

vast.ai에서 RTX PRO 4000 (Blackwell, 24GB)을 빌려
`nvidia/diffusiongemma-26B-A4B-it-NVFP4`를 서빙하고 WebUI로 바로 접속 가능한
단일 도커 이미지. GitHub에 올려 GHCR에서 vast.ai가 바로 pull.

## 핵심 결정

| 항목 | 결정 | 이유 |
|---|---|---|
| 서빙 엔진 | vLLM (`vllm/vllm-openai:gemma` 공식 이미지) | diffusiongemma는 diffusion LM + MoE + NVFP4 → ollama/llama.cpp 미지원. 모델 카드 공식 스택이 vLLM |
| WebUI | Open WebUI (pip, 별도 uv venv) | OpenAI 호환 API 직결. venv 격리로 vLLM 의존성 충돌 방지 |
| 모델 가중치 | 이미지에 미포함, 첫 부팅 시 HF 다운로드 (~16.5GB) → `/workspace/hf_cache` | 이미지 슬림 유지, vast 디스크에 캐시 |
| 배포 | GitHub Actions → ghcr.io 자동 빌드/푸시 | vast 템플릿에 이미지 주소만 입력 |
| 포트 | 8000 = vLLM API, 8080 = Open WebUI | vast에서 `-p 8000:8000 -p 8080:8080` |

## 구성 요소

- `Dockerfile` — base `vllm/vllm-openai:gemma`, curl 설치, uv venv에 open-webui 설치
- `entrypoint.sh` — vLLM 기동(:8000) → `/health` 폴링 → Open WebUI 기동(:8080), 둘 중 하나 죽으면 컨테이너 종료
- `.github/workflows/docker-publish.yml` — main push 시 GHCR 배포, 러너 디스크 확보 단계 포함
- `README.md` — vast.ai 템플릿 설정 가이드

## vLLM 기동 옵션 (모델 카드 준수)

`--trust-remote-code --attention-backend TRITON_ATTN --enable-auto-tool-choice
--tool-call-parser gemma4 --reasoning-parser gemma4
--override-generation-config '{"max_new_tokens": null}'
--default-chat-template-kwargs '{"enable_thinking":true}'`
+ 24GB VRAM 대응: `--max-num-seqs 4`, `--gpu-memory-utilization 0.92`,
`--max-model-len 32768` (256K 전체 컨텍스트는 24GB에서 KV cache 불가).
환경변수 `VLLM_USE_V2_MODEL_RUNNER=1` 필수.

## 에러 처리

- vLLM 프로세스 사망 시 헬스체크 루프가 감지하고 컨테이너 종료 (vast에서 로그 확인 가능)
- 환경변수로 오버라이드 가능: `MODEL_ID`, `MAX_NUM_SEQS`, `GPU_MEM_UTIL`, `MAX_MODEL_LEN`, `VLLM_EXTRA_ARGS`

## 검증

- 로컬: `docker build` 성공 (Mac에서 GPU 실행은 불가 — 실행 검증은 vast에서)
- GHCR: Actions 빌드 통과
- vast: 인스턴스 기동 → 8080 접속 → 채팅 응답 확인
