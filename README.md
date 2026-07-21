# vast-diffusiongemma

vast.ai에서 **RTX PRO 4000 (Blackwell, 24GB)** 으로
[nvidia/diffusiongemma-26B-A4B-it-NVFP4](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4)를
vLLM으로 서빙하고 Open WebUI로 바로 접속하는 단일 도커 이미지.

- `:8000` — vLLM OpenAI 호환 API (`/v1/chat/completions`)
- `:8080` — Open WebUI (채팅 UI)

> **왜 ollama/llama-server가 아니라 vLLM인가?**
> DiffusionGemma는 discrete diffusion LM + MoE 아키텍처에 NVFP4 양자화라서
> llama.cpp 계열(ollama, llama-server)이 지원하지 않는다.
> 모델 카드의 공식 서빙 스택은 vLLM이며, 이 이미지는 diffusion_gemma 지원이
> 내장된 공식 `vllm/vllm-openai:gemma` 이미지를 베이스로 쓴다.

## 1. GitHub → GHCR 배포

1. 이 저장소를 GitHub에 push하면 Actions가 자동으로 `ghcr.io/<owner>/<repo>:latest`를 빌드/푸시한다.
2. **GHCR 패키지를 public으로 전환**해야 vast.ai가 인증 없이 pull 가능:
   GitHub → 프로필 → Packages → 해당 이미지 → Package settings → Change visibility → Public.

## 2. vast.ai 인스턴스 설정

템플릿(Edit Image & Config)에서:

| 항목 | 값 |
|---|---|
| Image Path | `ghcr.io/<owner>/<repo>:latest` |
| Docker Options | `-p 8080:8080 -p 8000:8000` |
| Launch Mode | `docker ENTRYPOINT` (SSH/Jupyter 모드 아님) |
| Disk | **60GB 이상** (모델 ~16.5GB + WebUI + 이미지 레이어) |
| GPU | RTX PRO 4000 / RTX 5090 등 Blackwell 1장 (NVFP4 네이티브) |

인스턴스 시작 후:

1. 첫 부팅은 HF에서 가중치 ~16.5GB를 받으므로 회선에 따라 5~20분 소요.
   로그에서 `[entrypoint] vLLM ready` 확인.
2. 인스턴스 카드의 **포트 매핑 버튼**(IP:PORT 표시)에서 `8080` 매핑된 외부 주소로 접속 → Open WebUI.
   첫 접속 시 만드는 계정이 관리자가 된다.
3. API 직접 호출은 `8000` 매핑 주소:

```bash
curl http://<IP>:<mapped-8000>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "nvidia/diffusiongemma-26B-A4B-it-NVFP4", "messages": [{"role": "user", "content": "안녕"}]}'
```

## 3. 환경변수 (vast Docker Options에 `-e KEY=VALUE`로 추가)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_ID` | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` | 서빙할 HF 모델 |
| `MAX_MODEL_LEN` | `32768` | 컨텍스트 길이. 24GB에서 256K 전체는 불가 |
| `MAX_NUM_SEQS` | `4` | 동시 시퀀스 수 |
| `GPU_MEM_UTIL` | `0.92` | vLLM GPU 메모리 사용률 |
| `VLLM_EXTRA_ARGS` | (없음) | vllm serve 추가 인자 |
| `HF_TOKEN` | (없음) | gated 모델일 경우 HF 토큰 |

## 4. 로컬 빌드 (선택)

```bash
docker build -t vast-diffusiongemma .
docker run --gpus all -p 8080:8080 -p 8000:8000 vast-diffusiongemma
```

macOS에서는 빌드만 가능하고 실행은 NVIDIA GPU 필요.

## 참고 성능

diffusion 모델 특성상 256-토큰 블록 단위 denoising이라 짧은 응답/콜드 스타트는 느리고
(~16 tok/s), 캔버스가 채워진 warm 상태에서는 단일 스트림 ~140-160 tok/s 수준
(DGX Spark NVFP4 기준 벤치마크).
