# InsightFace Server

**언어:** [English](README.md) · [中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Русский](README.ru.md) · [Português](README.pt.md) · 한국어

> **GPU 한 장으로 50M+ 얼굴 벡터. INT8 특징 양자화 기반 초고속 검색을 실질적인 정확도 손실 없이 제공합니다.**

**Web UI, 직관적인 REST API, SQLite, 로컬 CPU 또는 NVIDIA GPU 추론을
하나의 컨테이너로 제공하는 셀프 호스팅 얼굴 인식 서버입니다.**

```text
이미지 업로드 -> 검출, 라이브니스 검사, 비교, 등록 또는 검색
```

> **모델 라이선스:** 공개 InsightFace 사전 학습 모델은 일반적으로 비상업적 연구
> 용도로만 제공됩니다. 상업적 사용에는
> [InsightFace](https://www.insightface.ai)의 별도 허가가 필요합니다.

InsightFace Server는 직접 관리하는 인프라에서 일반적인 얼굴 인식 워크플로를
실행하기 위한, AWS Rekognition보다 단순하고 개인정보 보호에 초점을 둔
대안입니다. 이미지, embedding, 모델, 인덱스를 자체 네트워크 안에 둘 수 있습니다.
AWS 호환 대체품은 아니며 SigV4, IAM, Region 또는 AWS 리소스 의미 체계를
구현하지 않습니다.

현재 릴리스: **0.3.1**, Linux x86_64.

| Runtime | Image |
| --- | --- |
| CPU | `ghcr.io/deepinsight/insightface-server:0.3.1-cpu` |
| NVIDIA GPU | `ghcr.io/deepinsight/insightface-server:0.3.1-cuda12` |

이동 태그 `cpu`와 `cuda12`는 각 Runtime 계열의 최신 안정 버전을 가리킵니다.
모호한 `latest` 태그는 사용하지 않습니다.
[Maintainer Guide — English](docs/maintainer-guide.md)의 릴리스 정책을
참고하세요.

**0.3.1 배포 변경:** Server와 모델 설치 도구를 root로 실행하고 모델을 하나의 쓰기
가능한 마운트로 통합하며 다운로드 시 addon 디렉터리를 자동 생성합니다. 호스트 UID/GID나
공유 그룹 설정이 필요 없습니다. 이미지 태그와 Compose를 함께 업데이트하고 기존 모델,
설정, 데이터를 보존하세요. 0.3.0부터 Raccoon 모델 설명 파일, 선택적 라이브니스와 BMP 입력을
지원하며 API/SDK의 `model_version`이 제거되었습니다.
[업그레이드 절차](docs/user-guide.ko.md#031으로-업그레이드)를 확인하세요.

![영문 InsightFace Server Dashboard](docs/images/customer/dashboard-en.jpg)

## 주요 기능

- SCRFD 얼굴 검출, 5점 landmark, 정렬, ArcFace embedding, L2
  normalization, 원본 cosine similarity, 정확한 1:N Person 검색.
- 여러 해상도의 후보를 합친 뒤 한 번의 NMS를 수행하며 `largest`와
  `center_largest` 단일 얼굴 선택을 지원.
- `Collection -> Person -> FaceSample`, 모델에 고정된 Collection,
  다중 이미지 등록의 부분 성공, metadata와 명확한 거절 사유.
- 등록 `review_mode`: `off`, `standard`, `strict`; 선택적인
  `external_trusted` 사전 계산 embedding.
- FP32, FP16, BF16, INT8 벡터 저장을 사용하는 정확한 GPU 검색.
- Dashboard, Collections, People, Detect, Compare, Search, RTSP 모니터링,
  System, Help를 제공하는 다국어 Web UI.
- `/v1` 아래 31개 snake_case REST operation, 보호되는
  `/v1/embeddings`, 가볍고 타입이 지정된 Python SDK.
- 제한된 메모리 이벤트, 여러 클라이언트, 선택적 `preview.mjpeg`를 지원하는
  서버 측 RTSP Monitor. 브라우저를 닫아도 모니터링은 중지되지 않습니다.
- SQLite를 영구 원본으로 사용하고, 재구축 가능한 메모리 정확 인덱스,
  하나의 쓰기 가능한 모델 마운트와 설정 디렉터리, 영구 `/data`, migration, health check, 조용한 CPU
  fallback을 금지하는 엄격한 CUDA 시작 검증 제공.
- JPEG, PNG, WebP, BMP 입력을 지원하며 원본 업로드는 기본적으로 보관하지 않음.

라이브니스는 기본적으로 꺼져 있습니다. `server/config/server.toml`에서 `inference.addons = []`, `addons.auto_download = []`가 기본값입니다. **시스템 → 라이브니스 검사**에서 모델을 다운로드하고 검증한 뒤 다음 시작 설정을 저장합니다. 수동으로 다시 시작해야 활성화되며 검증된 파일은 재사용합니다. 시작 시 다운로드하지 않으며, 활성 모델이 없으면 설치 방법을 안내하고 시작을 중단합니다. 평가한 얼굴은 핵심 필드 `status`, `is_live`, `live_score`를 반환하며 `input_rejected`에만 사용자용 설명인 `reason`이 추가됩니다. 기본 모드는 `normal`이며 등록 시 검사를 기본적으로 생략합니다(`liveness_on_registration = false`). [설정, Web 권한 및 업데이트 안내](docs/user-guide.ko.md#선택적-라이브니스-addon)를 참고하세요.

### RTX 5090 GPU 검색 성능

NVIDIA GeForce RTX 5090 한 장(32,607 MiB)에서 네이티브 CUDA exact-flat
인덱스는 INT8로 최대 **58.9M개의 512차원 이미지 벡터**를 저장했습니다.

| GPU 데이터 유형 | 최대 이미지 벡터 수 | 10M Top-5 p50 | 10M 직렬 QPS |
| --- | ---: | ---: | ---: |
| FP32 | 15.8M | 12.84 ms | 77.85 |
| FP16 | 30.7M | 6.83 ms | 146.32 |
| BF16 | 30.7M | 6.83 ms | 146.33 |
| INT8 | **58.9M** | **3.84 ms** | **260.81** |

INT8은 FP32 대비 실측 용량 3.73배, 10M Top-5 처리량 3.35배를 기록했습니다.
모든 값은 같은 RTX 5090, Driver 580.105.08, CUDA 12.9에서 측정한 GPU
결과입니다. 용량은 ONNX 모델과 Server 부하를 제외한 독립 네이티브 인덱스의
한계입니다. 속도는 정확히 10M 이미지 벡터, GPU 상주 Top-5 전체 정확 스캔,
동시 query 1개, warm-up 10회와 측정 100회 조건입니다. 검색은 각 저장 표현
내에서는 정확하지만 양자화로 인해 FP32 대비 score가 달라질 수 있습니다.
운영 환경에서는 모델, request, 동시성, 인덱스 재구축, allocator를 위한 VRAM
여유가 필요합니다.

### ICCV21-MFR 다인종 MR-ALL 정확도

[ICCV21-MFR](../challenges/iccv21-mfr/)의 다인종(MR) 테스트 세트에서 MR-ALL
전체 쌍 1:1 프로토콜과 FAR `1e-6` 조건으로 네이티브 검색 profile을
평가했습니다. 모든 profile은 Server API로 한 번만 추출하고 L2 normalization한
동일한 512차원 `buffalo_l` embedding을 사용했으며, 벡터 저장 및 검색 계산
표현만 변경했습니다.

| 검색 profile | FAR 1e-6의 MR-ALL | Cosine 임계값 | FP32 대비 차이 |
| --- | ---: | ---: | ---: |
| FP32 | 91.249107% | 0.407787 | — |
| FP16 | 91.249197% | 0.407787 | +0.000090%p |
| BF16 | 91.248502% | 0.407787 | -0.000605%p |
| **INT8** | **91.248005%** | **0.407739** | **-0.001102%p** |

**이 benchmark에서 INT8은 실질적인 정확도 손실이 없습니다.** challenge와
같이 소수점 둘째 자리까지 표시하면 FP32와 INT8 모두 **91.25% MR-ALL**이며,
반올림 전 차이도 0.0011%p에 불과합니다. 동시에 위에 제시한 실측 용량 3.73배와
10M Top-5 처리량 3.35배의 이점을 유지합니다. 이 비교는 벡터 저장 및 검색
정밀도를 측정한 것이며 INT8 모델 추론을 측정한 것이 아닙니다.

![영문 Collection 관리](docs/images/customer/collections-en.jpg)

![영문 RTSP Monitor, 비공개 주소 마스킹](docs/images/customer/monitoring-en.jpg)

## 빠른 시작

요구 사항:

- Docker Engine과 Docker Compose가 설치된 Linux x86_64;
- CUDA는 지원되는 NVIDIA GPU, NVIDIA Driver, NVIDIA Container Toolkit.

호스트에 Python, OpenCV, ONNX Runtime, CUDA Toolkit 또는 cuDNN을 설치할
필요가 없습니다. 공개 Image에는 모델, 고객 데이터, API Key, 운영 설정이
포함되지 않습니다.

전체 InsightFace 저장소 checkout에서 모델을 `server/.models`에 설치합니다.

저장소 루트에서 실행하고 `server/config/server.toml`을 준비하세요. Server와 모델 설치 도구는 root(`0:0`)로 실행됩니다. Compose는 `server/.models`가 없으면 생성하고 하나의 쓰기 가능한 `/models`로 마운트합니다. `addons/`는 addon 다운로드를 요청할 때 생성됩니다. UID/GID 내보내기, addon 디렉터리 수동 생성이나 권한 설정은 필요하지 않습니다.

```bash
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license
```

모델을 설치하면서 라이브니스도 설정하려면 대신 다음 설치 명령을 사용하세요.

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

Server를 먼저 실행할 필요는 없습니다. 새 배포는 다음 `up -d`에서 라이브니스가 활성화됩니다. 이미 실행 중이면 `docker compose -f server/deploy/compose.cpu.yml restart server`가 필요하며, `up -d`만으로는 저장된 설정을 다시 읽지 않습니다. CUDA는 `compose.cuda12.yml`을 사용하세요.

모델 도구는 `buffalo_l`, `buffalo_m`, `buffalo_s`, `buffalo_sc`,
`antelopev2`, `raccoon_s`, `raccoon_l`의 7개 패키지를 모두 지원합니다.
`manifest.json`과 서명된 `MODEL.LICENSE`를 기록하며 `models verify`로
검증할 수 있습니다. 모델 조건은 Server 소스 라이선스와 별개입니다.

CPU 시작:

```bash
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

대신 CUDA 12 시작:

```bash
docker compose -f server/deploy/compose.cuda12.yml pull
docker compose -f server/deploy/compose.cuda12.yml \
  run --rm models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cuda12.yml up -d
curl -fsS http://127.0.0.1:18098/v1/health
```

CPU는 `http://SERVER:18097/`, CUDA는 `http://SERVER:18098/`를 엽니다.
Collection을 만들고 한 장 이상의 사진으로 Person을 등록한 뒤 다른 사진으로
검색합니다. 데이터 volume을 유지하려면 `-v` 없이
`docker compose ... down`을 사용합니다.

제공되는 Compose는 격리 평가를 위해 인증을 기본적으로 끕니다. 다른 사용자나
네트워크에 공개하기 전에:

```bash
export INSIGHTFACE_AUTH_ENABLED=true
export INSIGHTFACE_API_KEY='충분히-긴-무작위-비밀로-교체'
docker compose -f server/deploy/compose.cpu.yml up -d
```

전체 첫 실행 절차는 [초보자 사용자 가이드](docs/user-guide.ko.md)를 참고하세요.

## 소스에서 빌드

완전한 로컬 소스 디렉터리에서 직접 빌드할 수 있습니다. 커밋하지 않은 변경 사항이
있거나 `.git` 디렉터리가 없어도 됩니다. Git 커밋이나 푸시는 빌드의 전제 조건이
아닙니다.

Dockerfile은 `server/`와 `python-package/insightface/`의 선택된 추론 모듈을
복사하므로 완전한 소스 디렉터리가 build context입니다.

CPU:

```bash
make -C server build-cpu
docker compose -f server/deploy/compose.cpu.yml \
  run --rm --pull never models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cpu.yml \
  up -d --no-build --pull never
```

CUDA 12:

```bash
make -C server build-cuda12
docker compose -f server/deploy/compose.cuda12.yml \
  run --rm --pull never models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cuda12.yml \
  up -d --no-build --pull never
```

`--pull never`는 로컬 빌드 Image 사용을 보장합니다. 빌드 중에는 고정된 base
Image와 의존성을, 모델 설치 중에는 라이선스를 수락한 모델 package를 별도로
다운로드합니다.

## 핵심 동작

- Similarity는 확률이 아닌 원본 cosine 값입니다. Threshold는 `0.0..1.0`,
  기본값은 `0.4`입니다.
- Collection은 모델과 embedding contract에 고정됩니다. 불일치 시에도 보이지만
  등록/검색은 `collection_model_mismatch`를 반환합니다.
- 시작 Detection Profile은 새 Collection에 복사되고, 이후 Profile은 다음
  요청부터 독립적으로 변경할 수 있습니다.
- 선택적 얼굴 저장은 112x112로 조정된 bounding-box JPEG crop이며 원본이나
  인식용 정렬 입력이 아닙니다. 기본적으로 꺼져 있습니다.
- SQLite commit이 원본입니다. 등록/삭제 성공 응답 전에 인덱스를 동기화하고
  재시작 후 SQLite에서 다시 구축합니다.
- 응답은 `x-request-id`를 포함하며 목록은 불투명한 서명 cursor를 사용합니다.

정확한 field, default, lifecycle, 오류 동작은 아래 상세 문서에서만 관리합니다.

## API와 SDK

주요 그룹:

- 시스템: `/v1/health`, `/v1/system`, `/v1/models`;
- 무상태 얼굴: `/v1/detect`, `/v1/compare`, `/v1/embeddings`;
- Collection, Person, FaceSample CRUD;
- Collection Person 검색;
- RTSP Monitor 설정, 상태, 이벤트, 미리보기.

모든 parameter, response, error, example은
[REST API 가이드](docs/api.ko.md)를 참고하세요. 대화형 OpenAPI는 `/docs`에
유지됩니다.

```python
from insightface_server import Client

with Client("http://localhost:18097", api_key=None) as client:
    faces = client.detect("photo.jpg")
    matches = client.search("employees", "unknown.jpg", limit=5)
```

SDK 설치, 입력 형식, 메서드와 전체 흐름은
[사용자 가이드](docs/user-guide.ko.md)를 참고하세요.

## 보안

얼굴 이미지와 embedding은 생체 데이터입니다. 네트워크 배포에서는 인증을 켜고,
신뢰하는 reverse proxy에서 HTTPS를 종료하며, Docker와 volume 접근을 제한하고,
광범위한 CORS를 끈 상태로 유지하고, backup, 보존, 삭제, 동의, incident response
정책을 정의하세요. 이미지, embedding, RTSP credential, API Key를 로그에
남기지 마세요.

Server는 TLS, 사용자 계정, RBAC, cloud IAM 또는 법적 준수 계층을 내장하지
않습니다. 운영과 보안은 [사용자 가이드](docs/user-guide.ko.md)를 참고하세요.

## 1단계 범위

AWS/CompreFace 호환, CUDA 11, Jetson, ARM64, Windows Container, TensorRT,
Kubernetes, 분산 Worker, 영구 Monitor 이벤트 또는 녹화/NVR,
deepfake, 인구통계 속성은 구현하지 않습니다.

## 문서

- [사용자 가이드](docs/user-guide.ko.md) — 설치, 설정, 모델, Web UI, SDK,
  GPU, 보안, 백업과 문제 해결.
- [REST API 가이드](docs/api.ko.md) — 모든 endpoint, 필드, 동작, 결과, 오류,
  pagination과 예시.
- [Maintainer Guide — English](docs/maintainer-guide.md) — 아키텍처, 검색 내부,
  테스트, 기여 규칙과 컨테이너 릴리스.

GitHub와 Web UI 도움말은 동일한 현지화 Markdown을 읽으며 표시 방식만 다릅니다.

## 라이선스

단일 라이선스 진입점은 [LICENSING.md](LICENSING.md)입니다. Server 소스와
Python SDK는 MIT License이지만 이 선언은 모델 파일, 모델 가중치, dataset,
타사 컴포넌트에는 적용되지 않습니다. 공개 InsightFace 모델은 별도 허가가
없다면 일반적으로 비상업적 연구로 제한됩니다. 상업 라이선스:
<https://www.insightface.ai>.
