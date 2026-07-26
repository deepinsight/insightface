# InsightFace Server 사용자 가이드

**언어:** [English](user-guide.md) · [中文](user-guide.zh-CN.md) · [日本語](user-guide.ja.md) · [Deutsch](user-guide.de.md) · [Español](user-guide.es.md) · [Français](user-guide.fr.md) · [Русский](user-guide.ru.md) · [Português](user-guide.pt.md) · 한국어

이 가이드는 처음 사용하는 사용자가 빈 작업 폴더에서 시작해 첫 검색에 성공할 때까지 단계별로 설명합니다. 같은 기능을 Web UI, `/v1` API, Python SDK에서 사용할 수 있습니다. 모든 HTTP 필드와 결과는 [API 사용 가이드](api.ko.md)를 확인하세요.

## 처음부터 첫 검색까지

CPU 버전에는 Linux x86_64, Docker Engine, Docker Compose가 필요합니다. CUDA 버전에는 호환 NVIDIA Driver와 NVIDIA Container Toolkit도 필요하지만 호스트 CUDA, cuDNN, ORT, Python, OpenCV는 설치하지 않아도 됩니다.

```bash
mkdir -p server/.models
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml run --rm models install buffalo_l
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

GPU는 `compose.cuda12.yml`과 포트 `18098`을 사용합니다. 설치 전 모델 라이선스가 표시되며, 별도 상용 라이선스가 없으면 공개 InsightFace 모델은 비상업 연구용으로만 사용할 수 있습니다.

제공되는 Compose는 격리된 평가 환경을 위해 인증이 기본적으로 꺼져 있습니다. 네트워크에 공개하기 전 `INSIGHTFACE_AUTH_ENABLED=true`와 긴 `INSIGHTFACE_API_KEY`를 설정하세요. Dashboard 확인, Collection 생성, Person 등록, 다른 이미지로 Search 순서로 진행합니다. 데이터 볼륨을 보존하려면 `docker compose ... down`에 `-v`를 붙이지 마세요.

## 1. 로그인과 준비 상태

CPU는 `http://SERVER:18097/`, CUDA 12는 `http://SERVER:18098/`을 엽니다. 인증이 켜져 있으면 **API 키 설정**에서 운영자가 제공한 Key를 입력하고 현재 탭에 적용합니다. Key는 탭 메모리에만 있으며 새로고침하거나 닫으면 삭제됩니다.

**대시보드** 또는 **시스템**에서 서비스, 데이터베이스, 모델, Provider가 준비되었는지 확인합니다. CUDA는 `CUDAExecutionProvider`를 표시해야 하며 CPU로 조용히 전환하지 않습니다.

## 2. Collection 만들기

**컬렉션** → **새 컬렉션**에서 안정적인 ID, 이름, cosine 임계값(초기 `0.4`),
사용 가능한 검색 프로필, 용량, 사람별 최대 FaceSample 수를 설정합니다. 112×112로
조정한 `bounding-box crop` JPEG 저장은 기본적으로 꺼져 있으며, 인식 모델의 정렬
입력이 아닙니다.

Collection은 모델 ID, 버전, digest, 차원, 전처리에 고정됩니다. 모델을 바꿔도 이전 Collection은 보이지만 계약이 다르면 등록과 검색이 명시적으로 거부됩니다.

검출 프로필은 Collection 생성 시 시스템 값을 복사하며 이후 입력 크기, 검출/NMS 임계값, 단일 얼굴 전략을 변경할 수 있습니다. `largest`는 면적을 우선하고, `center_largest`는 `면적 - 2.0 × 얼굴 상자 중심과 이미지 중심 사이의 픽셀 거리 제곱`을 최대화합니다. 검출 신뢰도는 이 점수에 포함되지 않습니다.

## 3. Person 등록

**사람**에서 Collection을 선택하고 **사람 등록**을 엽니다. ID, 이름, 외부 ID, JSON metadata와 여러 JPEG, PNG 또는 WebP를 지정할 수 있습니다.

- `off`: Collection의 단일 얼굴 전략을 사용하며 여러 얼굴을 허용합니다.
- `standard`: 하나의 사용 가능한 얼굴과 크기, 검출, 선명도, 밝기, 자세 검사를 요구합니다.
- `strict`: standard에 더해 최상의 클래스 내 similarity가 최상의 클래스 외 similarity보다 커야 합니다.

일괄 등록은 부분 성공과 각 거부 이유를 반환합니다. 원본은 저장하지 않습니다. `external_trusted`는 L2 정규화된 embedding을 받으며 이미지로 검출과 품질은 검사하지만 특징을 다시 추출하지 않습니다.

## 4. 검출, 비교, 검색

**검출**은 상자, 5개 점, 검출 점수, 품질을 표시하며 얼굴 없음은 정상적인 빈 목록입니다. **비교**는 시스템 또는 Collection 프로필로 각 이미지에서 한 얼굴을 선택하고 cosine `similarity`, `threshold`, `matched`를 반환합니다. 유사도는 확률이 아닙니다.

**검색**에서 Collection과 이미지를 선택합니다. 한 사람의 점수는 모든 FaceSample 중 최고 similarity입니다. 결과는 내림차순이며 일치 없음은 빈 목록입니다. 새 샘플은 SQLite에 commit된 다음 성공 응답 전에 메모리 인덱스에 추가됩니다. 재시작 시 SQLite에서 재구축합니다.

## 5. RTSP 카메라 모니터링

**카메라 모니터링**에서 영구 Monitor를 만들고 RTSP source, Collection, 추론 속도, 선택 임계값과 이벤트 정책을 설정합니다. 미리보기는 기본으로 꺼져 있으며 인식과 이벤트는 계속 작동합니다. 켜면 Web UI가 원본 프레임 위에 `/state` 결과로 등록 인물은 초록색, 미등록 얼굴은 주황색 상자를 그립니다.

Monitor는 브라우저와 독립적으로 실행되고 활성 작업은 서버 재시작 후 복원됩니다. 설정은 SQLite에, RTSP 자격증명은 `/data`에 암호화 저장되지만 영상 프레임과 이벤트는 저장하지 않습니다. 이벤트는 제한된 메모리 버퍼에만 남습니다. 디코더는 최신 프레임만 보관하고 오래된 프레임을 쌓지 않고 건너뜁니다.

## 6. 데이터와 보안

`/data`를 영속화하고 `/models`는 읽기 전용으로 마운트합니다. 대량 작업 전 SQLite와 face crop 영역을 함께 백업하세요. Key는 hash로 저장되며 같은 volume을 다른 `INSIGHTFACE_API_KEY`로 시작하면 활성 Key가 교체됩니다. 이미지, embedding, Key를 로그에 남기지 마세요.

개발자용 OpenAPI 스키마 탐색기는 `/docs`에 있으며 작업 중심 API 안내는 이 도움말에 있습니다. 문제 보고 시 `x-request-id`를 포함하세요. `401`은 Key, `409 collection_model_mismatch`는 모델 계약, `422 face_not_found`는 사용 가능한 얼굴을 확인합니다.

## 7. 모델과 라이선스

이미지에는 모델이 포함되지 않습니다. 일반 Server 시작은 오프라인이며, 일회성
`models` 서비스가 `server/.models`에 설치합니다.

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models verify buffalo_l
```

지원 패키지는 `buffalo_l`(`det_10g.onnx` + `w600k_r50.onnx`),
`buffalo_m`, `buffalo_sc`, `antelopev2`입니다. 설치하면 `manifest.json`과
서명된 `MODEL.LICENSE`가 생성됩니다. `--accept-license`가 없으면 조건만
표시하고 다운로드하지 않습니다. 공개 InsightFace 사전 학습 모델은 별도 상업용
라이선스가 없는 경우 비상업 연구용입니다.

## 8. 시작 설정과 검색

`server/config/server.toml`은 시작 시 한 번만 읽으며 변경 후 재시작해야 합니다.
기본값은 `input_sizes=[[96,96],[512,512]]`, 검출 threshold `0.50`, NMS `0.40`,
`single_face_selection="largest"`, 최대 100개 얼굴입니다. SCRFD는 각 해상도를
실행하고 원본 좌표로 합친 모든 후보에 전역 NMS를 한 번 수행합니다.
`max_concurrency="auto"`는 CPU 4, CUDA 8입니다.
`[web].disabled=true`이면 `/v1`과 `/openapi.json`만 제공합니다.

System은 현재 환경에서 사용 가능한 Profile만 표시합니다. Collection 생성 후 고정되며
Search 요청별로 바꿀 수 없습니다.

- `fp32_v1`: CPU/CUDA 표준
- `fp16_v1`: CUDA
- `bf16_v1`: 지원 CPU 또는 SM80+ CUDA
- `int8_x736_v1`: CPU/CUDA 권장 INT8, INT32 누적
- `int8_x1000_v1`: 기존 Collection 호환

모든 Profile은 전체 FaceSample을 순회하는 Flat 검색이며 ANN이 아닙니다. 공개 score는
raw cosine입니다. `capacity_rows=100000`, 배포 한도 `10000000`,
`max_faces_per_person=20`이 기본입니다. 512차원 순수 벡터는 행당 FP32 2,048 byte,
FP16/BF16 1,024 byte, INT8 512 byte 정도입니다.

## 9. SDK, 빌드와 데이터 운영

Python SDK는 경로, bytes, file-like object를 지원하고 Detect, Compare,
Collections, 등록, Search, Monitors의 타입 지정 메서드를 제공합니다. 전체 HTTP
계약은 [API 사용 가이드](api.ko.md)를 확인하세요.

전체 저장소에서 사용자가 직접 이미지를 빌드할 수 있습니다.

```bash
make -C server build-cpu
make -C server build-cuda12
```

로컬 이미지를 쓸 때 Compose에 `--pull never`를 추가합니다. 고정 Tag는
`0.2.0-cpu`, `0.2.0-cuda12`이고 이동 Tag `cpu`, `cuda12`는 최신 안정 버전을
가리키며 `latest`는 없습니다. 업그레이드 전 쓰기를 중지하고 `/data`와 crop을
SQLite-safe 방식으로 백업하세요. `docker compose down -v`는 데이터 Volume을
삭제하므로 사용하지 마세요.

## 10. GPU, 네트워크와 문제 해결

CUDA 이미지는 CUDA Runtime 12.9.1, cuDNN 9.24.0,
`onnxruntime-gpu==1.27.0`을 포함합니다. Turing/Ampere/Ada/Hopper는 R535+,
Blackwell/RTX 50은 570.26+가 필요하며 신규 배포는 안정적인 R580+를 권장합니다.
시작 시 GPU, Compute Capability, Driver, CUDA/cuDNN/ORT, Provider, 실제 모델
Session과 warm-up을 검사하며 CPU로 조용히 fallback하지 않습니다.

네트워크에 공개할 때 신뢰할 수 있는 Reverse Proxy에서 HTTPS를 종료하고 CORS origin,
rate, body, timeout을 제한하며 `/data`와 백업을 생체 데이터로 보호하세요. 이미지,
embedding, Key를 로그에 남기지 마세요. 1단계는 역할 없는 단일 API Key만 지원하며
multi-tenant 권한 시스템이 아닙니다.
