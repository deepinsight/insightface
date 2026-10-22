# InsightFace Server 사용자 가이드

**언어:** [English](user-guide.md) · [中文](user-guide.zh-CN.md) · [日本語](user-guide.ja.md) · [Deutsch](user-guide.de.md) · [Español](user-guide.es.md) · [Français](user-guide.fr.md) · [Русский](user-guide.ru.md) · [Português](user-guide.pt.md) · 한국어

이 가이드는 처음 사용하는 사용자가 빈 작업 폴더에서 시작해 첫 검색에 성공할 때까지 단계별로 설명합니다. 같은 기능을 Web UI, `/v1` API, Python SDK에서 사용할 수 있습니다. 모든 HTTP 필드와 결과는 [API 사용 가이드](api.ko.md)를 확인하세요.

모델은 `model_id`로 식별하며 응답에 별도의 `model_version`을 포함하지 않습니다.

동일한 인식 모델과 특징 계약으로 Server를 업그레이드하면 기존 Collection의 `embedding_contract_id`, 샘플과 embedding이 유지됩니다. 모델 변경은 별도 마이그레이션이며 계약이 다르면 등록과 검색에서 `collection_model_mismatch`를 반환합니다.

라이브니스를 사용하려면 [설정, 모델 설치와 결과 설명](#선택적-라이브니스-addon)을 확인하세요. 각 작업 절에서도 해당 동작에 미치는 영향을 설명합니다.

## 처음부터 첫 검색까지

CPU 버전에는 Linux x86_64, Docker Engine, Docker Compose가 필요합니다. CUDA 버전에는 호환 NVIDIA Driver와 NVIDIA Container Toolkit도 필요하지만 호스트 CUDA, cuDNN, ORT, Python, OpenCV는 설치하지 않아도 됩니다.

저장소 루트에서 실행하고 `server/config/server.toml`을 준비하세요. Server와 모델 설치 도구는 root(`0:0`)로 실행됩니다. Compose는 `server/.models`가 없으면 생성하고 하나의 쓰기 가능한 `/models`로 마운트합니다. `addons/`는 addon 다운로드를 요청할 때 생성됩니다. UID/GID 내보내기, addon 디렉터리 수동 생성이나 권한 설정은 필요하지 않습니다.

```bash
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml run --rm models install buffalo_l
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

모델을 설치하면서 라이브니스도 설정하려면 대신 다음 설치 명령을 사용하세요.

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

Server를 먼저 실행할 필요는 없습니다. 새 배포는 다음 `up -d`에서 라이브니스가 활성화됩니다. 이미 실행 중이면 `docker compose -f server/deploy/compose.cpu.yml restart server`가 필요하며, `up -d`만으로는 저장된 설정을 다시 읽지 않습니다. CUDA는 `compose.cuda12.yml`을 사용하세요.

GPU는 `compose.cuda12.yml`과 포트 `18098`을 사용합니다. 설치 전 모델 라이선스가 표시되며, 별도 상용 라이선스가 없으면 공개 InsightFace 모델은 비상업 연구용으로만 사용할 수 있습니다.

제공되는 Compose는 격리된 평가 환경을 위해 인증이 기본적으로 꺼져 있습니다. 네트워크에 공개하기 전 `INSIGHTFACE_AUTH_ENABLED=true`와 긴 `INSIGHTFACE_API_KEY`를 설정하세요. Dashboard 확인, Collection 생성, Person 등록, 다른 이미지로 Search 순서로 진행합니다. 데이터 볼륨을 보존하려면 `docker compose ... down`에 `-v`를 붙이지 마세요.

## 선택적 라이브니스 addon

`server/config/server.toml`의 라이브니스는 기본적으로 꺼져 있습니다. `inference.addons`와 `addons.auto_download`는 모두 `[]`이며 키가 없는 이전 설정도 비활성 상태를 유지합니다.

**명령줄에서 활성화하기: Server를 처음 시작하기 전에도 가능합니다.**

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

`--enable-liveness`는 먼저 기존 설정을 업데이트할 수 있는지 검사합니다. 기본 패키지, 설치 설정에 지정된 addons와 라이브니스 모델을 설치·검증한 뒤 `inference.addons`와 `addons.auto_download`에 `liveness`를 추가하고 다른 항목, 주석과 설정을 보존합니다. 검증된 캐시를 재사용해도 활성화 설정을 저장합니다. 다운로드 실패 시 설정은 바뀌지 않으며, 설정 저장 실패는 명확한 오류와 0이 아닌 종료 상태를 반환합니다. 유효한 캐시는 재시도 시 재사용할 수 있습니다.

두 Compose 서비스는 기존 `server/config` 전체를 `/etc/insightface`에 쓰기 가능하게 마운트하고 `create_host_path: false`를 유지합니다. 따라서 Server가 실행되지 않아도 설치 도구가 호스트 설정을 원자적으로 갱신할 수 있습니다. 디렉터리와 `server.toml`은 있어야 합니다.

Server를 먼저 실행할 필요는 없습니다. 새 배포는 다음 `up -d`에서 라이브니스가 활성화됩니다. 이미 실행 중이면 `docker compose -f server/deploy/compose.cpu.yml restart server`가 필요하며, `up -d`만으로는 저장된 설정을 다시 읽지 않습니다. CUDA는 `compose.cuda12.yml`을 사용하세요.

`--enable-liveness`를 지정하지 않으면 `models install`은 기존 동작을 유지하고 설정을 쓰지 않으며 기본 라이브니스는 꺼져 있습니다. `models addons install liveness`는 다운로드와 검증만 하며 활성화하지 않습니다. 아래의 **시스템 → 라이브니스 검사**에서도 활성화할 수 있습니다.

**시스템 → 라이브니스 검사**에서 **다운로드하고 다시 시작 후 활성화**를 선택하세요. SHA-256 검증 후 두 목록에 `liveness`를 추가하고 다른 항목, 주석과 설정은 보존합니다. 검증된 파일은 재사용합니다. 현재 실행 상태는 그대로이며 **Server를 수동으로 다시 시작**해야 적용됩니다. 오류 시 재시도할 수 있고 다운로드 실패로 활성화되지 않습니다.

시스템은 검증된 설치 상태(`installed`), 현재 실행 상태(`enabled`), 다음 시작에 적용할 저장 설정(`configured_enabled`), 재시작 필요 여부(`restart_required`)를 따로 표시합니다. 다운로드하거나 저장해도 현재 추론은 바뀌지 않습니다. 끄려면 같은 파일에 `inference.addons=[]`와 `addons.auto_download=[]`를 저장하고 수동으로 다시 시작하세요. Web 작업은 등록 설정을 바꾸지 않으며 기본값은 `liveness_on_registration=false`입니다.

**고급 사용: 수동 설정.** 다음 설정은 활성화 플래그나 Web 작업의 대안입니다. 다시 시작하기 전에 모델을 설치하세요.

```toml
[inference]
addons = ["liveness"]
liveness_mode = "normal"
liveness_threshold = 0.8
liveness_compare_scope = "both"
liveness_on_registration = false

[addons]
auto_download = ["liveness"]
```

### 모델 설치와 시작

`inference.addons`는 실행을, `addons.auto_download`는 기본 패키지 설치 시 추가 다운로드를 제어합니다. 후자를 `["liveness"]`로 설정하면 캐시된 기본 패키지에도 addon을 설치합니다. Server 시작 시 다운로드하지 않습니다. 설치 도구와 Server는 같은 파일을 읽습니다.

```bash
docker compose -f server/deploy/compose.cpu.yml run --rm models addons install liveness
docker compose -f server/deploy/compose.cpu.yml run --rm models addons verify liveness
docker compose -f server/deploy/compose.cpu.yml up -d --force-recreate
```

활성 모델이 없으면 `addon_model_missing`, 잘못된 파일이면 `addon_model_invalid`로 시작을 중단합니다. 설정된 addon을 자동으로 끄지 않습니다.

### Web 다운로드를 위한 마운트와 권한

표준 Compose는 Server와 모델 설치 도구에 하나의 쓰기 가능한 `/models` 마운트를
제공하고 `create_host_path: true`를 설정합니다. 두 서비스는 root(`0:0`)와 Docker 기본
capabilities로 실행되며 `cap_drop: [ALL]`은 설정하지 않습니다. 컨테이너 루트 파일시스템은
읽기 전용으로 유지하고 `no-new-privileges`도 유지합니다. 호스트 UID/GID나 `chmod 777`
설정은 필요하지 않습니다. root는 쓰기 가능한 마운트의 파일을 변경할 수 있으며 새로 다운로드한
파일의 호스트 소유자가 root가 될 수 있습니다.

두 서비스 모두 기존 `server/config` 전체를 `/etc/insightface`에 쓰기 가능하게 마운트하여
Web 작업과 `--enable-liveness`가 `server.toml`을 원자적으로 저장합니다. 이 디렉터리와 파일은
있어야 하며 Compose는 설정 원본을 자동 생성하지 않습니다. 사용자 지정 배포는 실제 경로를
사용하고 두 서비스 모두에 동일한 쓰기 가능한 디렉터리 마운트를 제공하세요. 사용자 지정 읽기 전용
마운트는 기존 모델 추론에는 사용할 수 있지만 Web 다운로드나 설정 저장에는 사용할 수 없습니다.

CUDA는 `compose.cuda12.yml`을 사용합니다. 파일이 있다고 라이브니스가 자동으로 켜지지는
않습니다. Web 저장 후 `docker compose -f server/deploy/compose.cpu.yml restart server`로
적용하세요. 마운트, 실행 사용자, capabilities나 프록시 환경 변수를 바꾸면 컨테이너를 다시
생성해야 합니다.

다운로드에 프록시가 필요하면 컨테이너 생성 전에 `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`를 설정하세요. Compose가 Server와 모델 도구 모두에 전달합니다. 컨테이너에서 접근 가능한 LAN 주소를 사용하세요. 컨테이너의 `127.0.0.1`은 Mac이 아닙니다. 이 작업은 기존 API Key 인증을 사용하며, 인증이 꺼져 있으면 API에 접근 가능한 사용자도 실행할 수 있습니다. 고정된 공개 라이브니스 모델만 다운로드하며 임의 URL 입력이나 기본 모델 패키지 전환은 제공하지 않습니다.

### 라이브니스 결과

| 결과 | `status` | `is_live` | `live_score` |
| --- | --- | --- | --- |
| 검사 통과 | `ok` | `true` | `[0, 1]` |
| 비생체 | `ok` | `false` | `[0, 1]` |
| 입력 거부 | `input_rejected` | `null` | `null` |

정렬된 얼굴 주변에서 원본 이미지의 유효 영역이 부족한 경우에만 `input_rejected`를 반환합니다. 이 결과에는 사용자용 설명인 `liveness.reason`이 추가되며, 실제 얼굴과 위조 결과에는 `reason`이 없습니다. FaceAnalysis와 API는 항상 영어 설명을 반환하고 Web UI만 표시 언어에 맞게 번역합니다. 프로그램에서는 `reason` 문구를 해석하지 말고 `status`와 `is_live`로 판단하세요. 이전에 저장된 결과에는 `reason`이 없을 수 있으므로 클라이언트는 일반적인 입력 거부 안내로 대체할 수 있습니다.

```json
{
  "status": "input_rejected",
  "is_live": null,
  "live_score": null,
  "reason": "Insufficient image area around the face for liveness detection. Move the face toward the center, step back from the camera, or use a less tightly cropped image."
}
```

`normal`은 라이브니스를 통과한 얼굴만 인식합니다. `observe`는 결과를 기록하고 인식을 계속합니다. 평가하지 않은 얼굴에는 `liveness`가 없습니다. 핵심 필드는 `status`, `is_live`, `live_score` 세 개입니다. 통과/fake는 `status: ok`, 불리언, 점수를 반환하고, 입력 거부는 `status: input_rejected`와 두 개의 `null`을 반환합니다.

Detect는 음성 결과도 HTTP 200으로 반환합니다. `normal`에서 임베딩·비교·검색은 HTTP 422 `liveness_fake` 또는 `liveness_input_rejected`와 `error.details.liveness`를 반환하며 비교에는 `details.side`가 추가됩니다. 추론 장애는 HTTP 503 `liveness_unavailable`입니다. 실행 오류는 `normal`과 `observe` 모두에서 작업을 중단하며 `input_rejected`로 변환하지 않습니다.

새 Person 등록과 FaceSample 추가는 기본적으로 라이브니스를 건너뜁니다. `[inference].liveness_on_registration=false`이면 모델을 실행하지 않고 새 샘플에 `liveness`를 포함하지 않습니다. `true`이고 addon이 활성화되어 있으면 `normal`/`observe` 정책을 적용하며 거부 항목에는 `reason`과 `liveness`가 포함됩니다. `review_mode` 품질 심사와 외부 임베딩 검증은 계속 수행합니다. 등록 검사가 활성화된 경우 `review_mode=off`와 `external_trusted`도 우회할 수 없습니다. 요청에서 이 시작 설정을 덮어쓸 수 없습니다. 이전에 저장한 결과는 계속 조회할 수 있습니다.

RTSP는 `liveness_blocked`를 `unknown`과 구분하고 `liveness_blocked_faces`로 집계합니다. 차단된 얼굴은 사람/미확인 입장 이벤트를 생성하지 않고 확인 프레임 수를 초기화합니다. 추론 장애 시 이전에 표시된 신원을 지웁니다.

`liveness_compare_scope`는 `/v1/compare`에서 `both`(기본값), `source`, `target` 중 검사 대상을 선택합니다. `live_score >= liveness_threshold`이면 통과입니다.

모델은 호스트의 `server/.models/addons/liveness.onnx`, 컨테이너의 `/models/addons/liveness.onnx`에 저장됩니다. `/v1/models`와 `/v1/system`의 `addons`는 현재 활성 addon을 표시합니다.

[전체 API 계약](api.ko.md#선택적-라이브니스-addon).

## 1. 로그인과 준비 상태

CPU는 `http://SERVER:18097/`, CUDA 12는 `http://SERVER:18098/`을 엽니다. 인증이 켜져 있으면 **API 키 설정**에서 운영자가 제공한 Key를 입력하고 현재 탭에 적용합니다. Key는 탭 메모리에만 있으며 새로고침하거나 닫으면 삭제됩니다.

**대시보드** 또는 **시스템**에서 서비스, 데이터베이스, 모델, Provider가 준비되었는지 확인합니다. CUDA는 `CUDAExecutionProvider`를 표시해야 하며 CPU로 조용히 전환하지 않습니다.

대시보드는 모델 이름 아래에 라이브니스 활성 또는 비활성을 항상 표시합니다. 시스템은 설치 여부, 현재 실행 상태, 다시 시작 대기를 구분합니다.

## 2. Collection 만들기

**컬렉션** → **새 컬렉션**에서 안정적인 ID, 이름, cosine 임계값(초기 `0.4`),
사용 가능한 검색 프로필, 용량, 사람별 최대 FaceSample 수를 설정합니다. 112×112로
조정한 `bounding-box crop` JPEG 저장은 기본적으로 꺼져 있으며, 인식 모델의 정렬
입력이 아닙니다.

Collection은 모델 ID, digest, 차원, 전처리에 고정됩니다. 모델을 바꿔도 이전 Collection은 보이지만 계약이 다르면 등록과 검색이 명시적으로 거부됩니다.

검출 프로필은 Collection 생성 시 시스템 값을 복사하며 이후 입력 크기, 검출/NMS 임계값, 단일 얼굴 전략을 변경할 수 있습니다. `largest`는 면적을 우선하고, `center_largest`는 `면적 - 2.0 × 얼굴 상자 중심과 이미지 중심 사이의 픽셀 거리 제곱`을 최대화합니다. 검출 신뢰도는 이 점수에 포함되지 않습니다.

## 3. Person 등록

**사람**에서 Collection을 선택하고 **사람 등록**을 엽니다. ID, 이름, 외부 ID, JSON metadata와 여러 JPEG, PNG, WebP 또는 BMP를 지정할 수 있습니다.

- `off`: Collection의 단일 얼굴 전략을 사용하며 여러 얼굴을 허용합니다.
- `standard`: 하나의 사용 가능한 얼굴과 크기, 검출, 선명도, 밝기, 자세 검사를 요구합니다.
- `strict`: standard에 더해 최상의 클래스 내 similarity가 최상의 클래스 외 similarity보다 커야 합니다.

일괄 등록은 부분 성공과 각 거부 이유를 반환합니다. 원본은 저장하지 않습니다. `external_trusted`는 L2 정규화된 embedding을 받으며 이미지로 검출과 품질은 검사하지만 특징을 다시 추출하지 않습니다.

Person 생성과 FaceSample 추가는 기본적으로 라이브니스를 건너뜁니다(`liveness_on_registration=false`). 활성화하면 `normal`은 fake/부적합 입력을 거부하고, `observe`는 결과를 저장하며 계속합니다. 품질 검토는 선택한 `review_mode`에 따라 적용됩니다. 거부 목록은 실제 `reason`과 라이브니스 결과를 따로 표시합니다.

## 4. 검출, 비교, 검색

**검출**은 상자, 5개 점, 검출 점수, 품질을 표시하며 얼굴 없음은 정상적인 빈 목록입니다. **비교**는 시스템 또는 Collection 프로필로 각 이미지에서 한 얼굴을 선택하고 cosine `similarity`, `threshold`, `matched`를 반환합니다. 유사도는 확률이 아닙니다.

**검색**에서 Collection과 이미지를 선택합니다. 한 사람의 점수는 모든 FaceSample 중 최고 similarity입니다. 결과는 내림차순이며 일치 없음은 빈 목록입니다. 새 샘플은 SQLite에 commit된 다음 성공 응답 전에 메모리 인덱스에 추가됩니다. 재시작 시 SQLite에서 재구축합니다.

평가한 얼굴에는 `liveness.status`, `liveness.is_live`, `liveness.live_score`가 포함됩니다. Fake와 `input_rejected`도 HTTP 200이며 인식 특징은 추출하지 않습니다. `input_rejected`는 얼굴 주변의 이미지 영역이 부족함을 뜻하며 `liveness.reason`에 이미지 조정 방법이 제공됩니다. `liveness`가 없으면 평가하지 않은 것입니다.

`liveness_compare_scope`(`both`, `source`, `target`)가 인식 전에 검사할 쪽을 정합니다. `normal`에서 거부되면 HTTP 422 `liveness_fake` / `liveness_input_rejected`, `error.details.liveness`, `error.details.side`를 반환하고 유사도는 반환하지 않습니다. `observe`는 비교를 계속하고 검사한 얼굴에 결과를 포함합니다.

라이브니스가 켜진 `normal`에서는 fake/부적합 쿼리에 HTTP 422 `liveness_fake` / `liveness_input_rejected`와 `error.details.liveness`를 반환하며 검색하지 않습니다. 이는 검색 성공 후 빈 일치 목록과 다릅니다. `observe`는 검색을 계속하고 쿼리 얼굴에 결과를 반환합니다.

## 5. RTSP 카메라 모니터링

**카메라 모니터링**에서 영구 Monitor를 만들고 RTSP source, Collection, 추론 속도, 선택 임계값과 이벤트 정책을 설정합니다. 미리보기는 기본으로 꺼져 있으며 인식과 이벤트는 계속 작동합니다. 켜면 Web UI가 원본 프레임 위에 `/state` 결과로 등록 인물은 초록색, 미등록 얼굴은 주황색 상자를 그립니다.

Monitor는 브라우저와 독립적으로 실행되고 활성 작업은 서버 재시작 후 복원됩니다. 설정은 SQLite에, RTSP 자격증명은 `/data`에 암호화 저장되지만 영상 프레임과 이벤트는 저장하지 않습니다. 이벤트는 제한된 메모리 버퍼에만 남습니다. 디코더는 최신 프레임만 보관하고 오래된 프레임을 쌓지 않고 건너뜁니다.

라이브니스가 켜진 `normal`에서 차단된 얼굴은 `status: liveness_blocked`와 별도의 결과를 가집니다. `unknown_faces` 대신 `liveness_blocked_faces`로 집계되며 입장 이벤트를 생성하지 않습니다. `observe`는 인식을 계속합니다. 입력 거부와 fake는 구분해서 표시합니다.

## 6. 데이터와 보안

`/data`, 쓰기 가능한 모델 루트와 설정 디렉터리를 영속화합니다. 컨테이너 루트 파일시스템은 읽기 전용이며 Web 모델 관리는 기존 API 인증을 적용합니다. 대량 작업 전 SQLite와 face crop 영역을 함께 백업하세요. Key는 hash로 저장되며 같은 volume을 다른 `INSIGHTFACE_API_KEY`로 시작하면 활성 Key가 교체됩니다. 이미지, embedding, Key를 로그에 남기지 마세요.

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
`buffalo_m`, `buffalo_s`, `buffalo_sc`, `antelopev2`, `raccoon_s`,
`raccoon_l`입니다. 설치하면 `manifest.json`과
서명된 `MODEL.LICENSE`가 생성됩니다. `--accept-license`를 생략하면 대화형 터미널에서는
다운로드 전에 확인을 요청합니다. 비대화형 명령에는 이 플래그가 필수이며, 없으면 다운로드하지
않고 종료합니다. `models verify`는 패키지 식별 정보, 서명, 유효 기간과 현재 허가를 검증합니다.
실행 중 라이선스 표시의 대체 처리와 달리, 이 명시적 검증에는 서명된 라이선스 파일이 필요합니다.
공개 InsightFace 사전 학습 모델은 별도 상업용
라이선스가 없는 경우 비상업 연구용입니다.

`raccoon_s`와 `raccoon_l`을 지원합니다. Server는 각 패키지에서 검출과 인식 모델만 설치하며 Raccoon verifier는 로드하지 않습니다. 별도 버전 번호 없이 모델 이름으로 식별합니다. Web 라이브니스 작업은 기본 모델을 바꾸지 않습니다. 인식 모델이 달라지면 호환되는 Collection을 사용해야 하며 기존 embedding을 새 모델의 특징으로 취급하지 않습니다.

## 8. 시작 설정과 검색

```toml
[inference]
addons = []
liveness_mode = "normal"
liveness_threshold = 0.8
liveness_compare_scope = "both"
liveness_on_registration = false

[addons]
auto_download = []
```

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

완전한 로컬 소스 디렉터리에서 직접 빌드할 수 있습니다. 커밋하지 않은 변경 사항이
있거나 `.git` 디렉터리가 없어도 됩니다. Git 커밋이나 푸시는 빌드의 전제 조건이
아닙니다.

```bash
make -C server build-cpu
make -C server build-cuda12
```

테스트를 통과한 뒤에는 테스트한 것과 동일한 이미지를 배포하세요. 이후 같은 소스를
커밋하거나 커밋만 정리하는 경우에는 다시 빌드할 필요가 없습니다. 코드, 프런트엔드
리소스, 내장 사용자 도움말 등 이미지에 포함되는 파일을 변경하면 다시 빌드하고
검증해야 합니다.

로컬 이미지를 쓸 때 Compose에 `--pull never`를 추가합니다. 고정 Tag는
`0.3.1-cpu`, `0.3.1-cuda12`이고 이동 Tag `cpu`, `cuda12`는 최신 안정 버전을
가리키며 `latest`는 없습니다. 업그레이드 전 쓰기를 중지하고 `/data`와 crop을
SQLite-safe 방식으로 백업하세요. `docker compose down -v`는 데이터 Volume을
삭제하므로 사용하지 마세요.

### 0.3.1으로 업그레이드

0.3.1은 Docker 배포를 간소화합니다. 두 서비스가 root와 Docker 기본 capabilities로
실행되며 하나의 쓰기 가능한 모델 마운트를 공유합니다. 모델 루트가 없으면 Compose가
생성하고 명시적인 addon 다운로드 시 `addons/`를 만듭니다. 호스트 UID/GID나 공유 그룹을
준비할 필요가 없습니다.

0.3.0부터 `raccoon_s`, `raccoon_l` 및 모델 설명 파일, 선택적 라이브니스, Web addon
설치와 BMP 입력을 지원합니다. Server는 Raccoon의 검출·인식 모델만 사용하고 verifier는
로드하지 않습니다. 0.3.1에서는 이 기능과 API 응답 계약을 변경하지 않습니다.

**1.** `server/config/server.toml` 설정과 배포별 재정의 설정을 유지하면서 Server 소스와
Compose 파일을 0.3.1으로 업데이트합니다. 기존 모델 경로, `/data` 볼륨 이름, 얼굴 이미지
저장소, 포트와 API 키 설정을 유지하세요. 사용자 정의 Compose 파일은 `server`와 `models`
두 서비스의 이미지를 환경에 맞게 `0.3.1-cpu` 또는 `0.3.1-cuda12`로 변경합니다.
아래 명령에도 평소 사용하는 Compose 파일, 재정의 설정과 프로젝트 이름을 적용하세요.

이미지 태그만 바꾸면 충분하지 않습니다. 사용자 지정 Compose와 재정의 파일도 갱신하여
두 서비스를 `user: "0:0"`으로 설정하고 `cap_drop: [ALL]`과 이전 UID/GID 및 공유 그룹
설정을 제거하세요. `/models`는 `create_host_path: true`인 하나의 쓰기 가능한 마운트로
통합하고 별도 `/models/addons` 마운트를 제거합니다. 루트 파일시스템 읽기 전용과
`no-new-privileges`는 유지합니다. 기존 설정 디렉터리와 파일도 보존하세요. 두 서비스 모두
원자적 설정 저장을 위해 전체 디렉터리에 쓸 수 있어야 합니다. 설치 도구의 기존 읽기 전용
단일 파일 마운트를 디렉터리 마운트로 교체하세요. 기존 모델과 addon 캐시는 그대로 유지되며 표준 배포에서는 재귀적 권한 변경이나
addon 디렉터리 사전 준비가 필요하지 않습니다.

**2.** 새 이미지를 내려받고 Server 컨테이너를 다시 생성합니다. 저장소 루트에서 기존 배포에
맞는 명령을 선택하세요.

CPU:

```bash
docker compose -f server/deploy/compose.cpu.yml pull server models
docker compose -f server/deploy/compose.cpu.yml up -d --no-build --force-recreate server
curl -fsS http://127.0.0.1:18097/v1/health
```

CUDA:

```bash
docker compose -f server/deploy/compose.cuda12.yml pull server models
docker compose -f server/deploy/compose.cuda12.yml up -d --no-build --force-recreate server
curl -fsS http://127.0.0.1:18098/v1/health
```

로컬에서 빌드한다면 먼저 0.3.1 이미지를 빌드하고, 이미지를 내려받는 대신
`up -d --no-build --pull never --force-recreate server`를 사용합니다.
`docker compose restart`만으로는 새 이미지로 바뀌거나 마운트 변경이 적용되지 않습니다.

**3.** 시작 시 데이터베이스 마이그레이션이 자동으로 적용됩니다. `/v1/health`에서 `ready`와
버전 `0.3.1`을 반환할 때까지 기다린 뒤 **시스템**에서 모델과 실행 공급자가 예상과
같은지 확인합니다. 기존 Collection과 사람이 남아 있는지 확인하고, 결과를 알고 있는
이미지로 검색하세요. 같은 모델과 특징 계약을 유지하면 샘플, 특징 벡터와 Collection
계약 ID가 보존되므로 다시 등록할 필요가 없습니다.

**업그레이드 후에도 라이브니스는 선택 사항입니다.** 제공 설정과 추가 모델 키가 없는 이전
설정은 모두 비활성 상태를 유지하므로 업그레이드만 할 때는 라이브니스 모델을 내려받을 필요가
없습니다. Server는 시작 시 모델을 다운로드하지 않습니다. 활성화하려면
[라이브니스 설정](#선택적-라이브니스-addon)에 따라
[Web 다운로드용 마운트와 권한](#web-다운로드를-위한-마운트와-권한)을 준비한 뒤
**시스템 → 라이브니스 검사 → 다운로드하고 다시 시작 후 활성화**를 선택합니다.
모델 설치와 설정 저장이 성공하면 Server를 수동으로 다시 시작하세요. 기본값은 `normal`,
임계값 `0.8`, `liveness_on_registration=false`입니다. 모델은
`<models_dir>/addons/liveness.onnx`에 저장됩니다.

**Raccoon 도입은 별도의 모델 변경입니다.** Server를 업그레이드해도 현재 모델 패키지는
유지됩니다. `raccoon_s` 또는 `raccoon_l`을 사용하려면
[모델 설치 안내](#7-모델과-라이선스)에 따라 별도의 모델 디렉터리에 설치하고, 해당 경로를
사용하도록 배포를 설정합니다. Collection은 새 모델의 특징 계약과 일치해야 합니다.
호환되는 Collection을 만들고 다시 등록하거나 별도의 데이터 마이그레이션을 진행하세요.
Web UI에서는 기본 모델 패키지를 전환할 수 없습니다.

**0.3.0 이후 API와 SDK 호환성:** 모델, Collection, FaceSample 결과에서 `model_version`이 제거됩니다.
모델은 `model_id`로 식별하며 Collection 호환성은 `embedding_contract_id`로 확인합니다.
제거된 필드를 필수로 사용하는 클라이언트를 수정하고, 제공되는 Python 클라이언트를 업데이트할
때는 SDK `0.3.1`을 사용하세요. 라이브니스를 평가한 경우 `liveness`에는 `status`, `is_live`,
`live_score` 세 핵심 필드가 포함되고 `input_rejected`에만 `reason`이 추가됩니다. 평가하지 않았다면 `liveness` 자체를 생략합니다.
인식 요청에 활성화하기 전에 [라이브니스 결과와 오류 처리](#라이브니스-결과)를 확인하세요.

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
