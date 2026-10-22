# InsightFace Server REST API 사용 가이드

**언어:** [English](api.md) · [中文](api.zh-CN.md) · [日本語](api.ja.md) · [Deutsch](api.de.md) · [Español](api.es.md) · [Français](api.fr.md) · [Русский](api.ru.md) · [Português](api.pt.md) · 한국어

이 문서는 모든 공개 API의 목적, 입력, 서버 처리, 성공 결과와 오류를 설명합니다. 설치와 첫 검색은 [사용자 가이드](user-guide.ko.md)를, 현재 실행 버전의 정확한 스키마는 `/docs`와 `/openapi.json`을 확인하세요.


모델은 `model_id`로 식별하며 응답에 별도의 `model_version`을 포함하지 않습니다. 기존 Collection의 `embedding_contract_id`는 유지됩니다.

라이브니스를 사용하려면 [설정, 모델 설치와 결과 설명](#선택적-라이브니스-addon)을 확인하세요. 각 작업 절에서도 해당 동작에 미치는 영향을 설명합니다.

## 공통 규칙

- 기본 경로 `/v1`, JSON은 `snake_case`, 이미지는 JPEG/PNG/WebP/BMP multipart입니다.
- 제공되는 Compose는 격리 평가용으로 인증이 기본 비활성화됩니다. 활성화 시 health 외에는 `Authorization: Bearer <api_key>`가 필요하고, 비활성화 시 빈 헤더를 보내지 말고 완전히 생략합니다.
- 모든 응답에 `x-request-id`, JSON에는 같은 `request_id`가 있습니다.
- confidence/quality/threshold는 `0..1`입니다. Similarity는 확률이 아닌 원본 cosine `[-1,1]`이며 기본 threshold는 `0.4`, `similarity >= threshold`일 때 일치합니다.
- cursor는 불투명하며 같은 경로, Collection, Person, filter에 변경 없이 다시 보냅니다.
- 일반 상태: 400 입력, 401 인증, 404 없음, 409 충돌, 413 크기, 422 이미지/얼굴, 429 제한, 503 timeout/model/index.

```bash
BASE_URL=http://127.0.0.1:18097
AUTH_HEADER="Authorization: Bearer ${INSIGHTFACE_API_KEY}"
curl -fsS "${BASE_URL}/v1/health"
```

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

**시스템 → 라이브니스 검사**에서 모델을 다운로드하고 다음 시작 시 활성화합니다. SHA-256 검증 후 두 목록에 `liveness`를 추가하고 다른 항목, 주석과 설정은 보존합니다. 검증된 파일은 재사용합니다. 현재 실행 상태는 그대로이며 **Server를 수동으로 다시 시작**해야 적용됩니다. 오류 시 재시도할 수 있고 다운로드 실패로 활성화되지 않습니다.

[Web 다운로드를 위한 마운트와 권한](user-guide.ko.md#web-다운로드를-위한-마운트와-권한).

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

[사용자 가이드의 초기 설정](user-guide.ko.md)에 있는 현재 Compose와 기존 설정을 사용하세요. Server와 설치 도구는 root로 하나의 쓰기 가능한 모델 마운트를 공유합니다. Compose가 모델 루트를 생성하고 명시적인 addon 다운로드 시 `addons/`를 만듭니다. 호스트 UID/GID나 수동 권한 설정은 필요하지 않습니다.

```bash
docker compose -f server/deploy/compose.cpu.yml run --rm models addons install liveness
docker compose -f server/deploy/compose.cpu.yml run --rm models addons verify liveness
docker compose -f server/deploy/compose.cpu.yml up -d --force-recreate
```

활성 모델이 없으면 `addon_model_missing`, 잘못된 파일이면 `addon_model_invalid`로 시작을 중단합니다. 설정된 addon을 자동으로 끄지 않습니다.

### 라이브니스 결과

| 결과 | `status` | `is_live` | `live_score` |
| --- | --- | --- | --- |
| 실제 얼굴 | `ok` | `true` | `[0, 1]` |
| 위조 | `ok` | `false` | `[0, 1]` |
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

`/v1/detect`는 미통과 결과도 HTTP 200으로 반환합니다. `normal`에서 임베딩·비교·검색은 HTTP 422 `liveness_fake` 또는 `liveness_input_rejected`와 `error.details.liveness`를 반환하며 비교에는 `details.side`가 추가됩니다. 추론 장애는 HTTP 503 `liveness_unavailable`입니다. 실행 오류는 `normal`과 `observe` 모두에서 작업을 중단하며 `input_rejected`로 변환하지 않습니다.

새 Person 등록과 FaceSample 추가는 기본적으로 라이브니스를 건너뜁니다. `[inference].liveness_on_registration=false`이면 모델을 실행하지 않고 새 샘플에 `liveness`를 포함하지 않습니다. `true`이고 addon이 활성화되어 있으면 `normal`/`observe` 정책을 적용하며 거부 항목에는 `reason`과 `liveness`가 포함됩니다. `review_mode` 품질 심사와 외부 임베딩 검증은 계속 수행합니다. 등록 검사가 활성화된 경우 `review_mode=off`와 `external_trusted`도 우회할 수 없습니다. 요청에서 이 시작 설정을 덮어쓸 수 없습니다. 이전에 저장한 결과는 계속 조회할 수 있습니다.

RTSP는 `liveness_blocked`를 `unknown`과 구분하고 `liveness_blocked_faces`로 집계합니다. 차단된 얼굴은 사람/미확인 입장 이벤트를 생성하지 않고 확인 프레임 수를 초기화합니다. 추론 장애 시 이전에 표시된 신원을 지웁니다.

`liveness_compare_scope`는 `/v1/compare`의 검사 대상을 정합니다. `both`(기본값)는 양쪽, `source`는 원본, `target`은 대상 이미지입니다. `live_score >= liveness_threshold`이면 실제 얼굴로 판정합니다.

`models addons install liveness`는 공개 모델을 `/models/addons/liveness.onnx`에 저장하며, Compose 호스트에서는 `server/.models/addons/liveness.onnx`입니다. 시작 오류는 `addon_model_missing`과 `addon_model_invalid`입니다. `/v1/models`와 `/v1/system`은 활성 추가 모델을 `addons`로 반환합니다.

[설정과 작업 안내](user-guide.ko.md#선택적-라이브니스-addon).

## 시스템

### `GET /v1/health`

**용도/입력:** 공개 readiness, 파라미터와 인증 없음. **결과:** 시작 상태와 SQLite quick_check를 확인하고 200 `status`, `auth_enabled`, `request_id`. **오류:** `503 not_ready`.

### `GET /v1/system`

**용도/입력:** 안전한 운영 진단, 파라미터 없음. **결과:** 200 OS/CPU/GPU, Driver, CUDA/cuDNN/ORT, Provider, 모델, DB, mount, 수량, 검색, 안전 설정, 추론 동시성. 비밀, 이미지, embedding은 제외됩니다. **오류:** 401, 503.

### `GET /v1/models`

`addons`는 기본 모델과 별개로 활성 addon을 반환합니다. `liveness` 항목과 시스템 응답의 `safe_config`에서 적용된 설정을 확인하세요. 이 읽기 전용 API는 모델을 설치하지 않습니다.

**용도/입력:** 검증된 detector/recognizer, Provider, License 확인. 파라미터 없음. **결과:** 200 `models`, `execution_provider`, `license`. **오류:** 401.

기본 패키지 `raccoon_s`와 `raccoon_l`은 CPU와 CUDA를 지원하며 시작 전에 모델 도구로 설치합니다. 이 API는 실행 중인 구성 요소를 나열하며 다운로드 목록이 아닙니다. 아래 Web 작업은 라이브니스만 관리합니다. Collection은 인식 모델 및 전처리 계약과 연결됩니다. 기본 패키지를 바꿔도 기존 특징 벡터가 변환되지 않으며 `409 collection_model_mismatch`가 발생할 수 있습니다. 라이브니스만 활성화하면 이 계약은 바뀌지 않습니다.

### `GET /v1/addons/liveness`

**용도:** 다운로드나 설정 변경 없이 설치 상태와 다음 시작 설정을 확인합니다. 관리용 API이며 별도의 라이브니스 추론 API가 아닙니다.

**결과:** HTTP 200. `enabled`는 현재 프로세스의 활성 상태입니다. `installed`는 파일이 공개된 SHA-256 검증을 통과했다는 뜻이며 활성화와는 다릅니다. `configured_enabled`는 현재 설정 파일에서 읽은 다음 시작의 선택이고, `restart_required`는 이 값이 `enabled`와 다름을 나타냅니다. 재시작 전까지 `/v1/system`의 `safe_config`는 현재 프로세스의 설정을 나타냅니다.

`state`는 `idle`(검증된 모델 없음), `downloading`(준비 중), `ready`(검증된 모델 있음), `error`(준비·파일·설정 오류) 중 하나입니다. `ready`만으로 활성화 설정 저장이나 재시작 완료를 판단하면 안 됩니다.

`can_enable`은 Web 준비 작업의 가능 여부입니다. 불가능할 때 `unavailable_code`는 안정적인 이유 코드, `unavailable_reason`은 설명을 제공하며, 가능하면 둘 다 `null`입니다. `error`는 `null` 또는 `code`와 `message`를 가진 객체입니다. `model_path`는 로컬 모델 경로이고, `config_file`은 선택한 TOML 경로 또는 `null`입니다. 응답에는 `request_id`도 포함됩니다.

`unavailable_code`의 값은 `config_file_missing`(설정 파일 미지정), `config_file_not_regular`(일반 파일 아님), `config_file_mount`(개별 파일 마운트), `config_not_writable`(설정 쓰기 불가), `addon_directory_not_writable`(추가 모델 디렉터리 쓰기 불가), `addon_config_invalid`(잘못된 설정), `addon_model_invalid`(잘못된 모델), `server_stopping`(종료 중)입니다.

```json
{
  "enabled": false,
  "installed": true,
  "configured_enabled": true,
  "restart_required": true,
  "can_enable": true,
  "unavailable_code": null,
  "unavailable_reason": null,
  "state": "ready",
  "error": null,
  "model_path": "/models/addons/liveness.onnx",
  "config_file": "/etc/insightface/server.toml",
  "request_id": "3ed21e89-4595-4eed-a699-1df42ca62032"
}
```

### `POST /v1/addons/liveness/enable`

**용도:** 모델을 다운로드하고 다음 시작에 활성화하도록 설정합니다. `Content-Type: application/json`과 빈 객체 `{}`를 보냅니다. 모델 URL이나 다른 매개변수는 받지 않습니다.

```bash
curl -sS "${BASE_URL}/v1/addons/liveness" -H "${AUTH_HEADER}"
curl -sS "${BASE_URL}/v1/addons/liveness/enable" -H "${AUTH_HEADER}" \
  -H 'Content-Type: application/json' -d '{}'
```

**결과:** HTTP 202는 GET과 같은 상태 필드를 반환하며 작업 접수를 의미합니다. 활성화 완료를 뜻하지 않습니다. 준비가 끝날 때까지 `GET /v1/addons/liveness`를 조회합니다. 중복 요청은 실행 중인 작업을 공유하며 브라우저를 닫아도 취소되지 않습니다.

다운로드와 SHA-256 검증을 마친 후에만 `config_file`의 `[inference].addons`와 `[addons].auto_download`에 `liveness`를 추가합니다. 다른 값과 주석은 보존하고 검증된 파일은 재사용합니다. `installed=true`, `configured_enabled=true`, `restart_required=true`이면 서버를 수동으로 재시작합니다. 새 프로세스에서는 `enabled=true`, `restart_required=false`가 됩니다. 실행 중 다시 로드하거나 기본 모델 패키지를 전환하는 API는 없습니다.

**오류:** 요청 오류는 일반 오류 형식을 사용합니다. 본문이 `{}`가 아니면 `400 invalid_addon_request`, 인증 실패는 `401 unauthorized`, 허용되지 않은 브라우저 출처는 `403 origin_not_allowed`, 경로·권한·설정 문제는 `409 addon_management_unavailable`, JSON이 아닌 콘텐츠 유형은 `415 json_required`입니다. 브라우저는 서버와 같은 출처 또는 명시적으로 허용된 CORS 출처를 사용해야 합니다.

접수 후에도 실패할 수 있습니다. GET은 HTTP 200을 유지하며 `state=error`와 `error.code`로 실패를 알립니다. `addon_download_failed`는 설정을 바꾸지 않으므로 서버의 네트워크나 프록시를 확인합니다. `addon_config_save_failed`는 설정이나 디렉터리 권한을 수정해야 하며 검증된 모델을 재사용할 수 있습니다. `addon_config_invalid`는 디스크의 TOML이 유효하지 않다는 뜻입니다. `addon_model_invalid`는 잘못된 파일을 교체하거나 삭제해야 하며 자동으로 덮어쓰지 않습니다. `addon_job_in_progress`는 다른 프로세스가 준비 중이므로 기다린 후 새로 고칩니다. 원인을 해결한 뒤 POST를 다시 보냅니다.

## 무상태 얼굴 처리

### `POST /v1/detect`

평가한 얼굴에는 `liveness.status`, `liveness.is_live`, `liveness.live_score`가 포함됩니다. Fake와 `input_rejected`도 HTTP 200이며 인식 특징은 추출하지 않습니다. `input_rejected`는 얼굴 주변의 이미지 영역이 부족함을 뜻하며 `liveness.reason`에 이미지 조정 방법이 제공됩니다. `liveness`가 없으면 평가하지 않은 것입니다.

**입력:** multipart `image` 필수, `max_faces` 1–100, 선택 `collection_id`. **처리/결과:** 여러 해상도를 합쳐 전역 NMS, 면적순 정렬; 200 `faces`의 box/5점/score/quality와 `processing_ms`. 얼굴 없음은 정상 빈 목록입니다. **오류:** 400 구 min_score, 404 Collection, 413, 422 invalid_image, 503.

```bash
curl -sS "${BASE_URL}/v1/detect" -H "${AUTH_HEADER}" -F 'image=@group.jpg' -F 'max_faces=10'
```

### `POST /v1/compare`

`liveness_compare_scope`(`both`, `source`, `target`)가 인식 전에 검사할 쪽을 정합니다. `normal`에서 거부되면 HTTP 422 `liveness_fake` / `liveness_input_rejected`, `error.details.liveness`, `error.details.side`를 반환하고 유사도는 반환하지 않습니다. `observe`는 비교를 계속하고 검사한 얼굴에 결과를 포함합니다.

**입력:** multipart `source`, `target`, 선택 `threshold` 0–1과 `collection_id`. **결과:** 각 이미지에서 한 얼굴을 골라 200 `matched`, cosine `similarity`, 실제 threshold, 두 face, 처리 시간. **오류:** 404, 413, 422 invalid_image/face_not_found, 503.

### `POST /v1/embeddings`

라이브니스가 켜진 `normal`에서는 위조/부적합 입력에 HTTP 422 `liveness_fake` / `liveness_input_rejected`와 `error.details.liveness`를 반환하며 embedding을 추출하지 않습니다. `observe`는 embedding과 얼굴의 라이브니스 결과를 반환합니다.

**입력:** multipart `image`, 선택 `collection_id`. **결과:** 200 선택 face, L2 embedding, 모델, 시간. 일반 등록에는 필요 없고 vector는 로그에 남지 않습니다. **오류:** 400 구 face_selection, 404, 413, 422, 503.

## Collections

### `POST /v1/collections`

**입력:** JSON `id`, `name`; 선택 description, threshold(0.4), metadata, save_face_crops, `detection`, `search`의 profile/capacity/max_faces_per_person/load_policy. **처리/결과:** 모델, 전처리, 검색 계약을 고정하고 201 완전한 `collection`. **오류:** 400 설정, 409 exists, 503 index.

```bash
curl -sS "${BASE_URL}/v1/collections" -H "${AUTH_HEADER}" -H 'Content-Type: application/json' -d '{"id":"employees","name":"Employees","threshold":0.4}'
```

### `GET /v1/collections`

**입력:** query `limit` 1–100(50), 선택 cursor. **결과:** 200 `collections`, nullable `next_cursor`. **오류:** 400 invalid_cursor, 401.

### `GET /v1/collections/{collection_id}`

**입력:** 경로 Collection ID. **결과:** 200 `collection`, Person/Face 수, `embedding_contract_id`. **오류:** 404.

### `PATCH /v1/collections/{collection_id}`

**입력:** ID; JSON name/description/threshold/metadata/save_face_crops, 검색 capacity/max/load와 detection. null, 알 수 없는 필드, 모델과 search profile 변경은 불가합니다. **결과:** 200 전체 Collection; detection은 다음 요청부터 적용됩니다. **오류:** 400, 404, 409, 503.

### `DELETE /v1/collections/{collection_id}`

**입력:** ID; query `force=false`, 비어 있지 않으면 true. **결과:** 204 본문 없음. **오류:** 404, 409 collection_not_empty, 503.

## Person과 FaceSample

### `POST /v1/collections/{collection_id}/persons`

Person 생성과 FaceSample 추가는 기본적으로 라이브니스를 건너뜁니다(`liveness_on_registration=false`). 활성화하면 `normal`은 위조/부적합 입력을 거부하고, `observe`는 결과를 저장하며 계속합니다. 품질 검토는 선택한 `review_mode`를 따릅니다. 거부 목록은 실제 `reason`과 라이브니스 결과를 따로 표시합니다.

**입력:** Collection; multipart 반복 `images`, 선택 id/name/external_id, JSON 문자열 metadata, `review_mode=off|standard|strict`, `embedding_mode=server|external_trusted`; 외부 모드는 벡터와 contract ID도 필요합니다. **처리/결과:** 이미지별 심사; 201 `person`, 승인 `faces`, `rejected_images`, 부분 성공 가능. 전부 거절되면 Person 없이 422. **오류:** 400, 404, 409 ID/계약/용량, 413, 422, 503.

```bash
curl -sS "${BASE_URL}/v1/collections/employees/persons" -H "${AUTH_HEADER}" -F 'id=alice' -F 'review_mode=off' -F 'images=@alice.jpg'
```

### `GET /v1/collections/{collection_id}/persons`

**입력:** Collection; query limit/cursor/`search`로 ID, 이름, external ID 검색. **결과:** 200 `persons`, `next_cursor`. **오류:** 400 cursor, 404.

### `GET /v1/collections/{collection_id}/persons/{person_id}`

**입력:** Collection과 Person ID. **결과:** 200 `person`, face_count. **오류:** 404.

### `PATCH /v1/collections/{collection_id}/persons/{person_id}`

**입력:** IDs; JSON name/external_id/object metadata. **결과:** 200 수정된 Person. **오류:** 400, 404, 409 external_id_exists.

### `DELETE /v1/collections/{collection_id}/persons/{person_id}`

**입력:** IDs. **결과:** Person, FaceSamples, embeddings, crops를 삭제하고 인덱스 동기화, 204. **오류:** 404, 503.

### `POST /v1/collections/{collection_id}/persons/{person_id}/faces`

Person 생성과 FaceSample 추가는 기본적으로 라이브니스를 건너뜁니다(`liveness_on_registration=false`). 활성화하면 `normal`은 위조/부적합 입력을 거부하고, `observe`는 결과를 저장하며 계속합니다. 품질 검토는 선택한 `review_mode`를 따릅니다. 거부 목록은 실제 `reason`과 라이브니스 결과를 따로 표시합니다.

**입력:** IDs; 반복 images와 Person 생성과 같은 review/embedding 필드. **결과:** 201 `faces`, `rejected_images`, 부분 성공. **오류:** 등록 오류와 404 Person.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces`

**입력:** IDs; query limit 1–100, cursor. **결과:** 200 `faces` metadata, `has_crop`, `next_cursor`, embedding/bytes 제외. **오류:** 400 cursor, 404.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}/image`

**입력:** 세 ID. **결과:** 저장된 경우 200 `image/jpeg`, 112×112 crop, `Cache-Control:no-store`; request ID는 헤더에만 있습니다. **오류:** 401, 404 face/face_image_not_found.

### `DELETE /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}`

**입력:** 세 ID. **결과:** embedding/crop/index row 삭제, 204. **오류:** 404, 503.

## 검색

### `POST /v1/collections/{collection_id}/search`

라이브니스가 켜진 `normal`에서는 위조/부적합 쿼리에 HTTP 422 `liveness_fake` / `liveness_input_rejected`와 `error.details.liveness`를 반환하며 검색하지 않습니다. 이는 검색 성공 후 빈 일치 목록과 다릅니다. `observe`는 검색을 계속하고 쿼리 얼굴에 결과를 반환합니다.

**입력:** Collection; multipart `image`, `limit` 1–100(5), 선택 threshold 또는 Collection 값. **처리/결과:** 선택 얼굴을 모든 sample과 비교하고 Person별 최고값 사용; 200 `searched_face`, 정렬 `matches`, threshold, 시간. 일치 없음은 빈 목록. **오류:** 404, 409 모델, 413, 422 이미지/얼굴, 503 index/timeout.

```bash
curl -sS "${BASE_URL}/v1/collections/employees/search" -H "${AUTH_HEADER}" -F 'image=@query.jpg' -F 'limit=5'
```

## RTSP Monitor

Monitor 설정은 SQLite에 영구 저장되고 활성화된 작업은 서버 재시작 후 복원됩니다. 영상 프레임은 저장하지 않으며 이벤트는 용량이 제한된 메모리 링 버퍼에만 유지됩니다.

### `POST /v1/monitors`

**용도:** 영구 RTSP 인식 Monitor를 만듭니다. **입력:** ID, 이름, `source`, Collection, `inference_fps`(2), 선택 임계값, 이벤트 버퍼/정책, `preview_enabled`(false)를 담은 JSON입니다. **결과:** 자격증명을 제거한 `monitor`와 201; 자격증명은 암호화 저장됩니다. **오류:** 400, 404, 409, 429.

### `GET /v1/monitors`

**용도:** Monitor 목록을 페이지 단위로 조회합니다. **입력:** `limit` 1~100(50)과 이전 응답의 불투명한 `cursor`를 변경 없이 사용합니다. **결과:** 200의 `monitors`, `next_cursor`이며 자격증명은 반환하지 않습니다. **오류:** 400 `invalid_cursor`, 401.

### `GET /v1/monitors/{monitor_id}`

**용도:** 한 Monitor의 설정과 실행 요약을 읽습니다. **입력:** 경로의 `monitor_id`입니다. **결과:** 200으로 이벤트 정책, 마스킹된 source, preview 설정과 상태를 반환합니다. **오류:** 401, 404 `monitor_not_found`.

### `PATCH /v1/monitors/{monitor_id}`

**용도:** ID 외 필드를 일부 수정하고 `enabled`로 시작/중지합니다. **입력:** 부분 JSON이며 `event_policy`도 일부만 보낼 수 있고 null 임계값은 Collection 값을 상속합니다. **결과:** 200의 전체 Monitor; source/Collection/속도/정책 변경은 작업을 재시작합니다. **오류:** 400, 404, 429.

### `DELETE /v1/monitors/{monitor_id}`

**용도:** Monitor를 영구 삭제합니다. **입력:** 경로의 `monitor_id`입니다. **결과:** 디코더, 추론, RTSP 연결을 중지하고 메모리 이벤트를 버린 뒤 204를 반환하며 Collection은 유지합니다. **오류:** 401, 404.

### `GET /v1/monitors/{monitor_id}/state`

라이브니스가 켜진 `normal`에서 차단된 얼굴은 `status: liveness_blocked`와 별도의 결과를 가집니다. `unknown_faces` 대신 `liveness_blocked_faces`로 집계되며 입장 이벤트를 생성하지 않습니다. `observe`는 인식을 계속합니다. 입력 거부와 fake는 구분해서 표시합니다.

**용도:** 화면 없는 클라이언트에서 현재 상태를 폴링합니다. **입력:** Monitor ID입니다. **결과:** 200으로 연결, 실제 FPS, 지연, 건너뛴 프레임, 현재 인식/미등록 얼굴, preview, 재연결과 안전한 오류를 반환하며 embedding은 제외합니다. **오류:** 401, 404.

### `GET /v1/monitors/{monitor_id}/events`

**용도:** 휘발성 입장/퇴장/오류/복구 이벤트를 조회합니다. **입력:** `limit` 1~1000과 마지막 불투명 `cursor`입니다. **결과:** 200의 `events`, `next_cursor`, `truncated`, `stream_reset`; 재시작 시 이벤트가 사라집니다. **오류:** 400 `invalid_cursor`, 401, 404.

### `GET /v1/monitors/{monitor_id}/preview.mjpeg`

**용도:** 기본 비활성인 원본 MJPEG preview를 엽니다. **입력:** ID와 일반 Bearer 헤더이며 API 키를 URL에 넣지 않습니다. **결과:** 시청자가 있을 때만 인코딩하는 장기 `multipart/x-mixed-replace`; 상자는 클라이언트가 `/state`로 그립니다. **오류:** 401, 404, 409 `preview_disabled`, 503.

## 재시도

GET은 재시도할 수 있습니다. DELETE 재시도 전 상태를 확인하세요. Person/Face 생성의 네트워크 결과가 불확실하면 POST 전 ID를 조회합니다. 429와 일시적 503만 제한된 exponential backoff와 jitter로 재시도하고 4xx는 요청을 수정하세요.
