# InsightFace Server REST API 사용 가이드

**언어:** [English](api.md) · [中文](api.zh-CN.md) · [日本語](api.ja.md) · [Deutsch](api.de.md) · [Español](api.es.md) · [Français](api.fr.md) · [Русский](api.ru.md) · [Português](api.pt.md) · 한국어

이 문서는 모든 공개 API의 목적, 입력, 서버 처리, 성공 결과와 오류를 설명합니다. 설치와 첫 검색은 [사용자 가이드](user-guide.ko.md)를, 현재 실행 버전의 정확한 스키마는 `/docs`와 `/openapi.json`을 확인하세요.

## 공통 규칙

- 기본 경로 `/v1`, JSON은 `snake_case`, 이미지는 JPEG/PNG/WebP multipart입니다.
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

## 시스템

### `GET /v1/health`

**용도/입력:** 공개 readiness, 파라미터와 인증 없음. **결과:** 시작 상태와 SQLite quick_check를 확인하고 200 `status`, `auth_enabled`, `request_id`. **오류:** `503 not_ready`.

### `GET /v1/system`

**용도/입력:** 안전한 운영 진단, 파라미터 없음. **결과:** 200 OS/CPU/GPU, Driver, CUDA/cuDNN/ORT, Provider, 모델, DB, mount, 수량, 검색, 안전 설정, 추론 동시성. 비밀, 이미지, embedding은 제외됩니다. **오류:** 401, 503.

### `GET /v1/models`

**용도/입력:** 검증된 detector/recognizer, Provider, License 확인. 파라미터 없음. **결과:** 200 `models`, `execution_provider`, `license`. **오류:** 401.

## 무상태 얼굴 처리

### `POST /v1/detect`

**입력:** multipart `image` 필수, `max_faces` 1–100, 선택 `collection_id`. **처리/결과:** 여러 해상도를 합쳐 전역 NMS, 면적순 정렬; 200 `faces`의 box/5점/score/quality와 `processing_ms`. 얼굴 없음은 정상 빈 목록입니다. **오류:** 400 구 min_score, 404 Collection, 413, 422 invalid_image, 503.

```bash
curl -sS "${BASE_URL}/v1/detect" -H "${AUTH_HEADER}" -F 'image=@group.jpg' -F 'max_faces=10'
```

### `POST /v1/compare`

**입력:** multipart `source`, `target`, 선택 `threshold` 0–1과 `collection_id`. **결과:** 각 이미지에서 한 얼굴을 골라 200 `matched`, cosine `similarity`, 실제 threshold, 두 face, 처리 시간. **오류:** 404, 413, 422 invalid_image/face_not_found, 503.

### `POST /v1/embeddings`

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

**입력:** IDs; 반복 images와 Person 생성과 같은 review/embedding 필드. **결과:** 201 `faces`, `rejected_images`, 부분 성공. **오류:** 등록 오류와 404 Person.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces`

**입력:** IDs; query limit 1–100, cursor. **결과:** 200 `faces` metadata, `has_crop`, `next_cursor`, embedding/bytes 제외. **오류:** 400 cursor, 404.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}/image`

**입력:** 세 ID. **결과:** 저장된 경우 200 `image/jpeg`, 112×112 crop, `Cache-Control:no-store`; request ID는 헤더에만 있습니다. **오류:** 401, 404 face/face_image_not_found.

### `DELETE /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}`

**입력:** 세 ID. **결과:** embedding/crop/index row 삭제, 204. **오류:** 404, 503.

## 검색

### `POST /v1/collections/{collection_id}/search`

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

**용도:** 화면 없는 클라이언트에서 현재 상태를 폴링합니다. **입력:** Monitor ID입니다. **결과:** 200으로 연결, 실제 FPS, 지연, 건너뛴 프레임, 현재 인식/미등록 얼굴, preview, 재연결과 안전한 오류를 반환하며 embedding은 제외합니다. **오류:** 401, 404.

### `GET /v1/monitors/{monitor_id}/events`

**용도:** 휘발성 입장/퇴장/오류/복구 이벤트를 조회합니다. **입력:** `limit` 1~1000과 마지막 불투명 `cursor`입니다. **결과:** 200의 `events`, `next_cursor`, `truncated`, `stream_reset`; 재시작 시 이벤트가 사라집니다. **오류:** 400 `invalid_cursor`, 401, 404.

### `GET /v1/monitors/{monitor_id}/preview.mjpeg`

**용도:** 기본 비활성인 원본 MJPEG preview를 엽니다. **입력:** ID와 일반 Bearer 헤더이며 API 키를 URL에 넣지 않습니다. **결과:** 시청자가 있을 때만 인코딩하는 장기 `multipart/x-mixed-replace`; 상자는 클라이언트가 `/state`로 그립니다. **오류:** 401, 404, 409 `preview_disabled`, 503.

## 재시도

GET은 재시도할 수 있습니다. DELETE 재시도 전 상태를 확인하세요. Person/Face 생성의 네트워크 결과가 불확실하면 POST 전 ID를 조회합니다. 429와 일시적 503만 제한된 exponential backoff와 jitter로 재시도하고 4xx는 요청을 수정하세요.
