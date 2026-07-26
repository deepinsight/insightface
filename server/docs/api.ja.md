# InsightFace Server REST API 利用ガイド

**言語:** [English](api.md) · [中文](api.zh-CN.md) · 日本語 · [Deutsch](api.de.md) · [Español](api.es.md) · [Français](api.fr.md) · [Русский](api.ru.md) · [Português](api.pt.md) · [한국어](api.ko.md)

この文書は全公開 API の用途、入力、サーバー処理、正常結果、主なエラーを説明します。コンテナとモデルの準備は [ユーザーガイド](user-guide.ja.md)、実行中の厳密な Schema は `/docs` または `/openapi.json` を参照してください。

## 共通規則と最初の呼び出し

- ベースパスは `/v1`、JSON は `snake_case`、画像は JPEG/PNG/WebP の multipart です。
- 同梱 Compose は隔離評価向けに認証を既定で無効にします。有効時は health 以外へ `Authorization: Bearer <api_key>` を送ります。無効時は空の Authorization を送らず省略します。
- 全レスポンスに `x-request-id`、JSON には同じ `request_id` があります。
- confidence/quality/threshold は `0..1`。similarity は確率ではなく `[-1,1]` の生 cosine で、既定しきい値は `0.4`、一致条件は `similarity >= threshold` です。
- `cursor` は不透明です。同じパス、Collection、Person、filter にそのまま返してください。
- 主な状態は 400 入力、401 認証、404 不在、409 競合、413 サイズ、422 画像/顔、429 上限、503 timeout/model/index です。

```bash
BASE_URL=http://127.0.0.1:18097
AUTH_HEADER="Authorization: Bearer ${INSIGHTFACE_API_KEY}"
curl -fsS "${BASE_URL}/v1/health"
```

## システム

### `GET /v1/health`

**用途/入力:** 公開 readiness。パラメーターなし、認証不要。**処理/結果:** DB quick check と起動状態を確認し、200 で `status`、`auth_enabled`、`request_id`。**エラー:** 未準備は `503 not_ready`。

### `GET /v1/system`

**用途/入力:** 安全な運用診断。パラメーターなし。**結果:** 200 で OS/CPU/GPU、Driver、CUDA/cuDNN/ORT、Provider、モデル、DB、mount、件数、検索 backend、安全設定、推論並行数。秘密・画像・embedding は含みません。**エラー:** 401、503。

### `GET /v1/models`

**用途/入力:** 検証済み detector/recognizer、実 Provider、License を確認。パラメーターなし。**結果:** 200 の `models`、`execution_provider`、`license`。**エラー:** 401。

## ステートレス顔処理

### `POST /v1/detect`

**入力:** multipart `image` 必須、`max_faces` 1～100、任意 `collection_id`。**処理/結果:** 複数解像度候補を統合して一度 NMS、面積順の `faces`、box/5点/score/quality、`processing_ms`。顔なしは 200 の空配列。**エラー:** 400 旧 min_score、404 Collection、413、422 invalid_image、503。

```bash
curl -sS "${BASE_URL}/v1/detect" -H "${AUTH_HEADER}" -F 'image=@group.jpg' -F 'max_faces=10'
```

### `POST /v1/compare`

**入力:** multipart `source`、`target` 必須、`threshold` 0～1、任意 `collection_id`。**処理/結果:** 各画像から設定戦略で1顔を選び、200 で `matched`、cosine `similarity`、実 threshold、両 face、処理時間。**エラー:** 404、413、422 invalid_image/face_not_found、503。

### `POST /v1/embeddings`

**入力:** multipart `image` 必須、任意 `collection_id`。**結果:** 200 で選択 face、L2 正規化 embedding、model、処理時間。通常登録には不要で、値はログされません。**エラー:** 400 旧 face_selection、404、413、422、503。

## Collection

### `POST /v1/collections`

**入力:** JSON `id`、`name` 必須。任意 `description`、`threshold`(既定0.4)、`metadata`、`save_face_crops`、`detection`、`search`。search は profile/capacity_rows/max_faces_per_person/load_policy。**処理/結果:** モデル・前処理・検索契約を固定し、201 で解決済み `collection`。**エラー:** 400 profile/detection/capacity、409 exists、503 index。

```bash
curl -sS "${BASE_URL}/v1/collections" -H "${AUTH_HEADER}" -H 'Content-Type: application/json' -d '{"id":"employees","name":"Employees","threshold":0.4}'
```

### `GET /v1/collections`

**入力:** query `limit` 1～100(既定50)、任意不透明 `cursor`。**結果:** 200 の `collections` と nullable `next_cursor`。**エラー:** 400 invalid_cursor、401。

### `GET /v1/collections/{collection_id}`

**入力:** path `collection_id`。**結果:** 200 の `collection`、Person/Face件数、`embedding_contract_id`。**エラー:** 404。

### `PATCH /v1/collections/{collection_id}`

**入力:** path ID。JSON で name/description/threshold/metadata/save_face_crops、search の capacity/max/load、detection を変更。null、未知フィールド、モデルと search profile の変更は不可。**結果:** 200 の完全な更新 Collection、次リクエストから反映。**エラー:** 400、404、409、503。

### `DELETE /v1/collections/{collection_id}`

**入力:** path ID、query `force=false`。非空を消す時のみ true。**結果:** 204 本文なし。**エラー:** 404、409 collection_not_empty、503。

## Person と FaceSample

### `POST /v1/collections/{collection_id}/persons`

**入力:** path Collection。multipart の repeatable `images` 必須、任意 id/name/external_id、JSON文字列 metadata、`review_mode=off|standard|strict`、`embedding_mode=server|external_trusted`。外部特徴では vector 配列と contract ID も必須。**処理/結果:** 画像ごとに審査し、201 で `person`、受理 `faces`、`rejected_images`。部分成功可。全失敗は 422 で Person を作りません。**エラー:** 400、404、409 ID/contract/capacity、413、422、503。

```bash
curl -sS "${BASE_URL}/v1/collections/employees/persons" -H "${AUTH_HEADER}" -F 'id=alice' -F 'review_mode=off' -F 'images=@alice.jpg'
```

### `GET /v1/collections/{collection_id}/persons`

**入力:** path Collection、query limit/cursor/`search`(ID・名前・外部ID)。**結果:** 200 の `persons` と `next_cursor`。**エラー:** 400 cursor、404。

### `GET /v1/collections/{collection_id}/persons/{person_id}`

**入力:** Collection ID と Person ID。**結果:** 200 の `person` と face_count。**エラー:** 404。

### `PATCH /v1/collections/{collection_id}/persons/{person_id}`

**入力:** path IDs、JSON name/external_id/object metadata。**結果:** 200 の更新 `person`。**エラー:** 400、404、409 external_id_exists。

### `DELETE /v1/collections/{collection_id}/persons/{person_id}`

**入力:** path IDs。**処理/結果:** Person、全 FaceSample、embedding、任意 crop を消し索引へ同期、204。**エラー:** 404、503。

### `POST /v1/collections/{collection_id}/persons/{person_id}/faces`

**入力:** path IDs、repeatable images と review/embedding fields（Person作成と同じ）。**結果:** 201 の `faces` と `rejected_images`、部分成功可。**エラー:** 登録エラーと 404 Person。

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces`

**入力:** path IDs、query limit 1～100 と cursor。**結果:** 200 の metadata `faces`、`has_crop`、`next_cursor`。embedding/画像byteは返しません。**エラー:** 400 cursor、404。

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}/image`

**入力:** 3 path IDs。**結果:** 保存済みの場合 200 `image/jpeg` 112×112 crop、`Cache-Control:no-store`。request ID はheaderのみ。**エラー:** 401、404 face/face_image_not_found。

### `DELETE /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}`

**入力:** 3 path IDs。**結果:** embedding/crop/索引行を削除し204。**エラー:** 404、503。

## 検索

### `POST /v1/collections/{collection_id}/search`

**入力:** path Collection、multipart `image`、`limit` 1～100(既定5)、任意 threshold(省略時Collection値)。**処理/結果:** 選択顔を全 FaceSample と比較し、Personごとの最高値を降順で返します。200 の `searched_face`、`matches`、threshold、処理時間。該当なしは空配列。**エラー:** 404、409 model、413、422 image/face、503 index/timeout。

```bash
curl -sS "${BASE_URL}/v1/collections/employees/search" -H "${AUTH_HEADER}" -F 'image=@query.jpg' -F 'limit=5'
```

## RTSP Monitor

Monitor設定はSQLiteに永続化され、有効なタスクはServer再起動後に復元されます。動画frameは保存せず、eventは上限付きmemory bufferだけに保持します。

### `POST /v1/monitors`

**用途:** 永続的なRTSP認識Monitorを作成します。**入力:** JSONのID、名前、`source`、Collection、`inference_fps`（既定2）、任意threshold、buffer/event policy、`preview_enabled`（既定false）。**結果:** 201で認証情報を除いた`monitor`。URL資格情報は暗号化保存されます。**エラー:** 400、404、409、429。

### `GET /v1/monitors`

**用途:** Monitor一覧をページングします。**入力:** `limit` 1～100（既定50）と、前回の不透明な`cursor`。**結果:** 200の`monitors`と`next_cursor`；認証情報は返しません。**エラー:** 400 `invalid_cursor`、401。

### `GET /v1/monitors/{monitor_id}`

**用途:** 1件の設定とruntime要約を取得します。**入力:** pathの`monitor_id`。**結果:** 200の`monitor`にevent policy、脱敏source、preview設定、状態を含みます。**エラー:** 401、404 `monitor_not_found`。

### `PATCH /v1/monitors/{monitor_id}`

**用途:** ID以外を部分更新し、`enabled`で開始/停止します。**入力:** JSONの変更フィールド；`event_policy`も部分更新でき、thresholdのnullはCollection値を継承します。**結果:** 200の完全な`monitor`。source/Collection/rate/policy変更時はtaskを再起動します。**エラー:** 400、404、429。

### `DELETE /v1/monitors/{monitor_id}`

**用途:** Monitorを恒久削除します。**入力:** pathの`monitor_id`。**結果:** decoder/inference/RTSP接続を停止し、memory eventを破棄して204；Collectionは削除しません。**エラー:** 401、404。

### `GET /v1/monitors/{monitor_id}/state`

**用途:** headless clientが現在状態をpollします。**入力:** pathのMonitor ID。**結果:** 200で接続、実効FPS、処理時間、skip、現在の認識/未登録face、preview、再接続、エラーを返し、embeddingは返しません。**エラー:** 401、404。

### `GET /v1/monitors/{monitor_id}/events`

**用途:** 保存しない最近のenter/exit/error/recovery eventを取得します。**入力:** `limit` 1～1000と前回の不透明な`cursor`。**結果:** 200で`events`、`next_cursor`、`truncated`、`stream_reset`；再起動でeventは失われます。**エラー:** 400 `invalid_cursor`、401、404。

### `GET /v1/monitors/{monitor_id}/preview.mjpeg`

**用途:** 既定で無効のraw MJPEG previewを開きます。**入力:** path IDと通常のBearer header；API keyをURLに入れません。**結果:** viewerがいる間だけencodeする長時間`multipart/x-mixed-replace`で、boxはclientが`/state`から描画します。**エラー:** 401、404、409 `preview_disabled`、503。

## クライアント実装チェック

GET は再試行可能です。DELETE は状態確認後に再試行してください。Person/Face作成で通信結果が不明な場合は、同じIDを再送する前にGETで確認します。429/一時503だけを上限付き指数バックオフとjitterで再試行し、4xx入力エラーは修正してください。
