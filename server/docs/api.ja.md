# InsightFace Server REST API 利用ガイド

**言語:** [English](api.md) · [中文](api.zh-CN.md) · 日本語 · [Deutsch](api.de.md) · [Español](api.es.md) · [Français](api.fr.md) · [Русский](api.ru.md) · [Português](api.pt.md) · [한국어](api.ko.md)

この文書は全公開 API の用途、入力、サーバー処理、正常結果、主なエラーを説明します。コンテナとモデルの準備は [ユーザーガイド](user-guide.ja.md)、実行中の厳密な Schema は `/docs` または `/openapi.json` を参照してください。


モデルは `model_id` で識別し、応答に独立した `model_version` は含めません。既存 Collection の `embedding_contract_id` は維持されます。

生体検知を使う場合は、[設定・モデルのインストール・結果の意味](#任意の生体検知-addon)を確認してください。各操作の節にも影響を説明しています。

## 共通規則と最初の呼び出し

- ベースパスは `/v1`、JSON は `snake_case`、画像は JPEG/PNG/WebP/BMP の multipart です。
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

## 任意の生体検知 addon

`server/config/server.toml` では生体検知は既定で無効です。`inference.addons` と `addons.auto_download` は両方 `[]` で、キーのない旧設定も無効のままです。

**コマンドラインで有効化する方法（Server の初回起動前も使用可能）：**

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

`--enable-liveness` は最初に既存設定を更新できるか確認します。基本パッケージ、設定された追加ダウンロード、生体検知モデルをインストール・検証してから、`inference.addons` と `addons.auto_download` に `liveness` を追加し、他の項目、コメント、設定を保持します。検証済みキャッシュを再利用する場合も有効化設定を保存します。ダウンロード失敗時は設定を変更せず、設定保存に失敗するとエラーと非ゼロ終了コードを返します。保存済みの有効なモデルは再試行時に再利用できます。

両 Compose サービスは既存の `server/config` 全体を `/etc/insightface` に書き込み可能でマウントし、`create_host_path: false` を保持します。Server が未起動でもインストーラーがホスト設定を原子的に更新できます。ディレクトリと `server.toml` は必須です。

Server を先に起動する必要はありません。新規環境では次の `up -d` で生体検知が有効になります。すでに実行中の場合は `docker compose -f server/deploy/compose.cpu.yml restart server` が必要で、`up -d` だけでは保存済み設定を再読込しません。CUDA では `compose.cuda12.yml` を使います。

`--enable-liveness` がない場合、`models install` は従来どおり設定を書き換えず、既定の生体検知は無効です。`models addons install liveness` はダウンロードと検証のみで、有効化しません。以下の **システム → 生体検知** から有効化する方法も使えます。

**システム → 生体検知** で再起動後に有効にするダウンロード操作を選びます。公開モデルの SHA-256 を検証後、両リストに `liveness` を追加し、他の項目、コメント、設定を保持します。検証済みキャッシュは再利用します。現在の処理は変わらず、**手動で Server を再起動**すると有効になります。失敗時はエラーと再試行を表示し、ダウンロード失敗では設定を有効にしません。

[Web ダウンロードのマウントと権限](user-guide.ja.md#web-ダウンロードのマウントと権限).

**高度な使い方：手動設定。** 以下は有効化フラグや Web 操作の代替です。再起動前にモデルをインストールしてください。

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

### モデルのインストールと起動

`inference.addons` は実行時の使用、`addons.auto_download` は基本モデルのインストール時の追加ダウンロードを制御します。後者を `["liveness"]` にすると、基本モデルがキャッシュ済みでも addon を追加します。起動時のダウンロードはありません。インストーラーと Server は同じ設定ファイルを読みます。

[ユーザーガイドの初期設定](user-guide.ja.md)にある現在の Compose と既存設定を使用します。Server とインストーラーは root で単一の書き込み可能なモデルマウントを共有します。Compose がモデルルートを作成し、明示的な addon ダウンロード時に `addons/` を作成します。ホスト UID/GID や手動の権限設定は不要です。

```bash
docker compose -f server/deploy/compose.cpu.yml run --rm models addons install liveness
docker compose -f server/deploy/compose.cpu.yml run --rm models addons verify liveness
docker compose -f server/deploy/compose.cpu.yml up -d --force-recreate
```

有効なモデルがない場合は `addon_model_missing`、不正なファイルは `addon_model_invalid` で起動を停止します。設定した addon を自動で無効にはしません。

### 生体検知の結果

| 結果 | `status` | `is_live` | `live_score` |
| --- | --- | --- | --- |
| 生体 | `ok` | `true` | `[0, 1]` |
| なりすまし | `ok` | `false` | `[0, 1]` |
| 入力拒否 | `input_rejected` | `null` | `null` |

位置合わせ後の顔の周囲に元画像の有効な領域が不足する場合にのみ `input_rejected` を返します。この結果には、ユーザー向けの説明 `liveness.reason` が追加されます。生体と なりすまし の結果には `reason` はありません。FaceAnalysis と API は常に英語の説明を返し、Web UI だけが表示言語に合わせて翻訳します。プログラムの判定には `status` と `is_live` を使い、`reason` の文面を解析しないでください。以前に保存された結果には `reason` がない場合があるため、クライアントは一般的な入力拒否メッセージにフォールバックできます。

```json
{
  "status": "input_rejected",
  "is_live": null,
  "live_score": null,
  "reason": "Insufficient image area around the face for liveness detection. Move the face toward the center, step back from the camera, or use a less tightly cropped image."
}
```

`normal` は生体検知に合格した顔だけを認識します。`observe` は結果を記録して認識を続けます。無効な場合、`liveness` 自体を省略します。基本の3項目は `status`、`is_live`、`live_score` です。合格と なりすまし は `status: ok`、真偽値とスコアを返します。入力拒否は `status: input_rejected`、残り2項目は `null` です。

検出は負の結果も HTTP 200 で返します。`normal` の特徴抽出・比較・検索は HTTP 422 `liveness_fake` または `liveness_input_rejected` と `error.details.liveness` を返します。比較には `details.side` もあります。推論障害は HTTP 503 `liveness_unavailable` です。 実行時の障害は `normal` と `observe` のどちらでも処理を中止し、`input_rejected` には変換しません。

新規 Person の登録と FaceSample の追加では、生体検知は初期状態で省略されます。`[inference].liveness_on_registration=false` の場合、モデルを実行せず、新しいサンプルの `liveness` を省略します。`true` にすると addon が有効な場合に `normal`/`observe` に従い、拒否項目には `reason` と `liveness` が付きます。`review_mode` の品質審査と外部特徴量の検証は引き続き実行されます。登録時の生体検知を有効にした場合、`review_mode=off` と `external_trusted` でも回避できません。リクエストからこの設定を上書きすることはできません。過去に保存された生体検知結果は引き続き参照できます。

RTSP では `liveness_blocked` を `unknown` と区別し、`liveness_blocked_faces` に計上します。ブロックされた顔に人物・不明者の入場イベントを発行せず、確認フレーム数をリセットします。推論障害時は古い認識表示を消します。

`liveness_compare_scope` は `/v1/compare` の検査対象を指定します。`both`（既定）は両方、`source` は比較元、`target` は比較先を検査します。`live_score >= liveness_threshold` で生体と判定します。

`models addons install liveness` は公開モデルを `/models/addons/liveness.onnx` に保存します。Compose ホスト側では `server/.models/addons/liveness.onnx` です。起動時のエラーは `addon_model_missing` と `addon_model_invalid` です。`/v1/models` と `/v1/system` は有効な追加モデルを `addons` に返します。

[設定と操作手順](user-guide.ja.md#任意の生体検知-addon).

## システム

### `GET /v1/health`

**用途/入力:** 公開 readiness。パラメーターなし、認証不要。**処理/結果:** DB quick check と起動状態を確認し、200 で `status`、`auth_enabled`、`request_id`。**エラー:** 未準備は `503 not_ready`。

### `GET /v1/system`

**用途/入力:** 安全な運用診断。パラメーターなし。**結果:** 200 で OS/CPU/GPU、Driver、CUDA/cuDNN/ORT、Provider、モデル、DB、mount、件数、検索 backend、安全設定、推論並行数。秘密・画像・embedding は含みません。**エラー:** 401、503。

### `GET /v1/models`

`addons` は有効な addon を基本モデルとは別に返します。`liveness` の有無と、システム応答の `safe_config` にある実際の設定を確認してください。これらは読み取り専用で、モデルをインストールしません。

**用途/入力:** 検証済み detector/recognizer、実 Provider、License を確認。パラメーターなし。**結果:** 200 の `models`、`execution_provider`、`license`。**エラー:** 401。

基本モデルの `raccoon_s` と `raccoon_l` は CPU と CUDA で利用でき、起動前にモデルツールでインストールします。この API は実行中のモデル構成を表示し、ダウンロード一覧ではありません。以下の Web 操作は生体検知だけを管理します。Collection は認識モデルと前処理の契約に結び付いています。基本モデルを変えても既存の特徴量は変換されず、`409 collection_model_mismatch` になる場合があります。生体検知の有効化だけでは契約は変わりません。

### `GET /v1/addons/liveness`

**用途:** モデルの保存状態と次回起動の設定を読み取ります。ダウンロードや設定変更は行いません。これは管理用であり、単独の生体検知推論 API ではありません。

**結果:** HTTP 200. `enabled` は現在のプロセスでの有効状態です。`installed` は公開モデルの SHA-256 検証に合格したファイルがあることを示し、有効化とは別です。`configured_enabled` は現在の設定ファイルから読み取った次回起動の選択で、`restart_required` はそれが `enabled` と異なることを示します。再起動までは `/v1/system` の `safe_config` は実行中の設定を示します。

`state` は `idle`（検証済みモデルなし）、`downloading`（準備中）、`ready`（検証済みモデルあり）、`error`（準備・ファイル・設定のエラー）のいずれかです。`ready` だけでは設定保存や再起動の完了を意味しません。

`can_enable` は Web から準備できるかを示します。利用できない場合、`unavailable_code` は安定した理由コード、`unavailable_reason` は説明文です。それ以外は両方 `null` です。`error` は `null` または `code` と `message` を持つオブジェクトです。`model_path` はローカルモデルのパス、`config_file` は選択した TOML のパスまたは `null` です。応答には `request_id` も含まれます。

`unavailable_code` の値は、`config_file_missing`（設定ファイル未指定）、`config_file_not_regular`（通常ファイルではない）、`config_file_mount`（単一ファイルのマウント）、`config_not_writable`（設定を書き込めない）、`addon_directory_not_writable`（追加モデルのディレクトリを書き込めない）、`addon_config_invalid`（無効な設定）、`addon_model_invalid`（無効なモデル）、`server_stopping`（終了中）です。

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

**用途:** モデルをダウンロードし、次回起動の設定を保存します。`Content-Type: application/json` と空のオブジェクト `{}` を送ります。モデル URL や他のパラメーターは受け付けません。

```bash
curl -sS "${BASE_URL}/v1/addons/liveness" -H "${AUTH_HEADER}"
curl -sS "${BASE_URL}/v1/addons/liveness/enable" -H "${AUTH_HEADER}" \
  -H 'Content-Type: application/json' -d '{}'
```

**結果:** HTTP 202 は GET と同じ状態フィールドを返します。処理の受付を示すもので、有効化の完了ではありません。準備が終わるまで `GET /v1/addons/liveness` を繰り返し確認します。重複要求は実行中の処理を共有し、ブラウザーを閉じても処理は続きます。

ダウンロードと SHA-256 検証に合格してから、`config_file` の `[inference].addons` と `[addons].auto_download` に `liveness` を追加します。他の値やコメントは保持し、検証済みの保存ファイルは再利用します。`installed=true`、`configured_enabled=true`、`restart_required=true` になったら Server を手動で再起動します。新しいプロセスでは `enabled=true`、`restart_required=false` になります。動作中の再読み込みや基本モデル切り替え API はありません。

**エラー:** 要求エラーは通常の形式です。空でない本文は `400 invalid_addon_request`、認証失敗は `401 unauthorized`、許可されないブラウザーの送信元は `403 origin_not_allowed`、パス・権限・設定の問題は `409 addon_management_unavailable`、JSON 以外は `415 json_required` です。ブラウザーは同一オリジン、または明示的に許可された CORS オリジンから接続します。

受付後の失敗は HTTP 200 の GET 応答内の `state=error` と `error.code` で確認します。`addon_download_failed` では設定を変更しません。Server のネットワークやプロキシを確認してください。`addon_config_save_failed` では設定やディレクトリ権限を修正し、検証済みファイルを再利用できます。`addon_config_invalid` はディスク上の TOML が無効です。`addon_model_invalid` は不正な保存ファイルの交換または削除が必要で、自動上書きしません。`addon_job_in_progress` は別プロセスの準備完了を待って再確認します。原因を修正してから POST を再送します。

## ステートレス顔処理

### `POST /v1/detect`

有効時は評価した顔に `liveness.status`、`liveness.is_live`、`liveness.live_score` を返します。なりすまし と `input_rejected` も HTTP 200 で、認識特徴は抽出しません。`input_rejected` は顔の周囲の画像領域が不足していることを示し、`liveness.reason` が画像の調整方法を説明します。`liveness` がなければ未評価です。

**入力:** multipart `image` 必須、`max_faces` 1～100、任意 `collection_id`。**処理/結果:** 複数解像度候補を統合して一度 NMS、面積順の `faces`、box/5点/score/quality、`processing_ms`。顔なしは 200 の空配列。**エラー:** 400 旧 min_score、404 Collection、413、422 invalid_image、503。

```bash
curl -sS "${BASE_URL}/v1/detect" -H "${AUTH_HEADER}" -F 'image=@group.jpg' -F 'max_faces=10'
```

### `POST /v1/compare`

`liveness_compare_scope`（`both`・`source`・`target`）で指定した側を認識前に検査します。`normal` では HTTP 422 `liveness_fake` または `liveness_input_rejected` と `error.details.liveness`、`error.details.side` を返し、類似度は返しません。`observe` は比較を続行し、各評価済みの顔に結果を返します。

**入力:** multipart `source`、`target` 必須、`threshold` 0～1、任意 `collection_id`。**処理/結果:** 各画像から設定戦略で1顔を選び、200 で `matched`、cosine `similarity`、実 threshold、両 face、処理時間。**エラー:** 404、413、422 invalid_image/face_not_found、503。

### `POST /v1/embeddings`

生体検知が有効な `normal` ではなりすまし・入力拒否に HTTP 422 `liveness_fake` / `liveness_input_rejected` と `error.details.liveness` を返し、特徴を抽出しません。`observe` は特徴と顔の生体検知結果を返します。

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

Person の新規登録と FaceSample の追加では既定で生体検知を省略します（`liveness_on_registration=false`）。有効にすると `normal` はなりすまし・入力拒否を拒否し、`observe` は結果を保持して続行します。品質審査は選択した `review_mode` に従い、拒否一覧は実際の `reason` と生体検知結果を別々に表示します。

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

Person の新規登録と FaceSample の追加では既定で生体検知を省略します（`liveness_on_registration=false`）。有効にすると `normal` はなりすまし・入力拒否を拒否し、`observe` は結果を保持して続行します。品質審査は選択した `review_mode` に従い、拒否一覧は実際の `reason` と生体検知結果を別々に表示します。

**入力:** path IDs、repeatable images と review/embedding fields（Person作成と同じ）。**結果:** 201 の `faces` と `rejected_images`、部分成功可。**エラー:** 登録エラーと 404 Person。

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces`

**入力:** path IDs、query limit 1～100 と cursor。**結果:** 200 の metadata `faces`、`has_crop`、`next_cursor`。embedding/画像byteは返しません。**エラー:** 400 cursor、404。

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}/image`

**入力:** 3 path IDs。**結果:** 保存済みの場合 200 `image/jpeg` 112×112 crop、`Cache-Control:no-store`。request ID はheaderのみ。**エラー:** 401、404 face/face_image_not_found。

### `DELETE /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}`

**入力:** 3 path IDs。**結果:** embedding/crop/索引行を削除し204。**エラー:** 404、503。

## 検索

### `POST /v1/collections/{collection_id}/search`

生体検知が有効な `normal` ではなりすまし・入力拒否に HTTP 422 `liveness_fake` / `liveness_input_rejected` と `error.details.liveness` を返し、検索しません。これは検索成功時の空リストとは異なります。`observe` は検索を続行し、クエリ顔に結果を返します。

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

生体検知が有効な `normal` では拒否した顔を `status: liveness_blocked` とし、生体検知結果を別に表示します。`liveness_blocked_faces` に計上し、`unknown_faces` や入場イベントには含めません。`observe` は認識を続けます。入力拒否と なりすまし は別々に表示されます。

**用途:** headless clientが現在状態をpollします。**入力:** pathのMonitor ID。**結果:** 200で接続、実効FPS、処理時間、skip、現在の認識/未登録face、preview、再接続、エラーを返し、embeddingは返しません。**エラー:** 401、404。

### `GET /v1/monitors/{monitor_id}/events`

**用途:** 保存しない最近のenter/exit/error/recovery eventを取得します。**入力:** `limit` 1～1000と前回の不透明な`cursor`。**結果:** 200で`events`、`next_cursor`、`truncated`、`stream_reset`；再起動でeventは失われます。**エラー:** 400 `invalid_cursor`、401、404。

### `GET /v1/monitors/{monitor_id}/preview.mjpeg`

**用途:** 既定で無効のraw MJPEG previewを開きます。**入力:** path IDと通常のBearer header；API keyをURLに入れません。**結果:** viewerがいる間だけencodeする長時間`multipart/x-mixed-replace`で、boxはclientが`/state`から描画します。**エラー:** 401、404、409 `preview_disabled`、503。

## クライアント実装チェック

GET は再試行可能です。DELETE は状態確認後に再試行してください。Person/Face作成で通信結果が不明な場合は、同じIDを再送する前にGETで確認します。429/一時503だけを上限付き指数バックオフとjitterで再試行し、4xx入力エラーは修正してください。
