# InsightFace Server ユーザーガイド

**言語:** [English](user-guide.md) · [中文](user-guide.zh-CN.md) · 日本語 · [Deutsch](user-guide.de.md) · [Español](user-guide.es.md) · [Français](user-guide.fr.md) · [Русский](user-guide.ru.md) · [Português](user-guide.pt.md) · [한국어](user-guide.ko.md)

このガイドは初めて利用する方のために、空の作業ディレクトリから最初の検索成功までを順番に説明します。同じ機能は Web UI、`/v1` API、Python SDK から利用できます。全 HTTP 項目とレスポンスは [API 利用ガイド](api.ja.md) を参照してください。

モデルは `model_id` で識別し、応答に独立した `model_version` は含めません。

同じ認識モデルと特徴契約で Server を更新する場合、既存 Collection の `embedding_contract_id`、サンプル、特徴量は維持されます。異なる認識モデルへの変更は別の移行で、契約が不一致なら登録・検索は `collection_model_mismatch` になります。

生体検知を使う場合は、[設定・モデルのインストール・結果の意味](#任意の生体検知-addon)を確認してください。各操作の節にも影響を説明しています。

## ゼロから起動して最初の検索を行う

CPU 版には Linux x86_64、Docker Engine、Docker Compose が必要です。CUDA 版には対応 NVIDIA Driver と NVIDIA Container Toolkit も必要ですが、ホスト側 CUDA、cuDNN、ORT、Python、OpenCV は不要です。

リポジトリのルートで実行し、`server/config/server.toml` を用意してください。Server とモデルインストーラーは root（`0:0`）で実行します。Compose は `server/.models` がなければ作成し、単一の書き込み可能な `/models` としてマウントします。`addons/` は addon のダウンロードを要求したときに作成されます。UID/GID の export、addon ディレクトリの手動作成、権限設定は不要です。

```bash
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml run --rm models install buffalo_l
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

モデルのインストールと同時に生体検知を設定する場合は、代わりに次のコマンドを使います：

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

Server を先に起動する必要はありません。新規環境では次の `up -d` で生体検知が有効になります。すでに実行中の場合は `docker compose -f server/deploy/compose.cpu.yml restart server` が必要で、`up -d` だけでは保存済み設定を再読込しません。CUDA では `compose.cuda12.yml` を使います。

GPU では `compose.cuda12.yml` とポート `18098` を使用します。モデルのダウンロード前にライセンスが表示されます。公開済み InsightFace 事前学習モデルは、別途商用ライセンスがない限り非商用研究用途に限定されます。

同梱 Compose は隔離評価向けに認証を既定で無効にしています。有効化する場合は起動前に `INSIGHTFACE_AUTH_ENABLED=true` と長い `INSIGHTFACE_API_KEY` を設定します。UI を開き、Dashboard確認 → Collection作成 → Person登録 → 別画像でSearchの順に進めます。停止は `docker compose ... down` を使用し、データを保持する場合は `-v` を付けないでください。

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

**システム → 生体検知** で **ダウンロードして再起動後に有効化** を選びます。公開モデルの SHA-256 を検証後、両リストに `liveness` を追加し、他の項目、コメント、設定を保持します。検証済みキャッシュは再利用します。現在の処理は変わらず、**手動で Server を再起動**すると有効になります。失敗時はエラーと再試行を表示し、ダウンロード失敗では設定を有効にしません。

システム画面は検証済みファイルの有無（`installed`）、現在の実行状態（`enabled`）、保存した次回起動設定（`configured_enabled`）、再起動の要否（`restart_required`）を別々に表示します。ダウンロードや設定保存だけでは現在の推論は変わりません。無効に戻すには同じ設定ファイルで `inference.addons=[]` と `addons.auto_download=[]` を保存して手動で再起動します。Web 操作は登録設定を変更せず、その既定値は `liveness_on_registration=false` です。

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

```bash
docker compose -f server/deploy/compose.cpu.yml run --rm models addons install liveness
docker compose -f server/deploy/compose.cpu.yml run --rm models addons verify liveness
docker compose -f server/deploy/compose.cpu.yml up -d --force-recreate
```

有効なモデルがない場合は `addon_model_missing`、不正なファイルは `addon_model_invalid` で起動を停止します。設定した addon を自動で無効にはしません。

### Web ダウンロードのマウントと権限

標準 Compose は Server とモデルインストーラーに単一の書き込み可能な `/models`
マウントを提供し、`create_host_path: true` を設定します。両方を root（`0:0`）と
Docker 既定の capabilities で実行し、`cap_drop: [ALL]` は指定しません。
コンテナのルートファイルシステムは読み取り専用のまま、`no-new-privileges` も保持します。
ホスト UID/GID や `chmod 777` の設定は不要です。root は書き込み可能なマウント内の
ファイルを変更でき、新規ダウンロードの所有者がホスト側で root になる場合があります。

両サービスが既存の `server/config` 全体を `/etc/insightface` に書き込み可能でマウントし、
Web 操作と `--enable-liveness` が `server.toml` を原子的に保存します。このディレクトリと
ファイルは必須で、Compose は設定元を自動作成しません。独自の配置では実際のパスを指定し、
両サービスに同等の書き込み可能なディレクトリマウントを
用意してください。独自の読み取り専用マウントは既存モデルの推論には使えますが、Web からの
ダウンロードや設定保存には使えません。

CUDA では `compose.cuda12.yml` を使用します。ファイルが存在するだけでは生体検知は
有効になりません。Web 保存後は `docker compose -f server/deploy/compose.cpu.yml restart server`
で反映します。マウント、実行ユーザー、capabilities、プロキシ環境変数を変えた場合は
コンテナを再作成してください。

ダウンロードにプロキシが必要なら、コンテナ作成前に `HTTP_PROXY`、`HTTPS_PROXY`、`NO_PROXY` を設定します。Compose は Server とモデルツールの両方へ渡します。プロキシにはコンテナから到達できる LAN アドレスを使います。コンテナ内の `127.0.0.1` は Mac ではありません。この操作は既存の API Key 認証を使い、認証が無効なら API に接続できる利用者も実行できます。対象は公開された固定の生体検知モデルだけで、任意の URL の入力や基本モデルパッケージの切り替えは提供しません。

### 生体検知の結果

| 結果 | `status` | `is_live` | `live_score` |
| --- | --- | --- | --- |
| 生体検知合格 | `ok` | `true` | `[0, 1]` |
| 非生体 | `ok` | `false` | `[0, 1]` |
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

`normal` は生体検知に合格した顔だけを認識します。`observe` は結果を記録して認識を続けます。無効な場合、`liveness` 自体を省略します。基本の3項目は `status`、`is_live`、`live_score` です。合格と fake は `status: ok`、真偽値とスコアを返します。入力拒否は `status: input_rejected`、残り2項目は `null` です。

検出は負の結果も HTTP 200 で返します。`normal` の特徴抽出・比較・検索は HTTP 422 `liveness_fake` または `liveness_input_rejected` と `error.details.liveness` を返します。比較には `details.side` もあります。推論障害は HTTP 503 `liveness_unavailable` です。 実行時の障害は `normal` と `observe` のどちらでも処理を中止し、`input_rejected` には変換しません。

新規 Person の登録と FaceSample の追加では、生体検知は初期状態で省略されます。`[inference].liveness_on_registration=false` の場合、モデルを実行せず、新しいサンプルの `liveness` を省略します。`true` にすると addon が有効な場合に `normal`/`observe` に従い、拒否項目には `reason` と `liveness` が付きます。`review_mode` の品質審査と外部特徴量の検証は引き続き実行されます。登録時の生体検知を有効にした場合、`review_mode=off` と `external_trusted` でも回避できません。リクエストからこの設定を上書きすることはできません。過去に保存された生体検知結果は引き続き参照できます。

RTSP では `liveness_blocked` を `unknown` と区別し、`liveness_blocked_faces` に計上します。ブロックされた顔に人物・不明者の入場イベントを発行せず、確認フレーム数をリセットします。推論障害時は古い認識表示を消します。

`liveness_compare_scope` は `/v1/compare` の対象を `both`（既定）、`source`、`target` から選びます。`live_score >= liveness_threshold` なら合格です。

モデルはホストの `server/.models/addons/liveness.onnx`、コンテナの `/models/addons/liveness.onnx` に配置します。`/v1/models` と `/v1/system` の `addons` は現在有効な addon を示します。

[API の詳細](api.ja.md#任意の生体検知-addon).

## 1. ログインと準備確認

CPU は `http://SERVER:18097/`、CUDA 12 は `http://SERVER:18098/` を開きます。認証が有効な場合は **API キーを設定** から管理者の Key を入力します。Key は現在のタブのメモリだけに保持され、再読み込みまたはタブを閉じると消えます。

**ダッシュボード** または **システム** で、サービス、データベース、モデル、Provider が ready であることを確認します。CUDA 版は `CUDAExecutionProvider` を表示しなければならず、CPU へ自動フォールバックしません。

ダッシュボードはモデル名の下に生体検知の有効・無効を常に表示します。システムではインストール、現在の実行状態、再起動待ちを区別します。

## 2. Collection を作成

**コレクション** → **新規コレクション** で、安定した ID、名前、既定 cosine
しきい値（初期値 `0.4`）、利用可能な検索プロファイル、容量、人物ごとの最大
FaceSample 数を設定します。112×112 にリサイズした `bounding-box crop` JPEG
の保存は既定でオフです。これは認識モデル用のアライン済み入力ではありません。

Collection はモデル ID、digest、次元、前処理に固定されます。モデル変更後も古い Collection は表示されますが、契約が異なる登録・検索は明示的に拒否されます。

検出設定は作成時にシステム既定値をコピーし、入力サイズ、検出/NMS しきい値、単一顔戦略を後から変更できます。`largest` は面積優先、`center_largest` は `面積 - 2.0 × 顔枠中心と画像中心のピクセル距離の二乗` を最大化します。検出信頼度はこのスコアに含みません。

## 3. Person を登録

**人物** で Collection を選び、**人物を登録** を開きます。ID、名前、外部 ID、JSON metadata と 1 枚以上の JPEG、PNG、WebP、または BMP を指定します。

- `off`: Collection の単一顔戦略を使用し、複数顔を許可します。
- `standard`: 1 つの有効顔を要求し、サイズ、検出値、鮮明度、明るさ、姿勢を確認します。
- `strict`: standard に加え、最良の人物内 similarity が最良の人物外 similarity より高いことを要求します。

一括登録は部分成功を返します。拒否理由を確認してから再試行してください。元画像は保存されません。`external_trusted` では L2 正規化済み embedding を利用でき、画像は品質確認に必要ですが特徴量の再抽出は行いません。

Person の新規登録と FaceSample の追加では既定で生体検知を省略します（`liveness_on_registration=false`）。有効にすると `normal` は fake・入力拒否を拒否し、`observe` は結果を保持して続行します。品質審査は引き続き選択した `review_mode` に従い、拒否一覧は実際の `reason` と生体検知結果を別々に表示します。

## 4. 検出・比較・検索

**検出** は顔矩形、5 点、検出値、品質を表示し、顔なしは空リストで成功します。**比較** は選択したシステムまたは Collection の戦略で各画像から 1 顔を選び、cosine `similarity`、`threshold`、`matched` を返します。Similarity は確率ではありません。

**検索** で Collection と画像を選択します。Collection の戦略で顔を選び、人物の全 FaceSample 中の最高 similarity を人物スコアとして降順に返します。一致なしは空リストです。新規 FaceSample は SQLite へ commit 後、応答前にメモリ索引へ追加されます。再起動時は SQLite から再構築します。

有効時は評価した顔に `liveness.status`、`liveness.is_live`、`liveness.live_score` を返します。fake と `input_rejected` も HTTP 200 で、認識特徴は抽出しません。`input_rejected` は顔の周囲の画像領域が不足していることを示し、`liveness.reason` が画像の調整方法を説明します。`liveness` がなければ未評価です。

`liveness_compare_scope`（`both`・`source`・`target`）で指定した側を認識前に検査します。`normal` では HTTP 422 `liveness_fake` または `liveness_input_rejected` と `error.details.liveness`、`error.details.side` を返し、類似度は返しません。`observe` は比較を続行し、各評価済みの顔に結果を返します。

生体検知が有効な `normal` では fake・入力拒否に HTTP 422 `liveness_fake` / `liveness_input_rejected` と `error.details.liveness` を返し、検索しません。これは検索成功時の空リストとは異なります。`observe` は検索を続行し、クエリ顔に結果を返します。

## 5. RTSP カメラ監視

**カメラ監視** で永続的な Monitor を作成し、RTSP 接続先、Collection、推論頻度、任意のしきい値、event policy を設定します。Preview は既定で無効で、無効でも認識と event は継続します。有効時は Web UI が raw frame と `/state` から緑の登録人物枠、オレンジの未登録顔枠を描画します。

Monitor はブラウザーと独立して動作し、有効な task は Server 再起動後に復元されます。設定は SQLite、RTSP 認証情報は `/data` に暗号化保存されますが、動画 frame と event は保存しません。Event は上限付き memory buffer だけに残り、再起動で失われます。Decoder は最新 frame だけを保持し、遅い処理では古い frame を queue に積まず skip します。

生体検知が有効な `normal` では拒否した顔を `status: liveness_blocked` とし、生体検知結果を別に表示します。`liveness_blocked_faces` に計上し、`unknown_faces` や入場イベントには含めません。`observe` は認識を続けます。入力拒否と fake は別々に表示されます。

## 6. データと安全性

`/data`、書き込み可能なモデルルート、設定ディレクトリを永続化します。コンテナのルートファイルシステムは読み取り専用で、Web モデル管理には既存の API 認証を適用します。大量削除前に SQLite と顔画像領域を一緒にバックアップしてください。API Key は hash 保存され、同じデータ volume で異なる `INSIGHTFACE_API_KEY` を指定して再起動すると Key がローテーションされます。画像、embedding、Key をログへ出力しないでください。

開発者向け OpenAPI スキーマエクスプローラーは `/docs`、操作別の API 説明はこのヘルプ内にあります。障害報告には応答ヘッダーの `x-request-id` を含めてください。`401` は Key、`409 collection_model_mismatch` はモデル契約、`422 face_not_found` は有効顔を確認します。

## 7. モデルとライセンス

イメージにはモデルを含みません。通常起動はオフラインで、1 回限りの `models`
サービスが `server/.models` へインストールします。

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models verify buffalo_l
```

対応パッケージは `buffalo_l`（`det_10g.onnx` + `w600k_r50.onnx`）、
`buffalo_m`、`buffalo_s`、`buffalo_sc`、`antelopev2`、`raccoon_s`、
`raccoon_l` です。インストール後は
`manifest.json` と署名済み `MODEL.LICENSE` が残ります。
`--accept-license` を省略した場合、対話型端末ではダウンロード前に確認を求めます。
非対話型のコマンドではこのフラグが必須で、省略するとダウンロードせず終了します。
`models verify` はパッケージの識別情報、署名、有効期間、現在の許諾を検証します。
実行時のライセンス表示の代替処理とは異なり、この明示的な検証には署名済みの
ライセンスファイルが必要です。
公開 InsightFace 学習済みモデルは、別途商用ライセンスがない限り非商用研究用です。

`raccoon_s` と `raccoon_l` はサポート対象です。Server は各パッケージの検出・認識モデルのみをインストールし、Raccoon verifier はロードしません。モデル名だけで識別し、独立したモデルバージョン番号は使いません。基本モデルの変更は Web の生体検知操作では行えません。認識モデルが変わる場合、従来の特徴量を新しいモデルの特徴量として扱わず、適合する Collection を用意してください。

## 8. 起動設定と検索

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

`server/config/server.toml` は起動時に一度だけ読み込まれ、変更にはコンテナ再起動が
必要です。既定値は `input_sizes=[[96,96],[512,512]]`、検出しきい値 `0.50`、
NMS `0.40`、`single_face_selection="largest"`、最大 100 顔です。SCRFD は各解像度
を実行し、元画像座標へ戻した全候補に 1 回だけグローバル NMS を行います。
`max_concurrency="auto"` は CPU 4、CUDA 8 です。`[web].disabled=true` では
`/v1` と `/openapi.json` だけを提供します。

利用可能な検索 Profile は System に表示されます。Collection 作成後は変更できず、
リクエスト単位でも指定できません。

- `fp32_v1`: CPU/CUDA の標準。
- `fp16_v1`: CUDA。
- `bf16_v1`: 対応 CPU または SM80+ CUDA。
- `int8_x736_v1`: CPU/CUDA の推奨 INT8。INT32 で累積。
- `int8_x1000_v1`: 既存 Collection 互換用。

すべての Profile は全 FaceSample を走査する Flat 検索で ANN ではありません。
公開スコアは常に raw cosine です。`capacity_rows` の既定は `100000`、上限ガードは
`10000000`、`max_faces_per_person` は `20` です。512 次元の純ベクトル容量は 1 行
あたり FP32 2,048 byte、FP16/BF16 1,024 byte、INT8 512 byte が目安です。

## 9. SDK、ビルド、データ運用

SDK は path、bytes、file-like object に対応し、`detect`、`compare`、
`create_collection`、`add_person`、`search`、Monitor 操作を型付きで提供します。
詳細な HTTP 契約は [API 利用ガイド](api.ja.md)を参照してください。

完全なローカルソースディレクトリから直接ビルドできます。未コミットの変更が
あっても、`.git` ディレクトリがなくても構いません。Git のコミットやプッシュは
ビルドの前提条件ではありません。

```bash
make -C server build-cpu
make -C server build-cuda12
```

テストに合格したら、テストしたものと同じイメージを公開します。その後、同じ
ソースをコミットしたりコミットを整理したりするだけなら、再ビルドは不要です。
コード、フロントエンドのリソース、同梱のユーザーヘルプなど、イメージに含まれる
ファイルを変更した場合は、再ビルドと検証が必要です。

ローカルイメージを使う Compose 操作には `--pull never` を付けます。公開固定 Tag は
`0.3.1-cpu` と `0.3.1-cuda12`、移動 Tag は `cpu` と `cuda12` で、`latest` は
ありません。アップグレード前に書き込みを止め、SQLite-safe な方法で `/data` と
crop を一緒にバックアップしてください。`docker compose down -v` は Volume を
削除するため使わないでください。

### 0.3.1 へのアップグレード

0.3.1 は Docker デプロイを簡素化します。両サービスを root と Docker 既定の
capabilities で実行し、単一の書き込み可能なモデルマウントを共有します。モデルルートが
なければ Compose が作成し、明示的な addon ダウンロード時に `addons/` を作成します。
ホスト UID/GID や共有グループの準備は不要です。

0.3.0 から `raccoon_s`、`raccoon_l` とそのモデル記述、生体検知、Web addon
インストール、BMP 入力に対応しています。Server は Raccoon の検出・認識モデルを使い、
verifier はロードしません。0.3.1 でこれらの機能や API 応答の契約は変わりません。

**1.** `server/config/server.toml` の設定とデプロイ環境固有の上書き設定を保持し、Server の
ソースと Compose ファイルを 0.3.1 に更新します。既存のモデルパス、`/data` の
ボリューム名、顔画像の保存先、ポート、API キー設定を維持してください。独自の
Compose ファイルを使う場合は、`server` と `models` 両サービスのイメージを環境に
合わせて `0.3.1-cpu` または `0.3.1-cuda12` に更新します。以下のコマンドにも、
普段と同じ Compose ファイル、上書き設定、プロジェクト名を適用してください。

イメージタグの変更だけでは不十分です。独自の Compose と上書きファイルも更新し、
両サービスを `user: "0:0"` に設定して、`cap_drop: [ALL]` と旧 UID/GID・共有グループ
設定を削除してください。`/models` は `create_host_path: true` の単一の書き込み可能な
マウントにし、独立した `/models/addons` マウントを削除します。ルートファイルシステムの
読み取り専用と `no-new-privileges` は保持します。既存の設定ディレクトリとファイルも
保持してください。両サービスで設定ディレクトリ全体の書き込みが必要です。インストーラーの
旧読み取り専用ファイルマウントをディレクトリマウントに置き換えます。既存モデルと addon キャッシュは
そのまま利用でき、標準構成では再帰的な権限変更や addon ディレクトリの事前作成は不要です。

**2.** 新しいイメージを取得し、Server コンテナを再作成します。リポジトリのルートで、
既存の環境に合うコマンドを選びます。

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

ローカルでビルドする場合は、先に 0.3.1 のイメージをビルドし、取得する代わりに
`up -d --no-build --pull never --force-recreate server` を使います。
`docker compose restart` だけでは新しいイメージへの切り替えやマウント変更は
適用されません。

**3.** 起動時にデータベースの移行が自動で適用されます。`/v1/health` が `ready` と
バージョン `0.3.1` を返すまで待ち、**システム**でモデルと実行プロバイダーが
想定どおりか確認します。既存の Collection と人物が残っていることを確かめ、
既知の画像で検索してください。同じモデルと特徴量の契約を維持する場合、サンプル、
特徴量、Collection の契約 ID は保持され、再登録は不要です。

**アップグレード後も生体検知は任意です。** 同梱設定と追加モデルのキーがない旧設定は
どちらも生体検知を無効にするため、更新するだけならモデルの追加ダウンロードは不要です。
Server は起動時にモデルをダウンロードしません。有効にするには
[生体検知の設定](#任意の生体検知-addon)に従って
[Web ダウンロードのマウントと権限](#web-ダウンロードのマウントと権限)を準備し、
**システム → 生体検知 → ダウンロードして再起動後に有効化** を選びます。
モデルのインストールと設定保存が成功してから、Server を手動で再起動してください。
既定値は `normal`、しきい値 `0.8`、`liveness_on_registration=false` です。
モデルの保存先は `<models_dir>/addons/liveness.onnx` です。

**Raccoon の導入は別のモデル変更です。** Server を更新しても、現在のモデルパッケージは
変わりません。`raccoon_s` または `raccoon_l` を使う場合は、
[モデルのインストール手順](#7-モデルとライセンス)に従って別のモデルディレクトリへ
インストールし、その保存先を使う環境を設定します。Collection は新しいモデルの特徴量の
契約に適合する必要があります。対応する Collection を作って再登録するか、別途データ移行を
行ってください。Web UI から基本モデルパッケージを切り替えることはできません。

**0.3.0 以降の API と SDK の互換性:** モデル、Collection、FaceSample の結果には `model_version` が
含まれなくなります。モデルは `model_id`、Collection の互換性は `embedding_contract_id` で
識別します。削除されたフィールドを必須とするクライアントを修正し、同梱の Python クライアントを
更新する場合は SDK `0.3.1` を使ってください。生体検知を実行した場合、`liveness` には
`status`、`is_live`、`live_score` の基本3項目が含まれ、`input_rejected` の場合だけ `reason` が追加されます。実行しなかった場合は `liveness` 自体を省略します。
認識リクエストに生体検知を有効にする前に、[結果とエラーの扱い](#生体検知の結果)を確認してください。

## 10. GPU、ネットワーク、トラブルシュート

CUDA イメージは CUDA Runtime 12.9.1、cuDNN 9.24.0、
`onnxruntime-gpu==1.27.0` を含みます。Turing/Ampere/Ada/Hopper は Driver R535
以上、Blackwell/RTX 50 は 570.26 以上、新規導入は安定版 R580 以上を推奨します。
起動時に GPU、Compute Capability、Driver、CUDA/cuDNN/ORT、Provider、実モデル
Session と warm-up を検証し、CPU への暗黙 fallback は拒否します。

ネットワーク公開時は信頼できる Reverse Proxy で HTTPS を終端し、CORS origin、
rate/body/timeout を制限してください。画像、embedding、Key をログに残さず、
`/data` とバックアップを生体情報として保護します。Phase 1 は権限区別のない単一
API Key であり、マルチテナント認可機能ではありません。
