# InsightFace Server ユーザーガイド

**言語:** [English](user-guide.md) · [中文](user-guide.zh-CN.md) · 日本語 · [Deutsch](user-guide.de.md) · [Español](user-guide.es.md) · [Français](user-guide.fr.md) · [Русский](user-guide.ru.md) · [Português](user-guide.pt.md) · [한국어](user-guide.ko.md)

このガイドは初めて利用する方のために、空の作業ディレクトリから最初の検索成功までを順番に説明します。同じ機能は Web UI、`/v1` API、Python SDK から利用できます。全 HTTP 項目とレスポンスは [API 利用ガイド](api.ja.md) を参照してください。

## ゼロから起動して最初の検索を行う

CPU 版には Linux x86_64、Docker Engine、Docker Compose が必要です。CUDA 版には対応 NVIDIA Driver と NVIDIA Container Toolkit も必要ですが、ホスト側 CUDA、cuDNN、ORT、Python、OpenCV は不要です。

```bash
mkdir -p server/.models
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml run --rm models install buffalo_l
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

GPU では `compose.cuda12.yml` とポート `18098` を使用します。モデルのダウンロード前にライセンスが表示されます。公開済み InsightFace 事前学習モデルは、別途商用ライセンスがない限り非商用研究用途に限定されます。

同梱 Compose は隔離評価向けに認証を既定で無効にしています。有効化する場合は起動前に `INSIGHTFACE_AUTH_ENABLED=true` と長い `INSIGHTFACE_API_KEY` を設定します。UI を開き、Dashboard確認 → Collection作成 → Person登録 → 別画像でSearchの順に進めます。停止は `docker compose ... down` を使用し、データを保持する場合は `-v` を付けないでください。

## 1. ログインと準備確認

CPU は `http://SERVER:18097/`、CUDA 12 は `http://SERVER:18098/` を開きます。認証が有効な場合は **API キーを設定** から管理者の Key を入力します。Key は現在のタブのメモリだけに保持され、再読み込みまたはタブを閉じると消えます。

**ダッシュボード** または **システム** で、サービス、データベース、モデル、Provider が ready であることを確認します。CUDA 版は `CUDAExecutionProvider` を表示しなければならず、CPU へ自動フォールバックしません。

## 2. Collection を作成

**コレクション** → **新規コレクション** で、安定した ID、名前、既定 cosine
しきい値（初期値 `0.4`）、利用可能な検索プロファイル、容量、人物ごとの最大
FaceSample 数を設定します。112×112 にリサイズした `bounding-box crop` JPEG
の保存は既定でオフです。これは認識モデル用のアライン済み入力ではありません。

Collection はモデル ID、バージョン、digest、次元、前処理に固定されます。モデル変更後も古い Collection は表示されますが、契約が異なる登録・検索は明示的に拒否されます。

検出設定は作成時にシステム既定値をコピーし、入力サイズ、検出/NMS しきい値、単一顔戦略を後から変更できます。`largest` は面積優先、`center_largest` は `面積 - 2.0 × 顔枠中心と画像中心のピクセル距離の二乗` を最大化します。検出信頼度はこのスコアに含みません。

## 3. Person を登録

**人物** で Collection を選び、**人物を登録** を開きます。ID、名前、外部 ID、JSON metadata と 1 枚以上の JPEG、PNG、または WebP を指定します。

- `off`: Collection の単一顔戦略を使用し、複数顔を許可します。
- `standard`: 1 つの有効顔を要求し、サイズ、検出値、鮮明度、明るさ、姿勢を確認します。
- `strict`: standard に加え、最良の人物内 similarity が最良の人物外 similarity より高いことを要求します。

一括登録は部分成功を返します。拒否理由を確認してから再試行してください。元画像は保存されません。`external_trusted` では L2 正規化済み embedding を利用でき、画像は品質確認に必要ですが特徴量の再抽出は行いません。

## 4. 検出・比較・検索

**検出** は顔矩形、5 点、検出値、品質を表示し、顔なしは空リストで成功します。**比較** は選択したシステムまたは Collection の戦略で各画像から 1 顔を選び、cosine `similarity`、`threshold`、`matched` を返します。Similarity は確率ではありません。

**検索** で Collection と画像を選択します。Collection の戦略で顔を選び、人物の全 FaceSample 中の最高 similarity を人物スコアとして降順に返します。一致なしは空リストです。新規 FaceSample は SQLite へ commit 後、応答前にメモリ索引へ追加されます。再起動時は SQLite から再構築します。

## 5. RTSP カメラ監視

**カメラ監視** で永続的な Monitor を作成し、RTSP 接続先、Collection、推論頻度、任意のしきい値、event policy を設定します。Preview は既定で無効で、無効でも認識と event は継続します。有効時は Web UI が raw frame と `/state` から緑の登録人物枠、オレンジの未登録顔枠を描画します。

Monitor はブラウザーと独立して動作し、有効な task は Server 再起動後に復元されます。設定は SQLite、RTSP 認証情報は `/data` に暗号化保存されますが、動画 frame と event は保存しません。Event は上限付き memory buffer だけに残り、再起動で失われます。Decoder は最新 frame だけを保持し、遅い処理では古い frame を queue に積まず skip します。

## 6. データと安全性

`/data` を永続化し、`/models` は読み取り専用にします。大量削除前に SQLite と顔画像領域を一緒にバックアップしてください。API Key は hash 保存され、同じデータ volume で異なる `INSIGHTFACE_API_KEY` を指定して再起動すると Key がローテーションされます。画像、embedding、Key をログへ出力しないでください。

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
`buffalo_m`、`buffalo_sc`、`antelopev2` です。インストール後は
`manifest.json` と署名済み `MODEL.LICENSE` が残ります。
`--accept-license` がない場合は条項を表示してダウンロードせず終了します。
公開 InsightFace 学習済みモデルは、別途商用ライセンスがない限り非商用研究用です。

## 8. 起動設定と検索

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

完全なリポジトリからユーザー自身でビルドできます。

```bash
make -C server build-cpu
make -C server build-cuda12
```

ローカルイメージを使う Compose 操作には `--pull never` を付けます。公開固定 Tag は
`0.2.0-cpu` と `0.2.0-cuda12`、移動 Tag は `cpu` と `cuda12` で、`latest` は
ありません。アップグレード前に書き込みを止め、SQLite-safe な方法で `/data` と
crop を一緒にバックアップしてください。`docker compose down -v` は Volume を
削除するため使わないでください。

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
