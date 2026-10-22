# InsightFace Server

**言語:** [English](README.md) · [中文](README.zh-CN.md) · 日本語 · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Русский](README.ru.md) · [Português](README.pt.md) · [한국어](README.ko.md)

> **GPU 1基で50M+の顔ベクトル。INT8特徴量子化による高速検索を、実質的な精度低下なく実現。**

**Web UI、分かりやすい REST API、SQLite、ローカル CPU または NVIDIA GPU
推論を1つのコンテナで提供するセルフホスト型顔認識サーバーです。**

```text
画像をアップロード -> 検出、生体検知、比較、登録、検索
```

> **モデルライセンス:** 公開 InsightFace 学習済みモデルは通常、非商用研究用途に
> 限定されます。商用利用には個別の許諾が必要です。
> [InsightFace 公式サイト](https://www.insightface.ai)からお問い合わせください。

InsightFace Server は、管理下のインフラで一般的な顔認識を行うための、AWS
Rekognition よりシンプルでプライバシー重視の選択肢です。画像、Embedding、
モデル、索引をネットワーク内に保持できます。AWS 互換製品ではなく、SigV4、
IAM、Region、AWS リソース意味論は実装しません。

現在のリリース: **0.3.1**、Linux x86_64。

| Runtime | Image |
| --- | --- |
| CPU | `ghcr.io/deepinsight/insightface-server:0.3.1-cpu` |
| NVIDIA GPU | `ghcr.io/deepinsight/insightface-server:0.3.1-cuda12` |

移動タグ `cpu` と `cuda12` は各 Runtime の最新安定版を示します。曖昧な
`latest` は提供しません。リリース方針は
[Maintainer Guide — English](docs/maintainer-guide.md)を参照してください。

**0.3.1 のデプロイ更新:** Server とモデルインストーラーを root で実行し、
モデルを単一の書き込み可能なマウントにまとめ、ダウンロード時に addon ディレクトリを
自動作成します。ホスト UID/GID や共有グループの設定は不要です。イメージタグと
Compose の両方を更新し、既存のモデル、設定、データを保持してください。0.3.0 から
Raccoon のモデル記述、生体検知、BMP 入力に対応し、API/SDK の `model_version` を
廃止しています。[アップグレード手順](docs/user-guide.ja.md#031-へのアップグレード)を参照してください。

![英語版 InsightFace Server Dashboard](docs/images/customer/dashboard-en.jpg)

## 主な機能

- SCRFD 検出、5点ランドマーク、アラインメント、ArcFace Embedding、L2
  normalization、元の cosine similarity、厳密な 1:N Person 検索。
- 複数解像度の候補を統合して1回の NMS を実行し、`largest` と
  `center_largest` の単一顔選択をサポート。
- `Collection -> Person -> FaceSample`、モデルに固定された Collection、
  複数画像登録、部分成功、metadata、明確な拒否理由。
- 登録 `review_mode` は `off`、`standard`、`strict`。`external_trusted`
  の事前計算済み Embedding も利用可能。
- GPU 厳密検索は FP32、FP16、BF16、INT8 のベクトル保存に対応。
- Dashboard、Collections、People、Detect、Compare、Search、RTSP Monitor、
  System、Help を備えた多言語 Web UI。
- `/v1` 配下の31個の snake_case REST operation。保護された
  `/v1/embeddings` と軽量 Python SDK を含みます。
- 有界メモリイベント、複数クライアント、任意の `preview.mjpeg` を備えた
  サーバー側 RTSP Monitor。ブラウザを閉じても監視は停止しません。
- SQLite を永続的な正本とし、再構築可能なメモリ内厳密索引、単一の書き込み可能なモデルマウントと設定ディレクトリ、永続 `/data`、migration、health check、CPU fallback を許さない
  CUDA 起動検証を提供。
- JPEG、PNG、WebP、BMP をサポートし、元のアップロード画像は既定で保持しません。

生体検知は既定で無効です。`server/config/server.toml` の `inference.addons = []` と `addons.auto_download = []` が初期値です。**システム → 生体検知** でモデルをダウンロード・検証すると次回起動の設定が保存され、手動再起動で有効になります。検証済みキャッシュは再利用します。起動時にはダウンロードせず、有効な addon のモデルがなければインストール方法を示して起動を停止します。評価済みの顔は基本の3項目 `status`、`is_live`、`live_score` を返し、`input_rejected` の場合だけユーザー向け説明 `reason` が追加されます。既定モードは `normal`、登録時は既定で省略します（`liveness_on_registration = false`）。[設定、Web 権限、更新手順](docs/user-guide.ja.md#任意の生体検知-addon)を参照してください。

### RTX 5090 GPU 検索性能

1基の NVIDIA GeForce RTX 5090（32,607 MiB）で、ネイティブ CUDA
exact-flat index は INT8 使用時に **58.9M 個の512次元画像ベクトル**を
格納できました。

| GPU データ型 | 最大画像ベクトル数 | 10M Top-5 p50 | 10M serial QPS |
| --- | ---: | ---: | ---: |
| FP32 | 15.8M | 12.84 ms | 77.85 |
| FP16 | 30.7M | 6.83 ms | 146.32 |
| BF16 | 30.7M | 6.83 ms | 146.33 |
| INT8 | **58.9M** | **3.84 ms** | **260.81** |

INT8 は FP32 に対し、実測容量 3.73 倍、10M Top-5 throughput 3.35 倍でした。
数値は同一の RTX 5090、Driver 580.105.08、CUDA 12.9 による GPU
測定です。容量は ONNX モデルや Server 負荷を含まない独立ネイティブ索引の
上限です。速度は正確に 10M 個の画像ベクトルを対象に、GPU 常駐 Top-5
全件厳密走査、同時1 query、warm-up 10回と測定100回で計測しました。
索引は各保存表現内では厳密ですが、量子化により FP32 から score が変わる
場合があります。本番ではモデル、request、並行処理、索引再構築、allocator
のための VRAM 余裕が必要です。

### ICCV21-MFR 多人種 MR-ALL 精度

[ICCV21-MFR](../challenges/iccv21-mfr/) の多人種（MR）テストセットで、
MR-ALL の全組合せ 1:1 プロトコルと FAR `1e-6` を用いてネイティブ検索
profile を評価しました。すべての profile は Server API で一度だけ抽出し
L2 normalization を施した同一の512次元 `buffalo_l` embedding を使用し、
ベクトルの保存形式と検索時の計算表現だけを変更しています。

| 検索 profile | FAR 1e-6 での MR-ALL | Cosine threshold | FP32 との差 |
| --- | ---: | ---: | ---: |
| FP32 | 91.249107% | 0.407787 | — |
| FP16 | 91.249197% | 0.407787 | +0.000090ポイント |
| BF16 | 91.248502% | 0.407787 | -0.000605ポイント |
| **INT8** | **91.248005%** | **0.407739** | **-0.001102ポイント** |

**このベンチマークでは INT8 に実質的な精度低下はありません。**
challenge と同じ小数第2位までの表示では FP32 と INT8 の MR-ALL はともに
**91.25%** で、丸め前の差もわずか0.0011ポイントです。同時に、上記の実測
容量3.73倍と 10M Top-5 throughput 3.35倍の利点を維持します。この比較は
ベクトル保存・検索精度の評価であり、INT8 モデル推論の評価ではありません。

![英語版 Collection 管理](docs/images/customer/collections-en.jpg)

![英語版 RTSP Monitor。プライベートアドレスはマスク済み](docs/images/customer/monitoring-en.jpg)

## クイックスタート

必要条件:

- Docker Engine と Docker Compose を備えた Linux x86_64。
- CUDA 版は、対応 NVIDIA GPU、NVIDIA Driver、NVIDIA Container Toolkit。

ホストに Python、OpenCV、ONNX Runtime、CUDA Toolkit、cuDNN は不要です。
公開 Image にモデル、顧客データ、API Key、本番設定は含まれません。

InsightFace リポジトリ全体の checkout で、`server/.models` にモデルを
インストールします。

リポジトリのルートで実行し、`server/config/server.toml` を用意してください。Server とモデルインストーラーは root（`0:0`）で実行します。Compose は `server/.models` がなければ作成し、単一の書き込み可能な `/models` としてマウントします。`addons/` は addon のダウンロードを要求したときに作成されます。UID/GID の export、addon ディレクトリの手動作成、権限設定は不要です。

```bash
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license
```

モデルのインストールと同時に生体検知を設定する場合は、代わりに次のコマンドを使います：

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

Server を先に起動する必要はありません。新規環境では次の `up -d` で生体検知が有効になります。すでに実行中の場合は `docker compose -f server/deploy/compose.cpu.yml restart server` が必要で、`up -d` だけでは保存済み設定を再読込しません。CUDA では `compose.cuda12.yml` を使います。

モデルツールは `buffalo_l`、`buffalo_m`、`buffalo_s`、`buffalo_sc`、
`antelopev2`、`raccoon_s`、`raccoon_l` の全 7 パッケージに対応します。
`manifest.json` と署名済み `MODEL.LICENSE` を作成し、`models verify` で
検証できます。モデル条件は Server ソースライセンスとは別です。

CPU を起動:

```bash
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

代わりに CUDA 12 を起動:

```bash
docker compose -f server/deploy/compose.cuda12.yml pull
docker compose -f server/deploy/compose.cuda12.yml \
  run --rm models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cuda12.yml up -d
curl -fsS http://127.0.0.1:18098/v1/health
```

CPU は `http://SERVER:18097/`、CUDA は `http://SERVER:18098/` を開きます。
Collection を作成し、Person に1枚以上の画像を登録して、別の画像で検索します。
データを保持するには `-v` を付けずに `docker compose ... down` します。

同梱 Compose は隔離評価向けに認証を既定で無効にします。他のユーザーや
ネットワークへ公開する前に次を設定します。

```bash
export INSIGHTFACE_AUTH_ENABLED=true
export INSIGHTFACE_API_KEY='十分に長いランダムな秘密へ置換'
docker compose -f server/deploy/compose.cpu.yml up -d
```

完全な初回手順は[初心者向けユーザーガイド](docs/user-guide.ja.md)を参照してください。

## ソースからビルド

完全なローカルソースディレクトリから直接ビルドできます。未コミットの変更が
あっても、`.git` ディレクトリがなくても構いません。Git のコミットやプッシュは
ビルドの前提条件ではありません。

Dockerfile は `server/` と `python-package/insightface/` の選択された推論
モジュールをコピーするため、完全なソースディレクトリが build context です。

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

`--pull never` によりローカル build の Image を使用します。build 時には固定済み
base image と依存を、モデル install 時には許諾したモデル package を別途
download します。

## コア動作

- Similarity は確率ではなく元の cosine 値です。Threshold は `0.0..1.0`、
  既定は `0.4`。
- Collection はモデルと embedding contract に固定されます。不一致でも表示は
  できますが、登録/検索は `collection_model_mismatch` を返します。
- 起動時 Detection Profile は新しい Collection にコピーされ、その後の
  Collection Profile は独立して変更でき、次の Request から有効になります。
- 任意の顔保存は 112x112 にリサイズした bounding-box JPEG crop であり、
  元画像や認識用のアライン済み入力ではありません。既定は無効です。
- SQLite commit が正本です。登録/削除の成功応答前に索引を同期し、再起動後は
  SQLite から再構築します。
- Response は `x-request-id` を持ち、一覧 API は不透明な署名 cursor を使います。

正確な field、default、lifecycle、failure behavior は以下の詳細文書が正本です。

## API と SDK

主な API group:

- System: `/v1/health`、`/v1/system`、`/v1/models`。
- Stateless face: `/v1/detect`、`/v1/compare`、`/v1/embeddings`。
- Collection、Person、FaceSample CRUD。
- Collection の Person search。
- RTSP Monitor の設定、state、events、preview。

全 parameter、response、error、example は
[REST API 利用ガイド](docs/api.ja.md)を参照してください。Interactive OpenAPI
は `/docs` に残ります。

```python
from insightface_server import Client

with Client("http://localhost:18097", api_key=None) as client:
    faces = client.detect("photo.jpg")
    matches = client.search("employees", "unknown.jpg", limit=5)
```

SDK のインストール、入力形式、メソッド、完全な手順は
[ユーザーガイド](docs/user-guide.ja.md)を参照してください。

## セキュリティ

顔画像と Embedding は生体情報です。ネットワーク公開時は認証を有効にし、信頼する
reverse proxy で HTTPS を終端し、Docker と volume へのアクセスを制限し、広い
CORS を無効のまま保ち、backup、retention、deletion、consent、incident response
を定義してください。画像、Embedding、RTSP credential、API Key を log に
記録しないでください。

Server は TLS、ユーザーアカウント、RBAC、cloud IAM、法令遵守レイヤーを内蔵
しません。運用とセキュリティは[ユーザーガイド](docs/user-guide.ja.md)を
参照してください。

## フェーズ1の範囲

AWS/CompreFace 互換、CUDA 11、Jetson、ARM64、Windows Container、
TensorRT、Kubernetes、分散 Worker、Monitor event の永続化、録画/NVR、
deepfake、属性分析は実装しません。

## ドキュメント

- [ユーザーガイド](docs/user-guide.ja.md) — インストール、設定、モデル、
  Web UI、SDK、GPU、セキュリティ、バックアップ、トラブルシュート。
- [REST API 利用ガイド](docs/api.ja.md) — 全公開 Endpoint、Field、挙動、
  Result、Error、Pagination、Example。
- [Maintainer Guide — English](docs/maintainer-guide.md) — Architecture、
  Search 内部、Test、Contribution、Container Release。

GitHub と Web UI Help は同じローカライズ済み User Guide/API Guide Markdown
を読み込み、表示方法だけが異なります。

## ライセンス

ライセンスの唯一の入口は [LICENSING.md](LICENSING.md) です。Server
ソースと Python SDK は MIT License ですが、この宣言はモデルファイル、
モデル重み、データセット、第三者コンポーネントには適用されません。公開
InsightFace 学習済みモデルは、別途許諾がない限り通常は非商用研究用途に
限定されます。商用ライセンス: <https://www.insightface.ai>。
