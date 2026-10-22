# InsightFace Server 用户指南

**语言：** [English](user-guide.md) · 中文 · [日本語](user-guide.ja.md) · [Deutsch](user-guide.de.md) · [Español](user-guide.es.md) · [Français](user-guide.fr.md) · [Русский](user-guide.ru.md) · [Português](user-guide.pt.md) · [한국어](user-guide.ko.md)

这是面向第一次使用者的分步操作指南：从一个空的项目目录开始，直到创建人员库、
注册人员并得到第一次搜索结果。相同能力可以通过 Web UI、`/v1` API 和 Python SDK
使用；每个HTTP字段和响应的完整说明请查看
[API使用手册](api.zh-CN.md)。

活体检测的使用方法请查看[活体配置、模型安装和返回值说明](#可选活体检测-addon)；下方各操作章节也说明了活体对该流程的影响。

## 从这里开始：从零启动到第一次成功搜索

CPU版需要Linux x86_64、Docker Engine和Docker Compose。CUDA版还需要兼容的
NVIDIA Driver与NVIDIA Container Toolkit；宿主机不需要安装CUDA、cuDNN、
ONNX Runtime、Python或OpenCV。

CPU启动示例：

在仓库根目录执行，并保留 `server/config/server.toml`。Server 和模型安装器统一以 root（`0:0`）运行。Compose 会在缺少时自动创建 `server/.models`，并将其作为单个可写目录挂载到 `/models`；请求下载 addon 时才创建 `addons/`。无需导出 UID/GID、手动创建 addon 目录或设置目录权限。

```bash
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml run --rm models install buffalo_l
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

如需在安装模型时同时配置活体，可改用下面的安装命令：

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

无需先启动 Server。全新部署随后执行 `up -d` 即会启用活体；已经运行的 Server 需要执行 `docker compose -f server/deploy/compose.cpu.yml restart server`，仅执行 `up -d` 不会重新加载已保存的设置。CUDA 部署改用 `compose.cuda12.yml`。

GPU版把`compose.cpu.yml`替换成`compose.cuda12.yml`，健康检查端口改为`18098`。
模型安装器会在下载前展示许可。InsightFace公开预训练模型默认仅允许非商业研究，
商业使用需要单独授权。

随项目提供的Compose配置在隔离评估环境中默认`auth_enabled=false`，此时API不用传
认证字段，Web UI也会隐藏API Key输入。对其他用户或网络开放前，应在首次启动前启用：

```bash
export INSIGHTFACE_AUTH_ENABLED=true
export INSIGHTFACE_API_KEY='replace-with-a-long-random-secret'
docker compose -f server/deploy/compose.cpu.yml up -d
```

CPU访问`http://服务器地址:18097/`，GPU访问`http://服务器地址:18098/`。第一次操作
按以下顺序完成：确认仪表盘全部就绪、创建Collection、用至少一张清晰图片注册Person、
再用该人员的另一张图片执行Search。没有匹配时返回空列表，这是正常成功结果。
停止服务使用`docker compose ... down`且不要加`-v`；`-v`会永久删除命名数据卷。

## 可选活体检测 addon

### 启用与模型安装

`server/config/server.toml` 默认关闭活体：`inference.addons` 和 `addons.auto_download` 均为 `[]`。旧配置缺少这些键时也保持关闭。

**通过命令行启用，也适用于首次启动 Server 之前：**

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

`--enable-liveness` 会先检查已有配置是否可以更新，再安装并校验基础模型包、配置要求附带安装的 addons 和活体模型，最后向 `inference.addons` 与 `addons.auto_download` 都追加 `liveness`，保留其他条目、注释和设置。已校验的缓存会复用，缓存命中时仍会保存启用设置。下载失败不会改变配置；保存配置失败会明确报错并以非零状态退出，已缓存的有效模型可在重试时复用。

两个 Compose 服务均将已有的整个 `server/config` 目录可写挂载到 `/etc/insightface`，并保留 `create_host_path: false`，因此安装器可在 Server 尚未运行时原子更新宿主机配置。该目录及 `server.toml` 必须存在。

无需先启动 Server。全新部署随后执行 `up -d` 即会启用活体；已经运行的 Server 需要执行 `docker compose -f server/deploy/compose.cpu.yml restart server`，仅执行 `up -d` 不会重新加载已保存的设置。CUDA 部署改用 `compose.cuda12.yml`。

不加 `--enable-liveness` 时，`models install` 保持原有行为，不写配置，默认活体仍关闭。`models addons install liveness` 只下载、校验 addon，不会启用活体。也可以按照下文通过 **系统 → 活体检测** 启用。

在 **系统 → 活体检测** 点击 **下载并在重启后启用**。Server 下载发布的模型并校验 SHA-256 后，自动向同一份配置文件的上述两个列表追加 `liveness`，保留其他条目、注释和设置；已校验的缓存直接复用。当前进程保持原状态，必须**手动重启 Server**后才启用。下载或配置保存失败会显示错误并允许重试；下载失败不会启用活体。仅有模型文件不代表已启用。

系统页面区分已校验的安装状态（`installed`）、当前运行状态（`enabled`）、已保存的下次启动配置（`configured_enabled`）和是否需要重启（`restart_required`）。下载或保存不改变当前推理。需要关闭时，在同一文件将 `inference.addons=[]` 和 `addons.auto_download=[]` 保存后手动重启。网页操作不修改注册开关，其默认值仍为 `liveness_on_registration=false`。

**高级用法：手动配置活体。** 以下设置可替代启用参数或网页操作；重启前请先安装模型。

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

`inference.addons` 控制运行时启用；`addons.auto_download` 独立控制模型安装时的附带下载。将后者设为 `["liveness"]` 后，安装任意受支持的基础包时都会补齐 addon，基础包已缓存也一样。**启动 Server 时不下载模型。** 安装工具和 Server 读取同一份配置文件。

```bash
docker compose -f server/deploy/compose.cpu.yml run --rm models install buffalo_l --accept-license
# 或者只安装 addon：
docker compose -f server/deploy/compose.cpu.yml run --rm models addons install liveness
docker compose -f server/deploy/compose.cpu.yml run --rm models addons verify liveness
docker compose -f server/deploy/compose.cpu.yml up -d --force-recreate
```

CUDA 部署改用 `compose.cuda12.yml`。独立 CLI 也支持
`models --config-file PATH --models-dir ROOT addons install liveness`。
模型来自[指定的 Release](https://github.com/deepinsight/insightface-model-addons/releases/download/addons/liveness.onnx)，
下载后校验固定 SHA-256。宿主机路径为 `server/.models/addons/liveness.onnx`，
容器路径为 `/models/addons/liveness.onnx`；所有 addon 平铺在同一 `addons/` 目录。
模型根目录使用单个可写挂载，下载 addon 时按需创建 `addons/` 子目录；配置目录也允许网页管理写入。

Docker 镜像只包含代码和依赖，不包含预训练权重。重建镜像不会给用户原有挂载目录补充
模型。默认关闭活体，旧用户升级无需额外下载。若用户手动启用活体但未安装模型，启动会报 `addon_model_missing`，并显示
完整路径和安装命令；文件损坏或无法读取则报 `addon_model_invalid`。不会静默关闭
活体。执行安装工具补齐模型后重新启动；损坏文件需要替换为经过校验的发布文件。
只升级代码、未启用活体的用户保持原有行为。数据库迁移保留已有样本和历史
embedding；活体功能不改变识别模型摘要或 `embedding_contract_id`。

### 网页下载所需的挂载与权限

默认 Compose 为 Server 和模型安装器提供单个可写的 `/models` 挂载，并设置
`create_host_path: true`。两者均以 root（`0:0`）运行，使用 Docker 默认 capabilities，
不再设置 `cap_drop: [ALL]`。容器根文件系统仍为只读，并保留 `no-new-privileges`。
这样无需配置宿主机 UID/GID 或执行 `chmod 777`。root 可以修改可写挂载中的文件，
新下载的文件在宿主机上可能归 root 所有。

两个服务都将已有的整个 `server/config` 目录可写挂载到 `/etc/insightface`，供网页操作
和 `--enable-liveness` 原子保存 `server.toml`。该目录和文件必须存在，Compose 不会
自动创建配置源。自定义部署使用实际的模型和配置路径，并为两个服务提供同等可写目录挂载；
自定义只读挂载可以读取已有模型，但无法支持网页下载或配置保存。

CUDA 使用 `compose.cuda12.yml`。模型文件已存在不会自动启用活体。网页保存成功后执行
`docker compose -f server/deploy/compose.cpu.yml restart server` 应用配置。
修改挂载、运行用户、capabilities 或代理环境变量时，需要重新创建容器。下载需要代理时，在创建容器前设置
`HTTP_PROXY`、`HTTPS_PROXY` 和 `NO_PROXY`，Compose 会传给 Server 和模型工具。
代理应使用容器可访问的局域网地址；容器内的 `127.0.0.1` 不代表 Mac。
该操作沿用 API Key 认证；关闭认证时，能访问 API 的用户也能准备活体。
本功能只下载固定发布的活体模型，不接受自定义下载地址，也不提供基础模型包切换。

### 读取活体结果

每张已执行活体的人脸增加 `liveness`，其中包含三个核心字段：

| 结果 | `status` | `is_live` | `live_score` |
| --- | --- | --- | --- |
| 活体通过 | `ok` | `true` | `[0, 1]` 分数 |
| fake | `ok` | `false` | `[0, 1]` 分数 |
| 人脸周围的有效图像区域不足 | `input_rejected` | `null` | `null` |

只有对齐后人脸周围的原图有效区域不足时，才返回 `input_rejected`。此时额外提供 `liveness.reason`，用于向用户解释；活体通过和 fake 结果不包含 `reason`。FaceAnalysis 和 API 始终返回英文提示，只有 Web UI 按界面语言翻译显示。程序应根据 `status` 和 `is_live` 判断，不要解析 `reason` 文本。旧的已保存结果可能没有 `reason`，客户端可回退到通用的输入被拒绝提示。

```json
{
  "status": "input_rejected",
  "is_live": null,
  "live_score": null,
  "reason": "Insufficient image area around the face for liveness detection. Move the face toward the center, step back from the camera, or use a less tightly cropped image."
}
```

`live_score >= liveness_threshold` 判为通过。未启用、注册跳过活体或比对范围未选中的一侧不返回
`liveness`；这与 `is_live: null` 表示输入被拒绝有明确区别。

- `normal`（默认）：先检测并选择人脸，再做活体，通过后才提取识别特征。选中的人脸
  不通过时，不会换用背景中另一张通过的人脸。
- `observe`：记录活体结果，fake 或输入不合格也继续识别；推理故障仍返回错误。
- `liveness_compare_scope` 支持 `both`（默认）、`source`、`target`，只决定
  `/v1/compare` 哪一侧做活体。注册使用下述独立开关。请求参数不能覆盖这些启动配置。

### 检测、识别和错误返回

`/v1/detect` 始终不提取 embedding，fake 和输入不合格都以 HTTP 200 返回逐脸结果。
`normal` 下，embedding、比对和 Collection 搜索在活体不通过时返回 HTTP 422：
错误码分别为 `liveness_fake` 和 `liveness_input_rejected`，
`error.details.liveness` 包含活体结果，输入被拒绝时还包含 `reason`；比对还提供 `error.details.side`
（`source` 或 `target`）。被拦截的操作不返回相似度或匹配结果。
推理故障返回 HTTP 503 `liveness_unavailable`，不归类为 fake。 运行故障在 `normal` 和 `observe` 下都会中止操作，不会转换为 `input_rejected`。

### 注册默认跳过活体

注册默认跳过活体。`[inference].liveness_on_registration = false` 时，新建人员和
追加 FaceSample 均不运行活体模型，新样本不包含 `liveness`。人脸检测、识别特征提取或
外部特征校验，以及所选 `review_mode` 的审核仍正常执行。此开关只能通过启动配置设置，
请求参数不能覆盖。

活体 addon 已启用时，设为 `liveness_on_registration = true`，注册才遵循
`normal`/`observe` 策略。`normal` 下活体拒绝项包含对应 `reason` 和 `liveness`，
批量注册允许部分成功；新建人员时全部被拒绝，仍返回 `registration_failed`，
其 `details.rejected_images` 提供逐图结果。`review_mode=off` 和
`embedding_mode=external_trusted` 不能绕过已开启的注册活体检查。
`observe` 下注册继续并保存活体结果。历史已保存的结果仍可查询，未做过活体的样本不包含该字段。

Web UI 的新建人员和追加样本拒绝列表优先显示实际 `reason`，活体结果另起一行显示。
例如，`low_quality` 可以与“活体通过”同时出现；活体通过不代表通过注册质量审核。

### RTSP 和 Web UI

RTSP 在 `normal` 下把未通过的人脸标为外层 `status: liveness_blocked`，不返回身份，
单独计入 `liveness_blocked_faces`，不计入 `unknown_faces`，不触发人员或陌生人进入事件，
并重新累计身份确认帧数。活体推理异常会清除过期的识别展示；`observe` 继续匹配。
Web UI 展示活体结果和明确的拒绝状态；`/v1/models` 与 `/v1/system` 在基础模型之外
单独列出已启用的 addon。

## 1. 登录并检查就绪状态

CPU 打开 `http://服务器地址:18097/`，CUDA 12 打开 `http://服务器地址:18098/`。如果启用了认证，点击 **配置 API Key**，粘贴管理员提供的 Key，再选择 **在此标签页使用**。Key 只保留在当前标签页内存中，刷新或关闭页面后即清除。

注册数据前请查看 **仪表盘** 或 **系统**。服务、数据库、模型和 Provider 均应为就绪。CUDA 部署必须显示 `CUDAExecutionProvider`，不会静默回退到 CPU。

仪表盘的模型名称下始终显示 **活体检测已启用**或**活体检测已禁用**。系统页面分别显示模型是否已安装、当前运行状态和待重启状态。

## 2. 创建 Collection

打开 **人员库**，选择 **新建人员库**，设置：

- 稳定 ID，例如 `employees`；
- 展示名称、描述和可选 metadata；
- 默认 cosine 阈值，初始建议 `0.4`；
- 当前主机支持的 search profile；
- 容量和每个 Person 最多 FaceSample 数；
- 检测输入尺寸、检测/NMS 阈值以及单脸挑选策略；
- 是否保存缩放为 112×112 的 `bounding-box crop` JPEG；它不是识别模型使用的
  对齐输入，默认关闭。

Collection 会固定绑定模型 ID、digest、特征维度和预处理版本。检测配置在创建时复制系统默认值，之后可以单独修改；修改从下一次请求生效并递增 `detection_revision`，但不会重新处理已有 FaceSample。`largest` 优先面积；`center_largest` 最大化 `人脸面积 - 2.0 × 人脸框中心到图像中心的像素距离平方`，检测置信度不参与该分数。

## 3. 注册 Person

打开 **人员**，选择 Collection，再点击 **注册人员**。可填写稳定的 Person ID、姓名、外部 ID 和 JSON metadata，然后拖入一张或多张 JPEG、PNG、WebP 或 BMP 图片。

入库审查模式：

- `off`：使用 Collection 的单脸挑选策略，允许图片中存在多张脸；
- `standard`：要求一张可用脸，并检查尺寸、检测分数、清晰度、亮度和姿态；
- `strict`：在 standard 基础上，要求样本的最佳类内相似度高于最佳类外相似度。

批量注册支持部分成功。请根据每张失败图片的原因处理后重试；系统不保存被拒绝的
原图。启用人脸图保存时，只保存缩放为 112×112 的 `bounding-box crop`，不保存
原始上传图片或识别模型使用的对齐输入。

可信系统可以用 `external_trusted` 提交预先抽取并 L2 归一化的 embedding。仍须同时提供图片完成检测和质量审查，但服务不会再次抽取特征；embedding contract 必须与 Collection 完全一致。

新建人员和追加 FaceSample 默认跳过活体（`liveness_on_registration=false`）。管理员开启后，`normal` 拒绝 fake 和输入不合格的图片，`observe` 保留结果并继续注册；入库质量审查仍遵循所选 `review_mode`。拒绝列表分别显示实际 `reason` 和活体结果，活体通过不代表质量审查通过。

## 4. 检测与比对

在 **检测** 中上传单图，可查看人脸框、五点关键点、检测分数和启发式质量信息。无人脸是成功的空列表。

在 **比对** 中分别上传 source 和 target，并可选择系统或 Collection 检测配置。配置中的策略从两张图各挑选一张可用脸，返回原始 cosine `similarity`、`threshold` 和 `matched`。Similarity 不是概率；任一图片没有可用脸时返回 `422 face_not_found`。

启用活体后，每张执行过活体的人脸会包含 `liveness.status`、`liveness.is_live`、`liveness.live_score`。检测对 fake 和 `input_rejected` 都返回 HTTP 200，且不提取识别特征。`input_rejected` 表示人脸周围的有效图像区域不足，`liveness.reason` 提供调整图片的提示。缺少 `liveness` 表示这张脸未执行活体。

活体在识别前执行，`liveness_compare_scope` 决定检查 `both`、`source` 或 `target`。`normal` 下任一被检查侧未通过时，返回 HTTP 422 `liveness_fake` 或 `liveness_input_rejected`，并提供 `error.details.liveness` 和 `error.details.side`，不返回相似度。`observe` 继续比对，并在执行过检查的人脸上返回活体结果。

## 5. 搜索人员库

打开 **搜索**，选择 Collection，上传查询图片并设置返回数量；也可以临时覆盖阈值。系统按 Collection 检测配置挑选查询脸，按相似度降序返回。Person 得分取其所有 FaceSample 的最高相似度。无匹配是成功的空列表。

新 FaceSample 会先提交到 SQLite，再加入内存索引，然后才返回成功；删除同时更新两处。重启时从 SQLite 重建索引，SQLite 始终是权威数据源。

启用活体且为 `normal` 时，查询图片的 fake 或输入不合格返回 HTTP 422 `liveness_fake` 或 `liveness_input_rejected`，详情为 `error.details.liveness`，不会执行搜索；这与搜索成功但匹配列表为空不同。`observe` 继续搜索，并在查询人脸上返回活体结果。

## 6. RTSP 摄像头监控

打开 **摄像头监控**，点击 **新建监控任务**。填写任务ID和名称，输入`rtsp://`或
`rtsps://`地址，选择Collection，并设置每秒推理次数和可选匹配阈值。事件策略可以
设置连续多少帧后确认、离开超时、重复事件冷却时间，以及内存中保留的最近事件数量。

**Web视频预览默认关闭。** 只有管理员需要查看画面时才开启；不开预览也会持续识别
和生成事件。开启后服务器传输原始JPEG帧，Web UI依据`/state`结果绘制标注：绿色框
表示已入库人员，橙色框表示检测到但未入库的人脸。

Monitor独立运行在服务器端，关闭浏览器不会停止；处于启用状态的任务会在Server重启
后自动恢复。使用 **启动/停止** 修改`enabled`，使用 **编辑** 更换RTSP源或调整参数，
使用 **删除** 永久移除任务。解码器只保留最新帧；推理耗时超过设定周期时直接跳过
过时帧，不会排队补跑。

Monitor配置保存在SQLite中，RTSP凭据加密保存在`/data`且API不会回传。视频帧不会
保存；进入、离开、错误和恢复事件只保留在有上限的内存环形缓冲区，进程重启后丢失。
Web UI/API跨越不可信网络时应使用HTTPS，并只允许可信管理员管理Monitor。

启用活体且为 `normal` 时，未通过的人脸显示外层 `status: liveness_blocked` 和独立活体结果，计入 `liveness_blocked_faces`，不计入 `unknown_faces`，也不触发人员或陌生人进入事件。`observe` 继续识别。界面会区分“输入被拒绝”和“活体未通过”。

## 7. 修改与删除

可在列表中修改 Collection 和 Person。删除 FaceSample 会同时删除 embedding 和可选裁剪图。删除非空 Collection 需要明确确认 `force`。批量或破坏性操作前先备份 `/data`。

## 8. API 与 Python SDK

面向开发者的 OpenAPI Schema 浏览器位于 `/docs`；任务式 API 使用说明就在本帮助中。每个响应都带 `x-request-id`，报告问题时请一并提供。

```python
from insightface_server import Client

client = Client("http://localhost:18097", api_key="your-key")
client.create_collection(collection_id="employees", name="员工库", threshold=0.4)
client.add_person("employees", person_id="alice", images=["alice-1.jpg", "alice-2.jpg"])
matches = client.search("employees", "query.jpg", limit=5)
```

## 9. 数据、备份与安全

- 持久化保存 `/data`、可写模型根目录和配置目录。容器根文件系统继续只读，网页模型管理仍由 API 鉴权控制。
- 停止写入后备份 SQLite 和裁剪图目录，或使用 SQLite 安全快照方式。
- API Key 只以 hash 保存。后续启动同一数据卷时传入不同 `INSIGHTFACE_API_KEY`，会主动轮换当前 Key。
- 不要记录图片、embedding 或 Key；除非确有需要，不要开启宽泛 CORS。
- 公开镜像不包含模型。**系统** 页面会读取当前模型包的`MODEL.LICENSE`并显示
  实际授权；文件缺失时默认显示为非商业。若文件存在但签名无效、模型不匹配、
  尚未生效或已经过期，Server仍会拒绝启动。商业使用需要单独许可，请访问
  <https://www.insightface.ai>。

## 10. 故障定位

`401 unauthorized` 表示当前标签页未配置 Key 或 Key 已轮换。`409 collection_model_mismatch` 表示 Collection 与当前模型契约不同。`422 face_not_found` 表示没有选出可用脸。CUDA 模式在 Driver、GPU、模型 Session、Provider 或 warm-up 检查失败时会主动终止。请查看 **系统**、容器日志和响应中的 `request_id`。

## 11. 模型与模型许可

镜像不包含模型。一次性的`models`工具把模型安装到`server/.models`，正常Server
启动无需联网：

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models verify buffalo_l
```

公开包包括：`buffalo_l`（`det_10g.onnx` + `w600k_r50.onnx`）、
`buffalo_m`（`det_2.5g.onnx` + `w600k_r50.onnx`）、
`buffalo_s`和`buffalo_sc`（`det_500m.onnx` + `w600k_mbf.onnx`）、
`antelopev2`（`scrfd_10g_bnkps.onnx` + `glintr100.onnx`）、`raccoon_s`
（`det_10g_wo.onnx` + `w600k_mbf.onnx`）以及`raccoon_l`
（`det_10g_wo.onnx` + `w600k_r50.onnx`）。Server只安装各包中的检测和识别
模型，不安装或加载Raccoon中供PrivateFrame使用的verifier。安装会生成`manifest.json`
与签名的`MODEL.LICENSE`。不带`--accept-license`时，交互终端会在下载前询问确认；
非交互命令必须带该参数，否则会退出且不下载。`models verify`会核验包身份、签名、有效期和当前授权状态；与运行时的缺失
文件显示回退不同，这个显式核验命令仍要求存在有效的签名许可文件。

InsightFace公开预训练模型默认仅限非商业研究；商业使用需另行获取授权。私有模型也
可以使用同样的manifest和离线签名许可。许可按`model_id`表达授权，是合规凭证，
不是DRM，也不要求模型文件SHA-256保持不变。

## 12. 仅启动时生效的配置

通用配置文件为`server/config/server.toml`，Compose将其所在目录可写挂载到
`/etc/insightface`，容器内文件为 `/etc/insightface/server.toml`。修改后必须重启容器，默认值如下：

```toml
[inference]
max_concurrency = "auto" # CPU为4，CUDA为8
addons = []
liveness_mode = "normal"
liveness_threshold = 0.8
liveness_compare_scope = "both"
liveness_on_registration = false

[addons]
auto_download = []

[detection]
input_sizes = [[96, 96], [512, 512]]
threshold = 0.50
nms_threshold = 0.40
single_face_selection = "largest"
max_detected_faces = 100

[web]
disabled = false
```

动态SCRFD会分别运行所有分辨率，把候选框映射回原图后合并，并只执行一次全局NMS。
推理设置只在启动时读取。网页活体操作可以保存下次启动的配置，不会热更新当前进程。
新Collection会复制系统检测配置，
之后可独立修改并从下一次请求生效。无状态Detect和Embeddings使用系统配置；Compare
可使用系统配置或指定Collection；注册与Search始终使用Collection配置。

将`[web].disabled=true`可只启动API。此时`/v1`和`/openapi.json`仍可用，但不会
注册`/`、`/docs`、帮助文档和前端静态资源。

## 13. 精确检索Profile与容量

**系统**接口只公布当前CPU/GPU真正可用的Profile。Collection在创建时固定Profile，
不能在单次Search请求中临时切换。

| Profile | 存储类型 | 常见可用环境 |
| --- | --- | --- |
| `fp32_v1` | FP32 | CPU与CUDA |
| `fp16_v1` | FP16 | CUDA |
| `bf16_v1` | BF16 | 支持的CPU或SM80+ CUDA |
| `int8_x736_v1` | INT8，scale 736 | CPU与CUDA；推荐INT8 |
| `int8_x1000_v1` | INT8，scale 1000 | 兼容已有Collection |

这些实现都会遍历全部有效FaceSample，属于Flat精确全量搜索，不是ANN索引。低精度
Profile会近似FP32分数；INT8点积使用INT32累加。对外相似度和阈值始终是原始cosine。

`capacity_rows`预留该Collection的最大有效行数，避免常规扩容停顿。512维向量的
大致纯特征占用为：FP32每行2,048字节，FP16/BF16每行1,024字节，INT8每行512字节，
还需额外计算ID与工作区。默认容量`100000`，部署级上限默认`10000000`。
`max_faces_per_person`默认`20`，限制单人样本数，不限制Person数量。

## 14. CUDA支持与严格启动检查

CUDA镜像包含CUDA Runtime 12.9.1、cuDNN 9.24.0、Python 3.11和
`onnxruntime-gpu==1.27.0`。宿主机只需Driver、Docker Engine、NVIDIA Container
Toolkit和兼容GPU。

- Turing、Ampere、Ada、Hopper：Driver R535或更高；
- Blackwell与RTX 50系列：Driver 570.26或更高；
- 新部署建议使用稳定的R580或更高版本。

架构兼容不等于所有GPU型号都已经正式认证。每次CUDA启动都会核验GPU型号、
Compute Capability、Driver、实际CUDA/cuDNN/ORT版本、`CUDAExecutionProvider`、
真实检测与识别Session以及真实warm-up推理，并审计Provider分配。任何关键检查失败
都会终止启动，不会静默回退CPU。使用前请在 **系统** 页面确认结果。

## 15. 构建、升级、备份与恢复

可以直接使用完整的本地源码目录构建，包括尚未提交的修改，也允许目录中没有
`.git`。Git 提交或推送都不是构建的前提。

```bash
make -C server build-cpu
make -C server build-cuda12
```

测试通过后，发布经过测试的同一个镜像。之后仅将相同源码提交或整理 Git 提交，
无需重新构建。如果修改了会进入镜像的文件，例如代码、前端资源或内置用户帮助
文档，则需要重新构建并验证。

随后在Compose的模型安装与`up`命令中加入`--pull never`，即可使用本地镜像。构建
使用固定基础镜像和锁定依赖，但仍需联网获取这些输入。公开版本Tag为
`0.3.1-cpu`和`0.3.1-cuda12`；移动Tag `cpu`/`cuda12`分别指向最新稳定版本，
明确不发布含义模糊的`latest`。

升级前停止写入，使用SQLite安全方式备份`/data`以及可选裁剪图，并保留`/models`
和许可文件。先用数据副本启动新镜像，检查migration、`/v1/health`、模型契约和一条
已知Search，再切换正式数据。停止使用`docker compose down`且不要带`-v`；
`docker compose down -v`会删除命名数据卷。

### 升级到 0.3.1

0.3.1 简化 Docker 部署：两个服务统一以 root 和 Docker 默认 capabilities 运行，
模型共用一个可写挂载。缺少模型根目录时由 Compose 自动创建，显式下载 addon 时创建
`addons/`，无需准备宿主机 UID/GID 或共享组。

自 0.3.0 起，Server 已支持 `raccoon_s`、`raccoon_l` 及其模型描述文件、可选活体、
网页 addon 安装和 BMP 输入。Server 使用 Raccoon 的检测与识别模型，不加载 verifier。
0.3.1 不改变这些功能和 API 响应契约。

**1.** 将 Server 代码和 Compose 文件更新到 0.3.1 对应版本，同时保留自己的
`server/config/server.toml` 设置及部署覆盖配置。保持原有模型路径、`/data`
数据卷名称、裁剪图存储、端口和 API Key 设置。自定义 Compose 文件需要将
`server` 和 `models` 两个服务的镜像都更新为对应的 `0.3.1-cpu` 或
`0.3.1-cuda12`。以下命令应使用你原部署的 Compose 文件、覆盖配置和项目名称。

只更新镜像标签不够。自定义 Compose 和覆盖文件也需将两个服务设置为
`user: "0:0"`，移除 `cap_drop: [ALL]` 及旧 UID/GID、共享组设置，将 `/models`
合并为单个可写挂载并设置 `create_host_path: true`，删除独立的 `/models/addons`
挂载。保留容器根文件系统只读和 `no-new-privileges`。保留已有配置目录和文件：
两个服务都需要整个配置目录可写以原子保存配置；将安装器原来的只读单文件挂载替换为
目录挂载。已有基础模型和
addon 缓存继续保留；标准部署无需递归重设权限或提前准备 addon 目录。

**2.** 拉取新镜像并重新创建 Server 容器。在仓库根目录选择已有部署对应的命令：

CPU：

```bash
docker compose -f server/deploy/compose.cpu.yml pull server models
docker compose -f server/deploy/compose.cpu.yml up -d --no-build --force-recreate server
curl -fsS http://127.0.0.1:18097/v1/health
```

CUDA：

```bash
docker compose -f server/deploy/compose.cuda12.yml pull server models
docker compose -f server/deploy/compose.cuda12.yml up -d --no-build --force-recreate server
curl -fsS http://127.0.0.1:18098/v1/health
```

自行构建时，先构建 0.3.1 镜像，使用
`up -d --no-build --pull never --force-recreate server`，无需拉取镜像。
仅执行 `docker compose restart` 不会切换到新镜像，也不会应用挂载变更。

**3.** 启动时自动执行数据库迁移。等待 `/v1/health` 返回 `ready` 和版本 `0.3.1`，
再在 **系统** 页面确认模型和执行提供程序符合预期。检查原有人员库、人员是否
保留，并执行一次已知图片的搜索。保持同一模型和特征契约时，已有样本、
embedding 和人员库契约 ID 均会保留，无需重新注册。

**升级后按需启用活体。** 默认配置以及未包含 addon 配置项的旧配置都保持活体
关闭，因此普通升级无需下载活体模型，Server 启动时也不会下载。需要启用时，
按照[活体设置说明](#可选活体检测-addon)准备[网页下载所需的挂载与权限](#网页下载所需的挂载与权限)，
在 **系统 → 活体检测** 点击 **下载并在重启后启用**，等待模型安装及配置保存
成功后手动重启 Server。默认模式为 `normal`、阈值为 `0.8`，
`liveness_on_registration=false`；模型位于 `<models_dir>/addons/liveness.onnx`。

**使用 Raccoon 是独立的模型切换。** 升级 Server 会保留当前模型包。需要使用
`raccoon_s` 或 `raccoon_l` 时，按照[模型安装说明](#11-模型与模型许可)，
在独立的模型目录中安装选定的包，再配置相应部署使用该目录。人员库必须匹配
新模型的特征契约，需要新建匹配的人员库并重新注册，或单独进行数据迁移。
Web UI 不提供基础模型包切换。

**自 0.3.0 起的 API 与 SDK 兼容性：**模型、Collection 和 FaceSample 结果不再包含
`model_version`；模型身份使用 `model_id`，人员库兼容性使用
`embedding_contract_id`。自有客户端应取消对旧字段的依赖，使用随项目提供的
Python SDK 时同步升级到 `0.3.1`。执行活体时，`liveness` 包含 `status`、
`is_live`、`live_score` 三个核心字段，仅 `input_rejected` 额外包含 `reason`；未执行时省略该结果。对识别请求启用活体前，
请了解[活体响应与错误规则](#检测识别和错误返回)。

跨网络使用时，应在可信反向代理终止HTTPS，只开放必要的CORS origin，并在边缘限制
速率、请求体和超时。数据卷及备份应按生物识别数据保护。第一阶段只有一个不区分权限
的API Key，不应把它当作多租户授权系统。


模型包直接以名字（例如 `buffalo_l`）标识，不再提供独立的 `model_version`。在识别模型和特征契约保持一致的情况下，Server 升级会保留已有人员库的特征契约 ID、样本和 embedding，无需重新注册。切换识别模型属于单独迁移，契约不匹配的人员库在注册和搜索时返回 `collection_model_mismatch`；新建人员库使用不含模型版本号的特征契约。
