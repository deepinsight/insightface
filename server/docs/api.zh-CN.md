# InsightFace Server REST API使用手册

**语言：** [English](api.md) · 中文 · [日本語](api.ja.md) · [Deutsch](api.de.md) · [Español](api.es.md) · [Français](api.fr.md) · [Русский](api.ru.md) · [Português](api.pt.md) · [한국어](api.ko.md)

本文覆盖每个公开接口的调用方式、参数含义、服务端执行过程、成功结果和常见错误。
如果容器和模型还没有启动，请先阅读
[分步用户指南](user-guide.zh-CN.md)。实时OpenAPI Schema位于`/docs`和
`/openapi.json`；本文负责解释“怎么用”和“结果代表什么”。

活体检测的使用方法请查看[活体配置、模型安装和返回值说明](#可选活体检测-addon)；下方各操作章节也说明了活体对该流程的影响。

## 通用约定

- API基础路径为`/v1`，JSON字段统一使用`snake_case`。
- Collection/PATCH请求使用`application/json`；图片和注册使用
  `multipart/form-data`；裁剪图返回`image/jpeg`；摄像头预览返回MJPEG流。
- JPEG、PNG、WebP和BMP均支持。默认压缩图片上限10 MiB、解码像素上限4000万、整个请求
  上限64 MiB；实际值以`GET /v1/system`为准。
- 每个响应都有`x-request-id`响应头；JSON响应还包含同一个`request_id`。排错时记录
  这个ID，但不要记录图片、embedding、API Key或RTSP凭据。
- `detection_score`和质量分数范围为`0.0..1.0`。`similarity`是原始cosine值，范围
  `[-1.0,1.0]`，不是概率。公开匹配阈值范围为`[0.0,1.0]`，判定为
  `similarity >= threshold`，默认阈值`0.4`。
- 列表接口的`cursor`是不透明令牌，只能原样交回同一个接口、同一Collection、同一
  Person和同一筛选条件，不要解析或自行构造。

项目附带的Compose配置在隔离评估环境中默认关闭认证。`GET /v1/health`始终公开；
管理员启用认证后，其他接口必须发送：

```http
Authorization: Bearer <api_key>
```

认证关闭时不要发送空的`Authorization`头，直接省略该字段。

统一错误格式：

```json
{
  "error": {
    "code": "face_not_found",
    "message": "No usable face was detected.",
    "details": {}
  },
  "request_id": "3ed21e89-4595-4eed-a699-1df42ca62032"
}
```

常用状态码：`400`参数错误、`401`未认证、`404`资源不存在、`409`状态或契约冲突、
`413`请求/图片过大、`422`图片或人脸不符合处理要求、`429`达到流数量限制、`500`
内部错误、`503`超时或模型/索引不可用。

## 第一次API调用

```bash
BASE_URL=http://127.0.0.1:18097
AUTH_HEADER="Authorization: Bearer ${INSIGHTFACE_API_KEY}"

curl -fsS "${BASE_URL}/v1/health"

curl -sS "${BASE_URL}/v1/collections" -H "${AUTH_HEADER}" \
  -H 'Content-Type: application/json' \
  -d '{"id":"employees","name":"员工库","threshold":0.4}'

curl -sS "${BASE_URL}/v1/collections/employees/persons" -H "${AUTH_HEADER}" \
  -F 'id=alice' -F 'name=Alice' -F 'review_mode=off' \
  -F 'images=@alice-enroll.jpg'

curl -sS "${BASE_URL}/v1/collections/employees/search" -H "${AUTH_HEADER}" \
  -F 'image=@alice-query.jpg' -F 'limit=5'
```

认证关闭时从后三条命令中删除`-H "${AUTH_HEADER}"`。

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

使用[用户指南初始设置](user-guide.zh-CN.md)中的当前 Compose 文件和已有配置。Server 与安装器以 root 运行，共用单个可写模型挂载；Compose 创建模型根目录，显式下载 addon 时创建 `addons/`，无需设置宿主机 UID/GID 或手动调整权限。

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
只升级代码、未启用活体的用户保持原有行为。数据库迁移为增量添加字段，不重算历史
embedding，也不改变识别模型摘要或 `embedding_contract_id`。

### 读取活体结果

每张已执行活体的人脸增加 `liveness`，其中包含三个核心字段：

| 结果 | `status` | `is_live` | `live_score` |
| --- | --- | --- | --- |
| 活体通过 | `ok` | `true` | `[0, 1]` 分数 |
| 非活体 | `ok` | `false` | `[0, 1]` 分数 |
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

### RTSP 和 Web UI

RTSP 在 `normal` 下把未通过的人脸标为外层 `status: liveness_blocked`，不返回身份，
单独计入 `liveness_blocked_faces`，不计入 `unknown_faces`，不触发人员或陌生人进入事件，
并重新累计身份确认帧数。活体推理异常会清除过期的识别展示；`observe` 继续匹配。
Web UI 展示活体结果和明确的拒绝状态；`/v1/models` 与 `/v1/system` 在基础模型之外
单独列出已启用的 addon。

## 系统接口

### `GET /v1/health`

**用途：** 容器健康检查和就绪探测，公开且无需认证。

**参数：** 无。

**执行与结果：** 检查启动状态和SQLite `quick_check`。就绪时HTTP 200：

```json
{"status":"ready","auth_enabled":false,"request_id":"..."}
```

```bash
curl -sS "${BASE_URL}/v1/health"
```

**常见错误：** 模型、数据库或索引尚未就绪时返回`503 not_ready`。

### `GET /v1/system`

**用途：** 管理员查看安全的运行诊断。

**参数：** 无。

**执行与结果：** HTTP 200返回Server/OS/CPU/GPU、Compute Capability、Driver、
CUDA、cuDNN、ORT、实际Provider、模型与License、数据库、挂载目录、Collection/
Person/Face数量、检索后端、安全配置、推理并发和最近错误。不会返回密钥、图片或特征。

```bash
curl -sS "${BASE_URL}/v1/system" -H "${AUTH_HEADER}"
```

**常见错误：** `401 unauthorized`、`503 request_timeout`。

### `GET /v1/models`

`addons` 列表单独返回当前启用的 addon，可检查其中是否有 `liveness`；系统响应的 `safe_config` 还返回实际生效的活体设置。这些接口只读，不负责安装模型。

**用途：** 查看当前已验证的检测/识别模型、实际Provider和模型授权摘要。

**参数：** 无。

**执行与结果：** HTTP 200返回`models`、`execution_provider`和`license`，不返回
ONNX文件内容或签名私钥。

```bash
curl -sS "${BASE_URL}/v1/models" -H "${AUTH_HEADER}"
```

**常见错误：** `401 unauthorized`。

支持 `raccoon_s`、`raccoon_l` 基础模型包及 CPU、CUDA 运行，应在启动前用模型工具安装。此接口列出正在使用的模型组件，不是可下载模型目录。下面的网页操作只管理活体。Collection 与识别模型及预处理契约绑定，换基础包不会转换已有特征，可能返回 `409 collection_model_mismatch`；仅启用活体不会改变这一契约。

### `GET /v1/addons/liveness`

**用途:** 只读检查模型安装状态和下次启动配置，不下载、不修改设置。这是管理接口，不是单独的活体推理接口。

**返回:** HTTP 200. `enabled` 表示当前进程是否启用活体；`installed` 表示本地文件通过发布模型的 SHA-256 校验，不代表活体已启用。`configured_enabled` 读取当前配置文件，表示下次启动的选择；`restart_required` 表示它与 `enabled` 不同。重启前，`/v1/system` 的 `safe_config` 仍反映当前进程的设置。

`state` 为 `idle`（没有已校验模型）、`downloading`（正在准备）、`ready`（已有校验通过的模型）或 `error`（准备、文件或配置出错）。仅有 `ready` 不能说明已保存启用设置或已完成重启。

`can_enable` 表示网页准备操作是否可用。不可用时，`unavailable_code` 提供稳定的原因代码，`unavailable_reason` 提供说明；可用时两者均为 `null`。`error` 为 `null` 或含 `code`、`message` 的对象。`model_path` 为本地模型路径，`config_file` 为选中的 TOML 路径或 `null`。响应还包含 `request_id`。

`unavailable_code` 的值包括 `config_file_missing`（未指定配置文件）、`config_file_not_regular`（不是普通文件）、`config_file_mount`（单独挂载配置文件）、`config_not_writable`（配置或所在目录不可写）、`addon_directory_not_writable`（addon 目录不可写）、`addon_config_invalid`（配置无效）、`addon_model_invalid`（模型无效）和 `server_stopping`（正在关闭）。

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

**用途:** 下载活体模型并保存下次启动配置。发送 `Content-Type: application/json` 和空对象 `{}`；不接受模型 URL 或其他参数。

```bash
curl -sS "${BASE_URL}/v1/addons/liveness" -H "${AUTH_HEADER}"
curl -sS "${BASE_URL}/v1/addons/liveness/enable" -H "${AUTH_HEADER}" \
  -H 'Content-Type: application/json' -d '{}'
```

**返回:** HTTP 202 返回与 GET 相同的状态字段，表示任务已接受，不代表活体已启用。轮询 `GET /v1/addons/liveness`，直到准备结束。重复请求共用正在执行的任务，关闭浏览器不会取消下载。

下载并通过 SHA-256 校验后，任务才向 `config_file` 的 `[inference].addons` 和 `[addons].auto_download` 添加 `liveness`，保留其他值和注释；已有合格缓存直接复用。当 `installed=true`、`configured_enabled=true`、`restart_required=true` 时，手动重启 Server。新进程应返回 `enabled=true`、`restart_required=false`。不会热加载，也不提供基础模型包切换接口。

**错误:** 请求错误使用统一错误对象：请求体不是 `{}` 时为 `400 invalid_addon_request`；需要认证但未通过时为 `401 unauthorized`；浏览器来源不允许时为 `403 origin_not_allowed`；路径、权限或配置不支持操作时为 `409 addon_management_unavailable`；非 JSON 类型为 `415 json_required`。浏览器来源必须与 Server 同源，或位于明确配置的 CORS 允许列表中。

任务接受后仍可能失败：GET 继续返回 HTTP 200，通过 `state=error` 和 `error.code` 表示失败。`addon_download_failed` 不修改配置，应检查 Server 的网络或代理后重试；`addon_config_save_failed` 需要修复配置或目录权限，下载成功的模型可复用；`addon_config_invalid` 表示磁盘 TOML 配置无效；`addon_model_invalid` 需要替换或删除损坏的缓存，系统不会静默覆盖；`addon_job_in_progress` 表示另一进程正在准备，应等待并刷新。修复原因后再重新 POST。

## 无状态人脸接口

### `POST /v1/detect`

启用活体后，每张执行过活体的人脸会包含 `liveness.status`、`liveness.is_live`、`liveness.live_score`。检测对 fake 和 `input_rejected` 都返回 HTTP 200，且不提取识别特征。`input_rejected` 表示人脸周围的有效图像区域不足，`liveness.reason` 提供调整图片的提示。缺少 `liveness` 表示这张脸未执行活体。

**用途：** 检测一张图片中的所有可用人脸，不写数据库。

**表单参数：** `image`必填；`max_faces`可选，1～100；`collection_id`可选，指定后
使用该Collection的检测配置，否则使用系统配置。旧参数`min_score`不再支持。

```bash
curl -sS "${BASE_URL}/v1/detect" -H "${AUTH_HEADER}" \
  -F 'image=@group.webp' -F 'max_faces=10' -F 'collection_id=employees'
```

**执行与结果：** 对配置中的每个输入尺寸检测，合并候选后做一次全局NMS，按人脸面积
降序返回。HTTP 200包含`faces`、`processing_ms`和`request_id`；每张脸包含像素/
归一化框、五点关键点、检测分数和质量信息。无人脸是成功的`faces: []`。

**常见错误：** `400 request_detection_override_not_supported`、`404` Collection、
`413`、`422 invalid_image`、`503 request_timeout`。

### `POST /v1/compare`

活体在识别前执行，`liveness_compare_scope` 决定检查 `both`、`source` 或 `target`。`normal` 下任一被检查侧未通过时，返回 HTTP 422 `liveness_fake` 或 `liveness_input_rejected`，并提供 `error.details.liveness` 和 `error.details.side`，不返回相似度。`observe` 继续比对，并在执行过检查的人脸上返回活体结果。

**用途：** 比对两张图片中按策略选中的单张脸，不持久化。

**表单参数：** `source`和`target`必填；`threshold`可选0～1，默认0.4；
`collection_id`可选，用于选择Collection检测配置。

```bash
curl -sS "${BASE_URL}/v1/compare" -H "${AUTH_HEADER}" \
  -F 'source=@source.jpg' -F 'target=@target.png' -F 'threshold=0.4'
```

**执行与结果：** 按`largest`或`center_largest`分别选脸、对齐、抽取并L2归一化特征，
计算原始cosine。HTTP 200返回`matched`、`similarity`、实际`threshold`、两张选中脸、
`processing_ms`和`request_id`。

**常见错误：** `404` Collection、`413`、`422 invalid_image`或`face_not_found`、
`503 request_timeout`。

### `POST /v1/embeddings`

启用活体且为 `normal` 时，fake 或输入不合格返回 HTTP 422 `liveness_fake` 或 `liveness_input_rejected`，详情为 `error.details.liveness`，不提取 embedding。`observe` 继续提取，并随人脸返回活体结果。

**用途：** 为可信集成方抽取一张选中脸的特征；普通注册/搜索不需要调用它。

**表单参数：** `image`必填；`collection_id`可选。旧`face_selection`请求参数不再支持。

```bash
curl -sS "${BASE_URL}/v1/embeddings" -H "${AUTH_HEADER}" \
  -F 'image=@portrait.jpg' -F 'collection_id=employees'
```

**执行与结果：** HTTP 200返回一个`faces`项、L2归一化embedding、`model`、
`processing_ms`和`request_id`。embedding属于敏感生物特征，服务不会记录其内容。

**常见错误：** `400 request_detection_override_not_supported`、`404`、`413`、
`422 invalid_image`或`face_not_found`、`503`。

## Collection接口

### `POST /v1/collections`

**用途：** 创建独立人员库，并固定模型、检测和搜索契约。

**JSON参数：** `id`、`name`必填；`description`默认空字符串；`threshold`默认0.4；
`metadata`默认`{}`；`save_face_crops`默认false。可选`detection`包含`input_sizes`、
`threshold`、`nms_threshold`、`single_face_selection`；可选`search`包含`profile`、
`capacity_rows`、`max_faces_per_person`和`load_policy`。ID为`_default`或1～64位，
首位是字母/数字，其余允许字母、数字、点、下划线和短横线。

```bash
curl -sS "${BASE_URL}/v1/collections" -H "${AUTH_HEADER}" \
  -H 'Content-Type: application/json' \
  -d '{
    "id":"employees",
    "name":"公司员工",
    "threshold":0.4,
    "search":{"profile":"fp32_v1","capacity_rows":100000,"max_faces_per_person":20,"load_policy":"lazy"},
    "detection":{"input_sizes":[[96,96],[512,512]],"threshold":0.5,"nms_threshold":0.4,"single_face_selection":"largest"}
  }'
```

**执行与结果：** 分配索引并固定当前模型ID、digest、512维特征与预处理版本。
HTTP 201返回完整`collection`、解析后的默认值、计数和时间戳。

**常见错误：** `400 invalid_detection_profile`、`unsupported_search_profile`或
`search_capacity_too_large`；`409 collection_exists`；`503 search_index_unavailable`。

### `GET /v1/collections`

**用途：** 分页列出人员库。

**查询参数：** `limit` 1～100，默认50；`cursor`可选不透明令牌。

```bash
curl -sS "${BASE_URL}/v1/collections?limit=50" -H "${AUTH_HEADER}"
```

**结果：** HTTP 200返回`collections`和可空`next_cursor`。**常见错误：**
`400 invalid_cursor`、`401 unauthorized`。

### `GET /v1/collections/{collection_id}`

**用途：** 获取一个人员库。**路径参数：** `collection_id`。

```bash
curl -sS "${BASE_URL}/v1/collections/employees" -H "${AUTH_HEADER}"
```

**结果：** HTTP 200返回`collection`、实时`person_count`、`face_count`和
`embedding_contract_id`。**常见错误：** `404 resource_not_found`。

### `PATCH /v1/collections/{collection_id}`

**用途：** 修改Collection可变策略。**路径参数：** `collection_id`。

**JSON参数：** 可提交`name`、`description`、`threshold`、`metadata`、
`save_face_crops`；`search`只能修改`capacity_rows`、`max_faces_per_person`、
`load_policy`；`detection`可修改检测配置。模型绑定和`search.profile`不可修改，未知字段
及显式null会被拒绝。检测修改从下一次请求生效，不重算已有特征。

```bash
curl -sS -X PATCH "${BASE_URL}/v1/collections/employees" \
  -H "${AUTH_HEADER}" -H 'Content-Type: application/json' \
  -d '{"threshold":0.45,"detection":{"single_face_selection":"center_largest"}}'
```

**结果：** HTTP 200返回完整更新后的`collection`。**常见错误：** `400`、`404`、
`409`容量/模型契约冲突、`503`索引更新失败。

### `DELETE /v1/collections/{collection_id}`

**用途：** 删除人员库。**路径参数：** `collection_id`；**查询参数：** `force`
布尔值，默认false。非空Collection必须明确`force=true`。

```bash
curl -sS -X DELETE "${BASE_URL}/v1/collections/employees?force=true" \
  -H "${AUTH_HEADER}"
```

**结果：** HTTP 204，无响应体。**常见错误：** `404`、`409 collection_not_empty`、
`503 search_index_unavailable`。

## Person与FaceSample接口

### `POST /v1/collections/{collection_id}/persons`

新建人员和追加 FaceSample 默认跳过活体（`liveness_on_registration=false`）。管理员开启后，`normal` 拒绝 fake 和输入不合格的图片，`observe` 保留结果并继续注册；原有入库审查仍然执行。拒绝列表分别显示实际 `reason` 和活体结果，活体通过不代表质量审查通过。

**用途：** 一次创建Person并注册一张或多张FaceSample。

**路径参数：** `collection_id`。**表单参数：** `images`必填且可重复，默认最多20张；
`id`可选，省略后生成UUID；`name`、`external_id`可选；`metadata`是JSON对象字符串，
默认`{}`；`review_mode`为`off|standard|strict`，默认`off`；`embedding_mode`为
`server|external_trusted`，默认`server`。外部模式还必须提交与图片一一对应的
`external_embeddings` JSON数组以及Collection返回的`embedding_contract_id`。

```bash
curl -sS "${BASE_URL}/v1/collections/employees/persons" -H "${AUTH_HEADER}" \
  -F 'id=employee-001' -F 'name=Alice' -F 'external_id=HR-1001' \
  -F 'metadata={"department":"sales"}' -F 'review_mode=standard' \
  -F 'images=@alice1.jpg' -F 'images=@alice2.webp'
```

**执行与结果：** `off`按Collection策略选脸并允许多人脸；`standard`要求恰好一张脸并
执行尺寸、检测分数、清晰度、亮度和姿态审查；`strict`还要求最佳类内相似度严格大于
最佳类外相似度。HTTP 201返回`person`、成功`faces`和逐图片`rejected_images`，允许
部分成功。所有图片失败时返回`422 registration_failed`且不创建Person。

**常见错误：** `400` ID/metadata/图片数量；`404` Collection；`409` Person、外部ID、
embedding契约、容量或每人样本上限冲突；`413`；`422 registration_failed`；`503
search_index_unavailable`。若503详情含`write_committed:true`，先查询Person再决定
是否重试。

### `GET /v1/collections/{collection_id}/persons`

**用途：** 分页列出或筛选人员。**路径参数：** `collection_id`。**查询参数：**
`limit` 1～100，默认50；`cursor`；`search`可选，匹配Person ID、姓名或外部ID。

```bash
curl -sS "${BASE_URL}/v1/collections/employees/persons?limit=50&search=alice" \
  -H "${AUTH_HEADER}"
```

**结果：** HTTP 200返回`persons`和`next_cursor`。**常见错误：**
`400 invalid_cursor`、`404` Collection。

### `GET /v1/collections/{collection_id}/persons/{person_id}`

**用途：** 获取一个Person。**路径参数：** `collection_id`、`person_id`。

```bash
curl -sS "${BASE_URL}/v1/collections/employees/persons/employee-001" \
  -H "${AUTH_HEADER}"
```

**结果：** HTTP 200返回`person`、当前`face_count`和时间戳。**常见错误：** `404`。

### `PATCH /v1/collections/{collection_id}/persons/{person_id}`

**用途：** 修改Person展示信息。**路径参数：** `collection_id`、`person_id`。
**JSON参数：** `name`、`external_id`、对象`metadata`；未知字段拒绝，metadata不可null。

```bash
curl -sS -X PATCH "${BASE_URL}/v1/collections/employees/persons/employee-001" \
  -H "${AUTH_HEADER}" -H 'Content-Type: application/json' \
  -d '{"name":"Alice Chen","metadata":{"department":"sales"}}'
```

**结果：** HTTP 200返回完整`person`。**常见错误：** `400`、`404`、
`409 external_id_exists`。

### `DELETE /v1/collections/{collection_id}/persons/{person_id}`

**用途：** 删除Person及其全部FaceSample、embedding和可选裁剪图。

```bash
curl -sS -X DELETE "${BASE_URL}/v1/collections/employees/persons/employee-001" \
  -H "${AUTH_HEADER}"
```

**结果：** HTTP 204，无响应体；成功后搜索不会再返回该Person。**常见错误：**
`404`、`503 search_index_unavailable`。

### `POST /v1/collections/{collection_id}/persons/{person_id}/faces`

新建人员和追加 FaceSample 默认跳过活体（`liveness_on_registration=false`）。管理员开启后，`normal` 拒绝 fake 和输入不合格的图片，`observe` 保留结果并继续注册；原有入库审查仍然执行。拒绝列表分别显示实际 `reason` 和活体结果，活体通过不代表质量审查通过。

**用途：** 给已有Person增量加入FaceSample。

**路径参数：** `collection_id`、`person_id`。**表单参数：** 可重复`images`、
`review_mode`、`embedding_mode`、`external_embeddings`、`embedding_contract_id`，
含义与创建Person完全相同。

```bash
curl -sS "${BASE_URL}/v1/collections/employees/persons/employee-001/faces" \
  -H "${AUTH_HEADER}" -F 'review_mode=standard' \
  -F 'images=@alice3.jpg' -F 'images=@alice4.png'
```

**结果：** HTTP 201返回成功`faces`和逐图片`rejected_images`，允许部分成功。
**常见错误：** 与注册Person相同，另有`404` Person。

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces`

**用途：** 分页列出FaceSample元数据，不返回embedding或图片字节。

**路径参数：** `collection_id`、`person_id`；**查询参数：** `limit` 1～100，默认50；
`cursor`可选。`has_crop`表示是否存在已保存裁剪图。

```bash
curl -sS "${BASE_URL}/v1/collections/employees/persons/employee-001/faces?limit=50" \
  -H "${AUTH_HEADER}"
```

**结果：** HTTP 200返回`faces`和`next_cursor`。**常见错误：**
`400 invalid_cursor`、`404` Collection或Person。

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}/image`

**用途：** 下载启用保存后存在的112×112管理用人脸裁剪图。它不是原始上传图片。

**路径参数：** `collection_id`、`person_id`、`face_id`。

```bash
curl -sS "${BASE_URL}/v1/collections/employees/persons/employee-001/faces/FACE_ID/image" \
  -H "${AUTH_HEADER}" -o face-crop.jpg
```

**结果：** HTTP 200 `image/jpeg`，带`Cache-Control: no-store`。非JSON响应的请求ID只在
`x-request-id`头中。**常见错误：** `401`、`404` FaceSample或
`face_image_not_found`。

### `DELETE /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}`

**用途：** 删除一个FaceSample、embedding和可选裁剪图。

```bash
curl -sS -X DELETE "${BASE_URL}/v1/collections/employees/persons/employee-001/faces/FACE_ID" \
  -H "${AUTH_HEADER}"
```

**结果：** HTTP 204，无响应体；返回成功前同步从活动索引移除。**常见错误：**
`404`、`503 search_index_unavailable`。

## 搜索接口

### `POST /v1/collections/{collection_id}/search`

启用活体且为 `normal` 时，查询图片的 fake 或输入不合格返回 HTTP 422 `liveness_fake` 或 `liveness_input_rejected`，详情为 `error.details.liveness`，不会执行搜索；这与搜索成功但匹配列表为空不同。`observe` 继续搜索，并在查询人脸上返回活体结果。

**用途：** 用一张查询图片在指定人员库中执行1:N Person搜索。

**路径参数：** `collection_id`。**表单参数：** `image`必填；`limit` 1～100，默认5；
`threshold`可选0～1，省略后使用Collection阈值。旧`face_selection`参数不再支持。

```bash
curl -sS "${BASE_URL}/v1/collections/employees/search" -H "${AUTH_HEADER}" \
  -F 'image=@unknown.webp' -F 'limit=5' -F 'threshold=0.4'
```

**执行与结果：** 按Collection配置选择查询脸，扫描所有有效FaceSample，每个Person取
最高FaceSample相似度，按相似度降序且只返回达到阈值的结果。HTTP 200返回
`searched_face`、`matches`、实际`threshold`、`processing_ms`和`request_id`。
无匹配是成功的`matches: []`。

**常见错误：** `404` Collection、`409 collection_model_mismatch`、`413`、
`422 invalid_image`或`face_not_found`、`503 search_index_unavailable`或超时。

## RTSP Monitor监控任务

Monitor是持久化的服务端RTSP识别任务。配置保存在SQLite中，处于启用状态的Monitor会
在Server重启后自动恢复。系统不保存视频帧；事件只存在于有容量上限的内存环形缓冲区，
进程重启后丢失。解码器只保留最新帧，因此推理变慢时会降低实际执行频率，而不会积压
已经过时的视频帧。

### `POST /v1/monitors`

**用途：** 创建并可选择立即启动一个Monitor。**请求体：** 使用JSON；`source.url`
只允许`rtsp://`或`rtsps://`。凭据使用AES-GCM加密保存在`/data`，API只返回脱敏地址。

```json
{
  "id": "front-gate",
  "name": "公司前门",
  "description": "主入口",
  "enabled": true,
  "source": {"type": "rtsp", "url": "rtsp://viewer:secret@camera.example/live"},
  "collection_id": "employees",
  "inference_fps": 2.0,
  "match_threshold": null,
  "event_buffer_size": 1000,
  "event_policy": {
    "confirm_frames": 3,
    "absence_timeout_seconds": 3.0,
    "cooldown_seconds": 10.0,
    "emit_unknown": true
  },
  "preview_enabled": false
}
```

`match_threshold: null`表示继承Collection阈值；`event_buffer_size`范围10～10000。
Web预览默认关闭，不打开预览也会持续识别并产生事件。

**结果：** HTTP 201返回完整`monitor`、脱敏源、实际默认值和运行摘要。
**常见错误：** `400 invalid_request`、`404` Collection、`409 monitor_exists`、
`429 monitor_limit_exceeded`。

### `GET /v1/monitors`

**用途：** 分页列出持久化Monitor配置和简要运行状态。**查询参数：** `limit`范围
1～100，默认50；`cursor`必须原样使用上次响应中的`next_cursor`，客户端不应解析。

```bash
curl -sS "${BASE_URL}/v1/monitors?limit=50" -H "${AUTH_HEADER}"
```

**结果：** HTTP 200返回有序`monitors`和可空`next_cursor`。**常见错误：**
`400 invalid_cursor`表示令牌无效、被修改或作用域不匹配；启用认证时也可能返回401。

### `GET /v1/monitors/{monitor_id}`

**用途：** 读取一个Monitor的持久化配置和最新运行摘要。**路径参数：**
`monitor_id`是创建时由调用方指定的ID；响应中的RTSP地址不包含用户名、密码和查询值。

```bash
curl -sS "${BASE_URL}/v1/monitors/front-gate" -H "${AUTH_HEADER}"
```

**结果：** HTTP 200返回`monitor`，包括事件策略、预览开关、时间戳和`runtime`。
**常见错误：** `404 monitor_not_found`、`401 unauthorized`。

### `PATCH /v1/monitors/{monitor_id}`

**用途：** 局部修改Monitor，`id`不可修改。**请求体：** 至少提供一个创建接口中的
可变字段；`event_policy`也支持局部字段。只有更换RTSP地址或凭据时才发送`source`；
将`match_threshold`设为`null`可恢复继承Collection阈值。

```bash
curl -sS -X PATCH "${BASE_URL}/v1/monitors/front-gate" \
  -H "${AUTH_HEADER}" -H 'Content-Type: application/json' \
  -d '{"inference_fps":1.5,"event_policy":{"confirm_frames":5}}'
```

修改源、Collection、执行频率、阈值或事件策略会重启该任务；`enabled`控制启停。
名称、描述、预览和缓冲容量可以在线生效。

**结果：** HTTP 200返回更新后的完整`monitor`。**常见错误：** `400
invalid_request`、`404` Monitor或Collection、`429 monitor_limit_exceeded`。

### `DELETE /v1/monitors/{monitor_id}`

**用途：** 永久删除一个Monitor配置。**路径参数：** `monitor_id`。操作会停止解码与
推理线程、释放RTSP连接并丢弃内存状态和事件，但不会删除其Collection。

```bash
curl -sS -X DELETE "${BASE_URL}/v1/monitors/front-gate" \
  -H "${AUTH_HEADER}"
```

**结果：** HTTP 204，无响应体。**常见错误：** `404 monitor_not_found`、
`401 unauthorized`。

### `GET /v1/monitors/{monitor_id}/state`

启用活体且为 `normal` 时，未通过的人脸显示外层 `status: liveness_blocked` 和独立活体结果，计入 `liveness_blocked_faces`，不计入 `unknown_faces`，也不触发人员或陌生人进入事件。`observe` 继续识别。界面会区分“输入被拒绝”和“活体未通过”。

**用途：** 供无界面客户端或Web UI轮询当前运行状态。**返回字段：** 包含连接状态、
源分辨率/FPS、配置与实际推理频率、耗时、跳帧、当前已识别与陌生人脸、预览查看者、
重连次数和安全的最近错误；不会包含embedding或源凭据。

```bash
curl -sS "${BASE_URL}/v1/monitors/front-gate/state" -H "${AUTH_HEADER}"
```

**结果：** HTTP 200返回`state`，停用的Monitor通常为`stopped`。**常见错误：**
`404 monitor_not_found`、`401 unauthorized`。

### `GET /v1/monitors/{monitor_id}/events`

**用途：** 通过短轮询获取最近的进入、离开、错误和恢复事件，无需保持长连接。
**查询参数：** `limit`范围1～1000，默认100；下一次请求原样携带上次的
`next_cursor`。cursor是包含内部任务epoch和序号的签名不透明字符串。

第一次不带cursor时返回最新的若干事件；后续只返回更新事件。`truncated: true`表示
客户端落后于环形缓冲区，`stream_reset: true`表示任务已重启，旧cursor属于上一个
epoch。事件不落盘，Server进程重启后会丢失。

**结果：** HTTP 200返回`events`、`next_cursor`、`has_more`、`truncated`和
`stream_reset`。**常见错误：** `400 invalid_cursor`、`404 monitor_not_found`、
`401 unauthorized`。

### `GET /v1/monitors/{monitor_id}/preview.mjpeg`

**用途：** 打开可选的原始MJPEG预览。**认证：** 与其他API一样使用Bearer请求头，
不要把API Key放进URL。接口返回未画框的`multipart/x-mixed-replace` JPEG流，客户端
结合`/state`自行绘制人脸框、ID和相似度。

只有`preview_enabled=true`且至少有一个查看者时才进行JPEG编码；关闭预览不会停止
识别，传输中断后客户端应采用有上限的退避方式重连。

**结果：** HTTP 200长连接二进制流，不是JSON。**常见错误：** `409
preview_disabled`、`503 stream_unavailable`、`404 monitor_not_found`、401。

## 生产客户端检查表

- 先调用`/v1/health`，再读取`/v1/system`确认Provider、模型和阈值配置。
- GET可以安全重试；DELETE重试前先读取资源状态。创建Person/FaceSample遇到网络结果
  不确定时，先按调用方指定ID查询，不要直接重复注册。
- `429`和临时`503`可使用带抖动的有界指数退避；其他4xx应修正请求而不是重试。
- 升级前保存当前镜像digest、模型ID/digest、数据库备份和API版本。不要让两个Server
  进程同时写同一个`/data`目录。


模型、模型组件、Collection 和 FaceSample 的响应均不再包含 `model_version`，模型包以 `model_id` 标识。已有 Collection 的 `embedding_contract_id` 保持不变；新建 Collection 使用不含模型版本号的契约。外部特征调用方应读取并使用目标 Collection 返回的契约 ID。
