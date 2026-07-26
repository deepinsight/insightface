# InsightFace Server REST API使用手册

**语言：** [English](api.md) · 中文 · [日本語](api.ja.md) · [Deutsch](api.de.md) · [Español](api.es.md) · [Français](api.fr.md) · [Русский](api.ru.md) · [Português](api.pt.md) · [한국어](api.ko.md)

本文覆盖每个公开接口的调用方式、参数含义、服务端执行过程、成功结果和常见错误。
如果容器和模型还没有启动，请先阅读
[分步用户指南](user-guide.zh-CN.md)。实时OpenAPI Schema位于`/docs`和
`/openapi.json`；本文负责解释“怎么用”和“结果代表什么”。

## 通用约定

- API基础路径为`/v1`，JSON字段统一使用`snake_case`。
- Collection/PATCH请求使用`application/json`；图片和注册使用
  `multipart/form-data`；裁剪图返回`image/jpeg`；摄像头预览返回MJPEG流。
- JPEG、PNG和WebP均支持。默认压缩图片上限10 MiB、解码像素上限4000万、整个请求
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

**用途：** 查看当前已验证的检测/识别模型、实际Provider和模型授权摘要。

**参数：** 无。

**执行与结果：** HTTP 200返回`models`、`execution_provider`和`license`，不返回
ONNX文件内容或签名私钥。

```bash
curl -sS "${BASE_URL}/v1/models" -H "${AUTH_HEADER}"
```

**常见错误：** `401 unauthorized`。

## 无状态人脸接口

### `POST /v1/detect`

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

**执行与结果：** 分配索引并固定当前模型ID、版本、digest、512维特征与预处理版本。
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
