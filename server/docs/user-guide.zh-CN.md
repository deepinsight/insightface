# InsightFace Server 用户指南

**语言：** [English](user-guide.md) · 中文 · [日本語](user-guide.ja.md) · [Deutsch](user-guide.de.md) · [Español](user-guide.es.md) · [Français](user-guide.fr.md) · [Русский](user-guide.ru.md) · [Português](user-guide.pt.md) · [한국어](user-guide.ko.md)

这是面向第一次使用者的分步操作指南：从一个空的项目目录开始，直到创建人员库、
注册人员并得到第一次搜索结果。相同能力可以通过 Web UI、`/v1` API 和 Python SDK
使用；每个HTTP字段和响应的完整说明请查看
[API使用手册](api.zh-CN.md)。

## 从这里开始：从零启动到第一次成功搜索

CPU版需要Linux x86_64、Docker Engine和Docker Compose。CUDA版还需要兼容的
NVIDIA Driver与NVIDIA Container Toolkit；宿主机不需要安装CUDA、cuDNN、
ONNX Runtime、Python或OpenCV。

CPU启动示例：

```bash
mkdir -p server/.models
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml run --rm models install buffalo_l
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

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

## 1. 登录并检查就绪状态

CPU 打开 `http://服务器地址:18097/`，CUDA 12 打开 `http://服务器地址:18098/`。如果启用了认证，点击 **配置 API Key**，粘贴管理员提供的 Key，再选择 **在此标签页使用**。Key 只保留在当前标签页内存中，刷新或关闭页面后即清除。

注册数据前请查看 **仪表盘** 或 **系统**。服务、数据库、模型和 Provider 均应为就绪。CUDA 部署必须显示 `CUDAExecutionProvider`，不会静默回退到 CPU。

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

Collection 会固定绑定模型 ID、版本、digest、特征维度和预处理版本。检测配置在创建时复制系统默认值，之后可以单独修改；修改从下一次请求生效并递增 `detection_revision`，但不会重新处理已有 FaceSample。`largest` 优先面积；`center_largest` 最大化 `人脸面积 - 2.0 × 人脸框中心到图像中心的像素距离平方`，检测置信度不参与该分数。

## 3. 注册 Person

打开 **人员**，选择 Collection，再点击 **注册人员**。可填写稳定的 Person ID、姓名、外部 ID 和 JSON metadata，然后拖入一张或多张 JPEG、PNG 或 WebP 图片。

入库审查模式：

- `off`：使用 Collection 的单脸挑选策略，允许图片中存在多张脸；
- `standard`：要求一张可用脸，并检查尺寸、检测分数、清晰度、亮度和姿态；
- `strict`：在 standard 基础上，要求样本的最佳类内相似度高于最佳类外相似度。

批量注册支持部分成功。请根据每张失败图片的原因处理后重试；系统不保存被拒绝的
原图。启用人脸图保存时，只保存缩放为 112×112 的 `bounding-box crop`，不保存
原始上传图片或识别模型使用的对齐输入。

可信系统可以用 `external_trusted` 提交预先抽取并 L2 归一化的 embedding。仍须同时提供图片完成检测和质量审查，但服务不会再次抽取特征；embedding contract 必须与 Collection 完全一致。

## 4. 检测与比对

在 **检测** 中上传单图，可查看人脸框、五点关键点、检测分数和启发式质量信息。无人脸是成功的空列表。

在 **比对** 中分别上传 source 和 target，并可选择系统或 Collection 检测配置。配置中的策略从两张图各挑选一张可用脸，返回原始 cosine `similarity`、`threshold` 和 `matched`。Similarity 不是概率；任一图片没有可用脸时返回 `422 face_not_found`。

## 5. 搜索人员库

打开 **搜索**，选择 Collection，上传查询图片并设置返回数量；也可以临时覆盖阈值。系统按 Collection 检测配置挑选查询脸，按相似度降序返回。Person 得分取其所有 FaceSample 的最高相似度。无匹配是成功的空列表。

新 FaceSample 会先提交到 SQLite，再加入内存索引，然后才返回成功；删除同时更新两处。重启时从 SQLite 重建索引，SQLite 始终是权威数据源。

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

- 持久化挂载 `/data`，`/models` 只读挂载。
- 停止写入后备份 SQLite 和裁剪图目录，或使用 SQLite 安全快照方式。
- API Key 只以 hash 保存。后续启动同一数据卷时传入不同 `INSIGHTFACE_API_KEY`，会主动轮换当前 Key。
- 不要记录图片、embedding 或 Key；除非确有需要，不要开启宽泛 CORS。
- 公开镜像不包含模型。InsightFace提供的开源预训练模型（包括 `buffalo_l`）仅限
  非商业研究使用；商业使用需要单独许可，请访问 <https://www.insightface.ai>。
  **系统** 页面也会显示相同提示。

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
`buffalo_sc`（`det_500m.onnx` + `w600k_mbf.onnx`）和`antelopev2`
（`scrfd_10g_bnkps.onnx` + `glintr100.onnx`）。安装会生成`manifest.json`
与签名的`MODEL.LICENSE`。不带`--accept-license`时，工具只显示许可并退出，不会
下载。`models verify`会核验包身份、签名、有效期和当前授权状态。

InsightFace公开预训练模型默认仅限非商业研究；商业使用需另行获取授权。私有模型也
可以使用同样的manifest和离线签名许可。许可按`model_id`表达授权，是合规凭证，
不是DRM，也不要求模型文件SHA-256保持不变。

## 12. 仅启动时生效的配置

通用配置文件为`server/config/server.toml`，Compose将其只读挂载到
`/etc/insightface/server.toml`。修改后必须重启容器，默认值如下：

```toml
[inference]
max_concurrency = "auto" # CPU为4，CUDA为8

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
系统配置只在启动时读取，不提供运行时修改API。新Collection会复制系统检测配置，
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

用户可从完整仓库自行构建：

```bash
make -C server build-cpu
make -C server build-cuda12
```

随后在Compose的模型安装与`up`命令中加入`--pull never`，即可使用本地镜像。构建
使用固定基础镜像和锁定依赖，但仍需联网获取这些输入。公开版本Tag为
`0.2.0-cpu`和`0.2.0-cuda12`；移动Tag `cpu`/`cuda12`分别指向最新稳定版本，
明确不发布含义模糊的`latest`。

升级前停止写入，使用SQLite安全方式备份`/data`以及可选裁剪图，并保留`/models`
和许可文件。先用数据副本启动新镜像，检查migration、`/v1/health`、模型契约和一条
已知Search，再切换正式数据。停止使用`docker compose down`且不要带`-v`；
`docker compose down -v`会删除命名数据卷。

跨网络使用时，应在可信反向代理终止HTTPS，只开放必要的CORS origin，并在边缘限制
速率、请求体和超时。数据卷及备份应按生物识别数据保护。第一阶段只有一个不区分权限
的API Key，不应把它当作多租户授权系统。
