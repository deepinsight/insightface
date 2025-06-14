# InspireFace Python API

InspireFace 提供了简单易用的 Python API，通过 ctypes 封装底层动态链接库实现。您可以通过 pip 安装最新发布版本，或使用项目自行编译的动态库进行配置。

## 快速安装

### 通过 pip 安装（推荐）

```bash
pip install inspireface
```

### 手动安装

1. 首先安装必要的依赖：
```bash
pip install loguru tqdm opencv-python
```

2. 将编译好的动态库复制到指定目录：
```bash
# 将编译好的动态库复制到对应系统架构目录
cp YOUR_BUILD_DIR/libInspireFace.so inspireface/modules/core/SYSTEM/CORE_ARCH/
```

3. 安装 Python 包：
```bash
python setup.py install
```

## 快速开始

以下是一个简单的示例，展示如何使用 InspireFace 进行人脸检测和关键点绘制：

```python
import cv2
import inspireface as isf

# 创建会话，启用所需功能
session = isf.InspireFaceSession(
    opt=isf.HF_ENABLE_NONE,  # 可选功能
    detect_mode=isf.HF_DETECT_MODE_ALWAYS_DETECT  # 检测模式
)

# 设置检测置信度阈值
session.set_detection_confidence_threshold(0.5)

# 读取图像
image = cv2.imread("path/to/your/image.jpg")
assert image is not None, "请检查图像路径是否正确"

# 执行人脸检测
faces = session.face_detection(image)
print(f"检测到 {len(faces)} 个人脸")

# 在图像上绘制检测结果
draw = image.copy()
for idx, face in enumerate(faces):
    # 获取人脸框位置
    x1, y1, x2, y2 = face.location
    
    # 计算旋转框参数
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    size = (x2 - x1, y2 - y1)
    angle = face.roll
    
    # 绘制旋转框
    rect = ((center[0], center[1]), (size[0], size[1]), angle)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    cv2.drawContours(draw, [box], 0, (100, 180, 29), 2)
    
    # 绘制关键点
    landmarks = session.get_face_dense_landmark(face)
    for x, y in landmarks.astype(int):
        cv2.circle(draw, (x, y), 0, (220, 100, 0), 2)
```

## 更多示例

项目提供了多个示例文件，展示了不同的功能：

- `sample_face_detection.py`: 基础人脸检测
- `sample_face_track_from_video.py`: 视频人脸跟踪
- `sample_face_recognition.py`: 人脸识别
- `sample_face_comparison.py`: 人脸比对
- `sample_feature_hub.py`: 特征提取
- `sample_system_resource_statistics.py`: 系统资源统计

## 运行测试

项目包含单元测试，您可以通过修改 `test/test_settings.py` 中的参数来调整测试内容：

```bash
python -m unittest discover -s test
```

## 注意事项

1. 确保系统已安装 OpenCV 和其他必要依赖
2. 使用前请确保动态库已正确安装
3. 建议使用 Python 3.7 或更高版本
