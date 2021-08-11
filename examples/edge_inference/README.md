## Test report on a private ID2Wild dataset

#### 1:N Identification

| Device       | Model                 | Backend              | Time | Gallery | Rank1-Acc | TAR@FAR<=e-3 |
| ------------ | --------------------- | -------------------- | ---- | ------- | --------- | ------------ |
| NVIDIA-V100  | IR50-Glint360K        | onnxruntime          | 4ms  | 50K     | 80.94     | 30.77        |
|              | IR50-Glint360K-Int8   | TensorRT             |      | 50K     |           |              |
| Khadas-A311D | IR50-Glint360K-Uint8  | Tengine-Uint8        | 26ms | 50K     | 77.83     | 26.58        |
| Khadas-A311D | IR50-Glint360K-Mixed* | Tengine              | 26ms | 50K     | 79.38     | 28.59        |
| NXP-imx8p    | IR50-Glint360K-Uint8  | Tengine-Uint8        | 24ms | 50K     | 77.87     | 26.80        |
| NXP-imx8p    | IR50-Glint360K-Mixed* | Tengine              | 24ms | 50K     | 79.42     | 28.39        |
| RV1126       | IR50-Glint360K-Uint8  | RKNN                 | 38ms | 50K     | 75.60     | 24.23        |
| RV1126       | R50-Glint360K-Mixed*  | RKNN                 | 38ms | 50K     | 77.82     | 26.30        |
| Jetson NX    | IR50-Glint360K-Int8   | TRT 7.1.3-1,CUDA10.2 | 16ms | 50K     | 79.26     | 31.07        |

(Mixed* means use float model for gallery while uint8 model for probe images)

(Result features are all in float32 type)



det: 11.5ms

Precision-Recall-Thresh: 0.9802412111880934 0.4781275423993992 0.5589999999999999

