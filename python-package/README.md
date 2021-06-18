## Python package of insightface README


For insightface pip-package <= 0.1.5, we use MXNet as inference backend, please download all models from [onedrive](https://1drv.ms/u/s!AswpsDO2toNKrUy0VktHTWgIQ0bn?e=UEF7C4), and put them all under `~/.insightface/models/` directory.

Starting from insightface>=0.2, we use onnxruntime as inference backend.

For insightface>=0.3.3, model package will be downloaded automatically. 
For insightface==0.3.2, you must first download our **antelope** model release by command:

```
insightface-cli model.download antelope
```

Model Pack:

| Name      | Detection Model  | Recognition Model  | Alignment |
| ----------------------- | -----------------   | ----- | ----- |
| **antelope** | SCRFD-10GF        | ResNet100@Glint360K | 2d106 & 3d68 |
