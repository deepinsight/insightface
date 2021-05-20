## Python package of insightface README

insightface==0.2 is ready now. Please update with `pip install -U insightface`

For insightface pip-package <= 0.1.5, we use MXNet as inference backend, please download all models from [onedrive](https://1drv.ms/u/s!AswpsDO2toNKrUy0VktHTWgIQ0bn?e=UEF7C4), and put them all under `~/.insightface/models/` directory.

Starting from insightface>=0.2, we use onnxruntime as inference backend, please download our **antelope** model release from [onedrive](https://1drv.ms/u/s!AswpsDO2toNKrU0ydGgDkrHPdJ3m?e=iVgZox), and put it under `~/.insightface/models/`, so there're onnx models at `~/.insightface/models/antelope/*.onnx`.

InsightFace-0.2 Model Pack:

| Name      | Detection Model  | Recognition Model  | Download |
| ----------------------- | -----------------   | ----- | ----- |
| **antelope** | SCRFD-10GF        | ResNet100@Glint360K | [link](https://1drv.ms/u/s!AswpsDO2toNKrU0ydGgDkrHPdJ3m?e=iVgZox) |
