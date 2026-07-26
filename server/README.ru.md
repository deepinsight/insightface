# InsightFace Server

**Языки:** [English](README.md) · [中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · Русский · [Português](README.pt.md) · [한국어](README.ko.md)

> **Один GPU. 50M+ векторов лиц. Сверхбыстрый поиск с INT8-квантованием признаков без существенной потери точности.**

**Самостоятельно размещаемый сервер распознавания лиц с Web UI, понятным REST
API, SQLite и локальным CPU- или NVIDIA GPU-инференсом в одном контейнере.**

```text
загрузить изображение -> обнаружить, сравнить, зарегистрировать или найти
```

> **Лицензия модели:** публичные предобученные модели InsightFace обычно
> доступны только для некоммерческих исследований. Для коммерческого применения
> требуется отдельное разрешение [InsightFace](https://www.insightface.ai).

InsightFace Server — более простая и ориентированная на приватность альтернатива
AWS Rekognition для типовых сценариев на собственной инфраструктуре. Изображения,
embeddings, модели и индексы могут оставаться внутри вашей сети. Это **не**
AWS-совместимая замена: SigV4, IAM, Region и семантика ресурсов AWS не
реализованы.

Текущая версия: **0.2.0**, Linux x86_64.

| Среда | Image |
| --- | --- |
| CPU | `ghcr.io/deepinsight/insightface-server:0.2.0-cpu` |
| NVIDIA GPU | `ghcr.io/deepinsight/insightface-server:0.2.0-cuda12` |

Подвижные теги `cpu` и `cuda12` указывают на последнюю стабильную версию своей
семьи. Неоднозначный `latest` не используется. См.
[Maintainer Guide — English](docs/maintainer-guide.md) о правилах релиза.

![Английский Dashboard InsightFace Server](docs/images/customer/dashboard-en.jpg)

## Основные возможности

- Детекция SCRFD, пять landmarks, выравнивание, ArcFace embeddings, L2
  normalization, исходный cosine similarity и точный поиск Person 1:N.
- Многоразмерная детекция с единым NMS после объединения и выбором
  `largest` либо `center_largest`.
- `Collection -> Person -> FaceSample`, привязка Collection к модели,
  регистрация нескольких изображений с частичным успехом, metadata и понятными
  причинами отказа.
- `review_mode` регистрации: `off`, `standard`, `strict`; дополнительно
  предвычисленные `external_trusted` embeddings.
- Точный GPU-поиск с хранением векторов FP32, FP16, BF16 и INT8.
- Многоязычный Web UI: Dashboard, Collections, People, Detect, Compare, Search,
  RTSP-мониторинг, System и Help.
- 29 snake_case REST-операций под `/v1`, включая защищённый
  `/v1/embeddings`, и лёгкий типизированный Python SDK.
- Серверные RTSP Monitors с ограниченными событиями в памяти, несколькими
  клиентами и необязательным `preview.mjpeg`; закрытие браузера не останавливает
  мониторинг.
- SQLite как постоянный источник, перестраиваемые точные индексы в памяти,
  `/models` только для чтения, постоянный `/data`, миграции, health checks и
  строгая проверка CUDA без скрытого CPU fallback.
- JPEG, PNG и WebP; исходные загрузки по умолчанию не сохраняются.

### Производительность GPU-поиска на RTX 5090

На одной NVIDIA GeForce RTX 5090 (32 607 MiB) нативный точный CUDA flat-индекс
с INT8 вместил до **58,9M 512-мерных векторов изображений**.

| Тип данных GPU | Максимум векторов изображений | 10M Top-5 p50 | 10M последовательных QPS |
| --- | ---: | ---: | ---: |
| FP32 | 15,8M | 12,84 ms | 77,85 |
| FP16 | 30,7M | 6,83 ms | 146,32 |
| BF16 | 30,7M | 6,83 ms | 146,33 |
| INT8 | **58,9M** | **3,84 ms** | **260,81** |

INT8 дал в 3,73 раза большую измеренную ёмкость и в 3,35 раза больший
10M Top-5 throughput, чем FP32. Это только GPU-измерения на одной RTX 5090 с
Driver 580.105.08 и CUDA 12.9. Ёмкость — изолированный предел нативного индекса
без загруженных ONNX-моделей и нагрузки Server. Скорость измерена ровно на 10M
векторах изображений, полным точным GPU-resident Top-5 сканированием, с одним
запросом одновременно, 10 прогревами и 100 измерениями. Поиск точен внутри
каждого сохранённого представления, однако квантование может изменить score
относительно FP32. В production нужен запас VRAM для моделей, запросов,
параллельности, перестроения индекса и allocator.

### Точность MR-ALL на многорасовом наборе ICCV21-MFR

Нативные профили поиска оценены на многорасовом (MR) тестовом наборе
[ICCV21-MFR](../challenges/iccv21-mfr/) по протоколу MR-ALL 1:1 для всех пар
при FAR `1e-6`. Во всех профилях используются одни и те же 512-мерные
L2-нормализованные embeddings `buffalo_l`, однократно извлечённые через Server
API; меняются только представление хранения векторов и вычисления при поиске.

| Профиль поиска | MR-ALL при FAR 1e-6 | Порог cosine | Разница с FP32 |
| --- | ---: | ---: | ---: |
| FP32 | 91,249107 % | 0,407787 | — |
| FP16 | 91,249197 % | 0,407787 | +0,000090 п. п. |
| BF16 | 91,248502 % | 0,407787 | -0,000605 п. п. |
| **INT8** | **91,248005 %** | **0,407739** | **-0,001102 п. п.** |

**INT8 не показывает существенной потери точности в этом benchmark:**
при принятом в challenge округлении до двух знаков FP32 и INT8 дают одинаковые
**91,25 % MR-ALL**, а разница без округления составляет лишь 0,0011 процентного
пункта. При этом сохраняются показанные выше преимущества: в 3,73 раза большая
измеренная ёмкость и в 3,35 раза больший 10M Top-5 throughput. Сравнение
относится к точности хранения векторов и поиска, а не к INT8-инференсу модели.

![Управление Collections на английском](docs/images/customer/collections-en.jpg)

![RTSP Monitor на английском; частный адрес скрыт](docs/images/customer/monitoring-en.jpg)

## Быстрый старт

Требования:

- Linux x86_64 с Docker Engine и Docker Compose;
- для CUDA — совместимая NVIDIA GPU, NVIDIA Driver и NVIDIA Container Toolkit.

На хосте не нужны Python, OpenCV, ONNX Runtime, CUDA Toolkit или cuDNN.
Публичные Images не содержат модели, клиентские данные, API Keys или
production-конфигурацию.

В полном checkout InsightFace установите модель в `server/.models`:

```bash
mkdir -p server/.models
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license
```

Инструмент также поддерживает `buffalo_m`, `buffalo_sc` и `antelopev2`. Он
создаёт `manifest.json` и подписанную `MODEL.LICENSE`; `models verify` проверяет
пакет. Условия модели отделены от лицензии исходного кода Server.

Запустить CPU:

```bash
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

Вместо этого запустить CUDA 12:

```bash
docker compose -f server/deploy/compose.cuda12.yml pull
docker compose -f server/deploy/compose.cuda12.yml \
  run --rm models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cuda12.yml up -d
curl -fsS http://127.0.0.1:18098/v1/health
```

Для CPU откройте `http://СЕРВЕР:18097/`, для CUDA —
`http://СЕРВЕР:18098/`. Создайте Collection, зарегистрируйте Person по одной
или нескольким фотографиям и найдите её по другой. `docker compose ... down`
без `-v` сохраняет volume.

Поставляемые Compose по умолчанию отключают аутентификацию для изолированной
оценки. Перед доступом других пользователей или сетей:

```bash
export INSIGHTFACE_AUTH_ENABLED=true
export INSIGHTFACE_API_KEY='замените-на-длинный-случайный-секрет'
docker compose -f server/deploy/compose.cpu.yml up -d
```

Полный первый сценарий описан в [руководстве пользователя](docs/user-guide.ru.md).

## Сборка из исходного кода

Dockerfiles копируют `server/` и выбранные inference-модули из
`python-package/insightface/`, поэтому контекстом сборки служит весь репозиторий.

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

`--pull never` гарантирует использование локального Image. Сборка всё равно
загружает зафиксированные базовые Images и зависимости, а установка модели —
отдельно принятый пакет модели.

## Основные правила

- Similarity — исходный cosine, не вероятность. Threshold находится в
  `0.0..1.0`, значение по умолчанию `0.4`.
- Collection фиксирует модель и embedding contract. При несовпадении она видна,
  но регистрация/поиск возвращает `collection_model_mismatch`.
- Стартовый Detection Profile копируется в новые Collections; затем профиль
  можно независимо менять для последующих запросов.
- Необязательное сохранение лица записывает bounding-box JPEG crop,
  масштабированный до 112x112, а не оригинал и не выровненный вход распознавания;
  по умолчанию выключено.
- Commit SQLite является источником истины. Индекс синхронизируется до успешного
  ответа регистрации/удаления и строится из SQLite после перезапуска.
- Ответы содержат `x-request-id`; списки используют непрозрачный подписанный
  cursor.

Точные поля, defaults, жизненный цикл и ошибки поддерживаются только в
подробной документации ниже.

## API и SDK

Основные группы:

- система: `/v1/health`, `/v1/system`, `/v1/models`;
- операции без состояния: `/v1/detect`, `/v1/compare`, `/v1/embeddings`;
- CRUD Collection, Person и FaceSample;
- поиск Person в Collection;
- конфигурация, состояние, события и preview RTSP Monitor.

Все параметры, ответы, ошибки и примеры содержит
[руководство REST API](docs/api.ru.md). Интерактивный OpenAPI доступен в `/docs`.

```python
from insightface_server import Client

with Client("http://localhost:18097", api_key=None) as client:
    faces = client.detect("photo.jpg")
    matches = client.search("employees", "unknown.jpg", limit=5)
```

Установка, входные типы, методы и полные сценарии SDK приведены в
[руководстве пользователя](docs/user-guide.ru.md).

## Безопасность

Изображения лиц и embeddings — биометрические данные. При сетевом доступе
включите аутентификацию, завершайте HTTPS на доверенном reverse proxy,
ограничьте Docker и volumes, не включайте широкий CORS и определите backup,
retention, удаление, согласие и реакцию на инциденты. Не записывайте в логи
изображения, embeddings, RTSP credentials и API Keys.

Server не включает TLS, учётные записи, RBAC, cloud IAM или правовой слой.
Эксплуатация и безопасность описаны в
[руководстве пользователя](docs/user-guide.ru.md).

## Границы первой фазы

Не реализованы AWS/CompreFace-совместимость, CUDA 11, Jetson, ARM64, Windows
Container, TensorRT, Kubernetes, распределённые Workers, постоянные события
Monitor или запись/NVR, liveness, deepfake и демографические атрибуты.

## Документация

- [Руководство пользователя](docs/user-guide.ru.md) — установка, настройки,
  модели, Web UI, SDK, GPU, безопасность, backup и диагностика.
- [Руководство REST API](docs/api.ru.md) — все endpoints, поля, поведение,
  результаты, ошибки, pagination и примеры.
- [Maintainer Guide — English](docs/maintainer-guide.md) — архитектура,
  внутренний поиск, тесты, вклад и релизы контейнеров.

GitHub и справка Web UI читают одни и те же локализованные Markdown; отличается
только отображение.

## Лицензия

Единая точка входа для лицензий — [LICENSING.md](LICENSING.md). Код Server и
Python SDK распространяются по MIT License; это заявление не охватывает файлы
и веса моделей, datasets или сторонние компоненты. Публичные модели InsightFace
обычно ограничены некоммерческими исследованиями без отдельного разрешения.
Коммерческое лицензирование: <https://www.insightface.ai>.
