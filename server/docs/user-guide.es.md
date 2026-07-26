# Guía de usuario de InsightFace Server

**Idiomas:** [English](user-guide.md) · [中文](user-guide.zh-CN.md) · [日本語](user-guide.ja.md) · [Deutsch](user-guide.de.md) · Español · [Français](user-guide.fr.md) · [Русский](user-guide.ru.md) · [Português](user-guide.pt.md) · [한국어](user-guide.ko.md)

Esta guía lleva a un usuario nuevo desde un directorio vacío hasta la primera búsqueda correcta. Las mismas funciones están disponibles en la Web UI, `/v1` y el SDK Python. Consulte todos los campos y resultados HTTP en la [guía de API](api.es.md).

## Desde cero hasta la primera búsqueda

CPU requiere Linux x86_64, Docker Engine y Docker Compose. CUDA añade un Driver NVIDIA compatible y NVIDIA Container Toolkit; no instale CUDA, cuDNN, ORT, Python ni OpenCV en el host.

```bash
mkdir -p server/.models
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml run --rm models install buffalo_l
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

Para GPU use `compose.cuda12.yml` y el puerto `18098`. El instalador muestra la licencia antes de descargar; los modelos públicos de InsightFace son solo para investigación no comercial salvo licencia comercial independiente.

El Compose incluido desactiva la autenticación por defecto para evaluación aislada. Antes de exponer el servicio, defina `INSIGHTFACE_AUTH_ENABLED=true` y un `INSIGHTFACE_API_KEY` largo. Después compruebe el Panel, cree una Collection, registre una Person y busque con otra foto. Detenga con `docker compose ... down` sin `-v` para conservar el volumen.

## 1. Acceso y estado

Abra `http://SERVIDOR:18097/` para CPU o `http://SERVIDOR:18098/` para CUDA 12. Si hay autenticación, use **Configurar clave API**, pegue la clave del operador y aplíquela a la pestaña. Solo permanece en memoria y se elimina al recargar o cerrar.

Compruebe en **Panel** o **Sistema** que servicio, base de datos, modelos y Provider estén listos. CUDA debe mostrar `CUDAExecutionProvider` y nunca vuelve silenciosamente a CPU.

## 2. Crear una Collection

En **Colecciones** → **Nueva colección**, indique un ID estable, nombre, umbral
coseno (`0.4` inicialmente), perfil disponible, capacidad y máximo de
FaceSamples por persona. Guardar como JPEG un `bounding-box crop` redimensionado
a 112×112 está desactivado por defecto; no es la entrada alineada de
reconocimiento.

La Collection queda fijada al ID, versión, digest, dimensión y preprocesamiento del modelo. Tras cambiar el modelo, una colección antigua sigue visible, pero su registro y búsqueda se rechazan si el contrato no coincide.

El perfil de detección copia los valores del sistema al crear la Collection y después permite cambiar tamaños de entrada, umbrales de detección/NMS y estrategia de un rostro. `largest` prioriza el área; `center_largest` maximiza `área - 2,0 × distancia en píxeles al cuadrado entre el centro del cuadro y el de la imagen`. La confianza de detección no participa en esta puntuación.

## 3. Registrar una Person

En **Personas**, seleccione la Collection y **Registrar persona**. Puede indicar ID, nombre, ID externo, metadata JSON y una o varias imágenes JPEG, PNG o WebP.

- `off`: usa la estrategia de un rostro de la Collection y permite varios rostros.
- `standard`: exige un rostro utilizable y valida tamaño, detección, nitidez, iluminación y pose.
- `strict`: además exige que la mejor similitud interna sea mayor que la mejor similitud con otra persona.

El lote permite éxito parcial y explica cada rechazo. No se guardan originales. `external_trusted` acepta un embedding normalizado L2; la imagen sigue siendo obligatoria para detección y calidad, pero no se vuelve a extraer el vector.

## 4. Detectar, comparar y buscar

**Detectar** muestra cajas, cinco puntos, detección y calidad; sin rostros devuelve una lista vacía correcta. **Comparar** usa el perfil del sistema o de una Collection para elegir un rostro por imagen y devuelve `similarity` coseno, `threshold` y `matched`. La similitud no es probabilidad.

En **Buscar**, seleccione Collection e imagen. La puntuación de una persona es la mayor similitud entre sus FaceSamples. Los resultados se ordenan de mayor a menor; sin coincidencia es una lista vacía. Cada muestra se confirma primero en SQLite y se añade al índice antes de responder. Al reiniciar, el índice se reconstruye desde SQLite.

## 5. Monitorización de cámara RTSP

En **Monitorización de cámaras**, cree un Monitor persistente y configure fuente RTSP, Collection, frecuencia, umbral opcional y política de eventos. La vista previa está desactivada por defecto; reconocimiento y eventos continúan sin ella. Al activarla, la Web UI dibuja cajas verdes para personas registradas y naranjas para rostros desconocidos usando `/state` sobre imágenes crudas.

El Monitor funciona independientemente del navegador y las tareas activas se restauran tras reiniciar el servidor. La configuración vive en SQLite y las credenciales RTSP cifradas en `/data`; no se guardan fotogramas ni eventos. Los eventos solo permanecen en un búfer de memoria limitado. El decodificador conserva el último fotograma y omite los obsoletos en lugar de acumularlos.

## 6. Datos y seguridad

Conserve `/data` y monte `/models` como solo lectura. Antes de operaciones masivas, copie SQLite y los recortes juntos. Las claves se guardan como hash; iniciar el mismo volumen con otro `INSIGHTFACE_API_KEY` rota la clave activa. No registre imágenes, embeddings ni claves.

El explorador de esquemas OpenAPI para desarrolladores está en `/docs`; las instrucciones prácticas de la API están en esta ayuda. Incluya `x-request-id` al comunicar incidencias. `401` indica clave, `409 collection_model_mismatch` contrato de modelo y `422 face_not_found` ausencia de rostro válido.

## 7. Modelos y licencias

Las imágenes no incluyen modelos. El inicio normal permanece sin conexión; el
servicio puntual `models` instala en `server/.models`:

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models verify buffalo_l
```

Se admiten `buffalo_l` (`det_10g.onnx` + `w600k_r50.onnx`), `buffalo_m`,
`buffalo_sc` y `antelopev2`. La instalación crea `manifest.json` y
`MODEL.LICENSE` firmada. Sin `--accept-license`, la herramienta muestra los
términos y termina sin descargar. Los modelos públicos preentrenados de
InsightFace son solo para investigación no comercial salvo licencia comercial
independiente.

## 8. Configuración de inicio y búsqueda

`server/config/server.toml` se lee una vez al arrancar; los cambios requieren
reiniciar. Los valores son `input_sizes=[[96,96],[512,512]]`, umbral de
detección `0.50`, NMS `0.40`, `single_face_selection="largest"` y máximo 100
rostros. SCRFD ejecuta cada resolución, lleva los candidatos a la imagen
original y hace un único NMS global. `max_concurrency="auto"` equivale a CPU 4
y CUDA 8. `[web].disabled=true` conserva solo `/v1` y `/openapi.json`.

System anuncia únicamente los perfiles disponibles. El perfil se fija al crear
la Collection y no puede cambiarse por petición:

- `fp32_v1`: CPU/CUDA estándar;
- `fp16_v1`: CUDA;
- `bf16_v1`: CPU compatible o CUDA SM80+;
- `int8_x736_v1`: INT8 recomendado en CPU/CUDA, acumulación INT32;
- `int8_x1000_v1`: compatibilidad de Collections existentes.

Todos recorren cada FaceSample y no son índices ANN; la salida sigue siendo raw
cosine. `capacity_rows` vale `100000`, el límite global `10000000` y
`max_faces_per_person=20`. Para 512 dimensiones, solo el vector ocupa unos
2.048 bytes FP32, 1.024 FP16/BF16 o 512 INT8 por fila.

## 9. SDK, compilación y operación de datos

El SDK Python admite ruta, bytes y objeto tipo archivo, con métodos tipados para
Detect, Compare, Collections, registro, Search y Monitors. Consulte el contrato
HTTP en la [guía de API](api.es.md).

El usuario puede compilar desde un checkout completo:

```bash
make -C server build-cpu
make -C server build-cuda12
```

Use `--pull never` con Compose para usar la imagen local. Los tags inmutables
son `0.2.0-cpu` y `0.2.0-cuda12`; `cpu` y `cuda12` apuntan a la última estable
y no existe `latest`. Antes de actualizar, pare las escrituras y haga una copia
SQLite segura de `/data` y los crops. No use `docker compose down -v`: elimina
el volumen de datos.

## 10. GPU, red y resolución de problemas

La imagen CUDA contiene CUDA Runtime 12.9.1, cuDNN 9.24.0 y
`onnxruntime-gpu==1.27.0`. Turing/Ampere/Ada/Hopper requieren R535 o posterior,
Blackwell/RTX 50 requieren 570.26 o posterior; para nuevas instalaciones se
recomienda una R580 estable o posterior. El inicio valida GPU, Compute
Capability, Driver, CUDA/cuDNN/ORT, Provider, Sessions reales y warm-up, y
rechaza el fallback silencioso a CPU.

Al exponer la red, termine HTTPS en un proxy inverso de confianza, limite
orígenes CORS, tasa, cuerpo y tiempo, y proteja `/data` y copias como datos
biométricos. No registre imágenes, embeddings ni claves. La fase uno tiene una
única API Key sin roles; no es autorización multi-tenant.
