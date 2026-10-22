# Guía de usuario de InsightFace Server

**Idiomas:** [English](user-guide.md) · [中文](user-guide.zh-CN.md) · [日本語](user-guide.ja.md) · [Deutsch](user-guide.de.md) · Español · [Français](user-guide.fr.md) · [Русский](user-guide.ru.md) · [Português](user-guide.pt.md) · [한국어](user-guide.ko.md)

Esta guía lleva a un usuario nuevo desde un directorio vacío hasta la primera búsqueda correcta. Las mismas funciones están disponibles en la Web UI, `/v1` y el SDK Python. Consulte todos los campos y resultados HTTP en la [guía de API](api.es.md).

Los modelos se identifican mediante `model_id`; las respuestas omiten `model_version`.

Una actualización del Server con el mismo modelo de reconocimiento y contrato conserva `embedding_contract_id`, muestras y embeddings de las Collections existentes. Cambiar el modelo es una migración distinta; si el contrato no coincide, registro y búsqueda devuelven `collection_model_mismatch`.

Para usar la prueba de vida, consulte [configuración, instalación y resultados](#addon-opcional-de-prueba-de-vida). Cada flujo explica también sus efectos.

## Desde cero hasta la primera búsqueda

CPU requiere Linux x86_64, Docker Engine y Docker Compose. CUDA añade un Driver NVIDIA compatible y NVIDIA Container Toolkit; no instale CUDA, cuDNN, ORT, Python ni OpenCV en el host.

Ejecute los comandos desde la raíz del repositorio con `server/config/server.toml` presente. Server y el instalador de modelos se ejecutan como root (`0:0`). Compose crea `server/.models` si falta y lo monta una sola vez con escritura en `/models`. El subdirectorio `addons` se crea al descargar un addon. No hacen falta exportaciones UID/GID ni preparar directorios o permisos manualmente. El arranque normal no descarga modelos; la prueba de vida está desactivada por defecto.

```bash
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml run --rm models install buffalo_l
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

Opcionalmente puede configurar la prueba de vida al instalar el modelo; sustituya el comando de instalación por:

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

Server no tiene que estar en ejecución. En una instalación nueva, el siguiente `up -d` activa la prueba de vida; si Server ya está ejecutándose, use `docker compose -f server/deploy/compose.cpu.yml restart server`. `up -d` por sí solo no recarga los ajustes guardados. Para CUDA use `compose.cuda12.yml`.

Para GPU use `compose.cuda12.yml` y el puerto `18098`. El instalador muestra la licencia antes de descargar; los modelos públicos de InsightFace son solo para investigación no comercial salvo licencia comercial independiente.

El Compose incluido desactiva la autenticación por defecto para evaluación aislada. Antes de exponer el servicio, defina `INSIGHTFACE_AUTH_ENABLED=true` y un `INSIGHTFACE_API_KEY` largo. Después compruebe el Panel, cree una Collection, registre una Person y busque con otra foto. Detenga con `docker compose ... down` sin `-v` para conservar el volumen.

## Addon opcional de prueba de vida

La prueba de vida está desactivada por defecto en `server/config/server.toml`: `inference.addons` y `addons.auto_download` son `[]`. Las configuraciones antiguas sin estas claves siguen desactivadas.

**Activar desde la línea de comandos, incluso antes del primer inicio de Server:**

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

`--enable-liveness` comprueba primero que la configuración existente pueda actualizarse. Instala y verifica el paquete base, los addons configurados para instalación y la prueba de vida, y después añade `liveness` a `inference.addons` y `addons.auto_download`, conservando otras entradas, comentarios y ajustes. Reutiliza cachés verificados y guarda la activación incluso si el modelo ya estaba en caché. Un fallo de descarga no modifica la configuración; un fallo al guardarla devuelve un error explícito y un código de salida distinto de cero. Los archivos válidos en caché pueden reutilizarse al reintentar.

Ambos servicios Compose montan todo el directorio existente `server/config` con escritura en `/etc/insightface`, con `create_host_path: false`. Así el instalador actualiza atómicamente la configuración del host sin que Server esté ejecutándose. Deben existir el directorio y `server.toml`.

Server no tiene que estar en ejecución. En una instalación nueva, el siguiente `up -d` activa la prueba de vida; si Server ya está ejecutándose, use `docker compose -f server/deploy/compose.cpu.yml restart server`. `up -d` por sí solo no recarga los ajustes guardados. Para CUDA use `compose.cuda12.yml`.

Sin `--enable-liveness`, `models install` mantiene su comportamiento y no escribe configuración; la prueba de vida sigue desactivada por defecto. `models addons install liveness` solo descarga y verifica el addon, sin activarlo. También puede usar **Sistema → Detección de vida** como se explica a continuación.

En **Sistema → Detección de vida**, seleccione **Descargar y activar tras reiniciar**. Tras verificar SHA-256, se añade `liveness` a ambas listas conservando las demás entradas, comentarios y opciones. Se reutiliza una copia ya verificada. **Reinicie el Server manualmente** para aplicar el cambio. Los errores permiten reintentar; una descarga fallida no activa la prueba.

Sistema distingue instalación verificada (`installed`), ejecución actual (`enabled`), configuración guardada para el próximo inicio (`configured_enabled`) y necesidad de reinicio (`restart_required`). Descargar o guardar no cambia la inferencia en curso. Para desactivar, guarde `inference.addons=[]` y `addons.auto_download=[]` en el mismo archivo y reinicie manualmente. La acción Web no modifica el ajuste de registro; su valor por defecto sigue siendo `liveness_on_registration=false`.

**Avanzado: configuración manual.** Estos ajustes son una alternativa al parámetro de activación o a la acción Web; instale el modelo antes de reiniciar.

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

### Instalación del modelo y arranque

`inference.addons` controla el uso y `addons.auto_download` la descarga adicional al instalar un paquete base. Con `["liveness"]` se instala el addon incluso con el paquete base en caché. No hay descargas al iniciar el Server. Instalador y Server leen el mismo archivo.

```bash
docker compose -f server/deploy/compose.cpu.yml run --rm models addons install liveness
docker compose -f server/deploy/compose.cpu.yml run --rm models addons verify liveness
docker compose -f server/deploy/compose.cpu.yml up -d --force-recreate
```

Un modelo activado ausente detiene el inicio con `addon_model_missing`; uno inválido produce `addon_model_invalid`. No se desactiva silenciosamente el addon.

### Montajes y permisos para descargas Web

Los archivos Compose incluidos montan todo `/models` con escritura; no se necesita un montaje separado de `/models/addons`. Server y el instalador usan root (`0:0`), y la instalación crea `addons` cuando hace falta. Todo el directorio existente `server/config` se monta con escritura en `/etc/insightface` en ambos servicios para que la acción Web y `--enable-liveness` guarden `server.toml` de forma atómica. Esta configuración no requiere exportaciones UID/GID, `chgrp` ni `chmod`. Los archivos o montajes deliberadamente de solo lectura siguen impidiendo la gestión Web; revise los permisos de los montajes personalizados.

El montaje de modelos de cada servicio usa `create_host_path: true`. Los servicios conservan las capabilities predeterminadas de Docker; no se aplica `cap_drop: [ALL]`. El resto del sistema de archivos sigue siendo de solo lectura y se mantiene `no-new-privileges`. Root puede modificar archivos en los montajes con escritura; las nuevas descargas pueden pertenecer a root en el host.

En despliegues personalizados use las rutas reales; para CUDA use `compose.cuda12.yml`. Los montajes antiguos de solo lectura siguen funcionando con la prueba de vida desactivada. La acción Web explica por qué no está disponible; también puede instalar por CLI y editar la configuración manualmente. Tras guardar desde la Web, aplique los cambios con `docker compose -f server/deploy/compose.cpu.yml restart server`. Cambiar montajes o variables de proxy requiere recrear el contenedor.

Si necesita proxy, configure `HTTP_PROXY`, `HTTPS_PROXY` y `NO_PROXY` antes de crear el contenedor; Compose los transmite al Server y a la herramienta de modelos. Use una dirección LAN accesible desde el contenedor: su `127.0.0.1` no es el Mac. La acción usa la autenticación API Key existente; si está desactivada, también puede ejecutarla quien tenga acceso a la API. Solo descarga el modelo de vida publicado y fijado; no acepta URLs arbitrarias ni cambia el paquete de modelo base.

### Resultados de la prueba de vida

| Resultado | `status` | `is_live` | `live_score` |
| --- | --- | --- | --- |
| Prueba superada | `ok` | `true` | `[0, 1]` |
| No vivo | `ok` | `false` | `[0, 1]` |
| Entrada rechazada | `input_rejected` | `null` | `null` |

Solo una superficie insuficiente de la imagen original alrededor del rostro alineado produce `input_rejected`. Este resultado añade `liveness.reason`, una explicación para el usuario; los resultados de rostro vivo o falsificación omiten `reason`. FaceAnalysis y la API devuelven siempre este texto en inglés; solo la interfaz Web traduce su presentación. Use `status` e `is_live` en la lógica del programa, sin interpretar el texto de `reason`. Los resultados antiguos guardados pueden carecer de `reason`; el cliente puede mostrar entonces un aviso genérico de entrada rechazada.

```json
{
  "status": "input_rejected",
  "is_live": null,
  "live_score": null,
  "reason": "Insufficient image area around the face for liveness detection. Move the face toward the center, step back from the camera, or use a less tightly cropped image."
}
```

`normal` reconoce solo rostros que superan la prueba; `observe` registra el resultado y continúa el reconocimiento. Si no se evalúa, se omite `liveness`. Los tres campos principales son `status`, `is_live` y `live_score`: aprobado/fake usa `status: ok`, booleano y puntuación; una entrada rechazada usa `status: input_rejected` y dos valores `null`.

Detect devuelve HTTP 200 incluso con resultados negativos. En `normal`, embeddings, comparación y búsqueda devuelven HTTP 422 `liveness_fake` o `liveness_input_rejected` con `error.details.liveness`; comparación añade `details.side`. Un fallo de inferencia devuelve HTTP 503 `liveness_unavailable`. Los fallos de ejecución interrumpen la operación tanto en `normal` como en `observe`; no se convierten en `input_rejected`.

El registro de personas y la adición de FaceSamples omiten la prueba de vida por defecto: `[inference].liveness_on_registration=false` no ejecuta el modelo y omite `liveness` en las muestras nuevas. Con `true` y el addon habilitado se aplica `normal`/`observe`; los rechazos incluyen `reason` y `liveness`. La revisión de calidad según `review_mode` y la validación de embeddings externos siguen activas. `review_mode=off` y `external_trusted` no evitan una prueba de registro habilitada. Las peticiones no pueden modificar esta configuración de inicio. Los resultados previamente guardados permanecen disponibles.

RTSP distingue `liveness_blocked` de `unknown` y usa el contador `liveness_blocked_faces`. Los rostros bloqueados no generan eventos de entrada de personas/desconocidos y reinician la confirmación. Los fallos de inferencia borran las identidades mostradas anteriormente.

`liveness_compare_scope` selecciona `both` (predeterminado), `source` o `target` para `/v1/compare`. Se aprueba con `live_score >= liveness_threshold`.

El modelo se guarda en `server/.models/addons/liveness.onnx` en el host y en `/models/addons/liveness.onnx` en el contenedor. `addons` en `/v1/models` y `/v1/system` muestra los addons activos.

[Contrato completo de la API](api.es.md#addon-opcional-de-prueba-de-vida).

## 1. Acceso y estado

Abra `http://SERVIDOR:18097/` para CPU o `http://SERVIDOR:18098/` para CUDA 12. Si hay autenticación, use **Configurar clave API**, pegue la clave del operador y aplíquela a la pestaña. Solo permanece en memoria y se elimina al recargar o cerrar.

Compruebe en **Panel** o **Sistema** que servicio, base de datos, modelos y Provider estén listos. CUDA debe mostrar `CUDAExecutionProvider` y nunca vuelve silenciosamente a CPU.

El Panel muestra siempre la prueba de vida activada o desactivada debajo del modelo. Sistema distingue instalación, estado actual y reinicio pendiente.

## 2. Crear una Collection

En **Colecciones** → **Nueva colección**, indique un ID estable, nombre, umbral
coseno (`0.4` inicialmente), perfil disponible, capacidad y máximo de
FaceSamples por persona. Guardar como JPEG un `bounding-box crop` redimensionado
a 112×112 está desactivado por defecto; no es la entrada alineada de
reconocimiento.

La Collection queda fijada al ID, digest, dimensión y preprocesamiento del modelo. Tras cambiar el modelo, una colección antigua sigue visible, pero su registro y búsqueda se rechazan si el contrato no coincide.

El perfil de detección copia los valores del sistema al crear la Collection y después permite cambiar tamaños de entrada, umbrales de detección/NMS y estrategia de un rostro. `largest` prioriza el área; `center_largest` maximiza `área - 2,0 × distancia en píxeles al cuadrado entre el centro del cuadro y el de la imagen`. La confianza de detección no participa en esta puntuación.

## 3. Registrar una Person

En **Personas**, seleccione la Collection y **Registrar persona**. Puede indicar ID, nombre, ID externo, metadata JSON y una o varias imágenes JPEG, PNG, WebP o BMP.

- `off`: usa la estrategia de un rostro de la Collection y permite varios rostros.
- `standard`: exige un rostro utilizable y valida tamaño, detección, nitidez, iluminación y pose.
- `strict`: además exige que la mejor similitud interna sea mayor que la mejor similitud con otra persona.

El lote permite éxito parcial y explica cada rechazo. No se guardan originales. `external_trusted` acepta un embedding normalizado L2; la imagen sigue siendo obligatoria para detección y calidad, pero no se vuelve a extraer el vector.

La creación de Person y la adición de FaceSamples omiten la prueba de vida por defecto (`liveness_on_registration=false`). Si se activa, `normal` rechaza fake/entradas no aptas; `observe` conserva el resultado y continúa. La revisión de calidad sigue el `review_mode` seleccionado. Los rechazos muestran el `reason` real y el resultado de vida por separado.

## 4. Detectar, comparar y buscar

**Detectar** muestra cajas, cinco puntos, detección y calidad; sin rostros devuelve una lista vacía correcta. **Comparar** usa el perfil del sistema o de una Collection para elegir un rostro por imagen y devuelve `similarity` coseno, `threshold` y `matched`. La similitud no es probabilidad.

En **Buscar**, seleccione Collection e imagen. La puntuación de una persona es la mayor similitud entre sus FaceSamples. Los resultados se ordenan de mayor a menor; sin coincidencia es una lista vacía. Cada muestra se confirma primero en SQLite y se añade al índice antes de responder. Al reiniciar, el índice se reconstruye desde SQLite.

Cada rostro evaluado incluye `liveness.status`, `liveness.is_live` y `liveness.live_score`. Fake e `input_rejected` también devuelven HTTP 200, sin extraer características de reconocimiento. `input_rejected` indica una superficie de imagen insuficiente alrededor del rostro; `liveness.reason` explica cómo ajustar la imagen. Si falta `liveness`, no se evaluó.

`liveness_compare_scope` (`both`, `source`, `target`) elige los lados evaluados antes del reconocimiento. En `normal`, un rechazo devuelve HTTP 422 `liveness_fake` / `liveness_input_rejected`, `error.details.liveness` y `error.details.side`, sin similitud. `observe` continúa y adjunta el resultado a los rostros evaluados.

Con prueba de vida en `normal`, fake/consulta no apta devuelve HTTP 422 `liveness_fake` / `liveness_input_rejected` y `error.details.liveness`; no se ejecuta la búsqueda. No equivale a una lista vacía de coincidencias correcta. `observe` continúa y devuelve el resultado en el rostro consultado.

## 5. Monitorización de cámara RTSP

En **Monitorización de cámaras**, cree un Monitor persistente y configure fuente RTSP, Collection, frecuencia, umbral opcional y política de eventos. La vista previa está desactivada por defecto; reconocimiento y eventos continúan sin ella. Al activarla, la Web UI dibuja cajas verdes para personas registradas y naranjas para rostros desconocidos usando `/state` sobre imágenes crudas.

El Monitor funciona independientemente del navegador y las tareas activas se restauran tras reiniciar el servidor. La configuración vive en SQLite y las credenciales RTSP cifradas en `/data`; no se guardan fotogramas ni eventos. Los eventos solo permanecen en un búfer de memoria limitado. El decodificador conserva el último fotograma y omite los obsoletos en lugar de acumularlos.

Con prueba de vida en `normal`, los rostros bloqueados tienen `status: liveness_blocked` y un resultado separado. Cuentan en `liveness_blocked_faces`, no en `unknown_faces`, y no generan eventos de entrada. `observe` continúa el reconocimiento. La interfaz distingue entrada rechazada y fake.

## 6. Datos y seguridad

Conserve y respalde `/data`, el directorio de modelos y la configuración. El despliegue incluido usa root con las capabilities predeterminadas de Docker y `/models` con escritura; el servicio puede modificar los modelos, configuración y datos montados. El resto del sistema de archivos del contenedor sigue siendo de solo lectura y se conserva `no-new-privileges`; no se necesita `privileged`. Restrinja el acceso a Docker y al host, y no monte directorios del host ajenos al servicio. Antes de operaciones masivas, copie SQLite y los recortes juntos. Las claves se guardan como hash; iniciar el mismo volumen con otro `INSIGHTFACE_API_KEY` rota la clave activa. No registre imágenes, embeddings ni claves.

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
`buffalo_s`, `buffalo_sc`, `antelopev2`, `raccoon_s` y `raccoon_l`. La
instalación crea `manifest.json` y
`MODEL.LICENSE` firmada. Sin `--accept-license`, la herramienta muestra los
términos y pide confirmación antes de descargar en una terminal interactiva.
Los comandos no interactivos requieren esa opción y, si falta, terminan sin
descargar. Los modelos públicos preentrenados de
InsightFace son solo para investigación no comercial salvo licencia comercial
independiente.

Se admiten `raccoon_s` y `raccoon_l`. El Server instala solo detección y reconocimiento de cada paquete; no carga el verificador Raccoon. El modelo se identifica por nombre, sin un número de versión independiente. La acción Web de vida no cambia el modelo base. Si cambia el modelo de reconocimiento, use una Collection compatible; los embeddings anteriores no se convierten en características del nuevo modelo.

## 8. Configuración de inicio y búsqueda

```toml
[inference]
addons = []
liveness_mode = "normal"
liveness_threshold = 0.8
liveness_compare_scope = "both"
liveness_on_registration = false

[addons]
auto_download = []
```

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

Puede compilar directamente desde un directorio local con el código fuente
completo, incluso con cambios sin confirmar o sin un directorio `.git`. No es
necesario hacer commits ni subirlos con Git antes de compilar.

```bash
make -C server build-cpu
make -C server build-cuda12
```

Cuando las pruebas pasen, publique la misma imagen que se probó. Confirmar
después el mismo código u organizar sus commits no requiere volver a compilar.
Los cambios en archivos incluidos en la imagen, como código, recursos del
frontend o ayuda de usuario incorporada, requieren una nueva compilación y
validación.

Use `--pull never` con Compose para usar la imagen local. Los tags inmutables
son `0.3.1-cpu` y `0.3.1-cuda12`; `cpu` y `cuda12` apuntan a la última estable
y no existe `latest`. Antes de actualizar, pare las escrituras y haga una copia
SQLite segura de `/data` y los crops. No use `docker compose down -v`: elimina
el volumen de datos.

### Actualizar a 0.3.1

0.3.1 simplifica el despliegue: Server y el instalador usan root, Compose crea el directorio de modelos cuando hace falta y un único `/models` con escritura sustituye al montaje separado de addons.

Desde 0.3.0 se admiten `raccoon_s`, `raccoon_l`, sus manifiestos, prueba de vida opcional, instalación Web de addons e imágenes BMP. Server usa la detección y el reconocimiento de Raccoon; no carga el verificador. Estas funciones y los contratos de respuesta API no cambian en 0.3.1.

**1.** Actualice el código del Server y los archivos Compose a la versión 0.3.1
conservando sus ajustes de `server/config/server.toml` y las personalizaciones
del despliegue. Mantenga la ruta de modelos, el nombre del volumen `/data`,
el almacenamiento de recortes, los puertos y los ajustes de clave API. En
archivos Compose personalizados, actualice las imágenes de ambos servicios,
`server` y `models`, a `0.3.1-cpu` o `0.3.1-cuda12` según corresponda. Aplique
a los comandos siguientes los mismos archivos Compose, personalizaciones y
nombre de proyecto que utiliza habitualmente.

Antes de iniciar, actualice también sus overrides de Compose, además de las etiquetas: ambos servicios deben usar `user: "0:0"`, sin `cap_drop: [ALL]` ni los antiguos ajustes UID/GID o `group_add`. Use en ambos servicios un único bind mount con escritura en `/models` y `create_host_path: true`; elimine el montaje separado de `/models/addons` y `x-addons-path`. Conserve la ruta real de modelos y sus archivos, incluidos los addons. Los montajes de configuración mantienen `create_host_path: false`, por lo que deben seguir existiendo `server/config` y `server.toml`. Conserve el volumen de datos y haga una copia de datos, modelos y configuración antes del cambio. Quite de sus overrides los valores antiguos de usuario y montajes; cambiar solo la imagen no aplica estas modificaciones.

Mantenga el sistema de archivos del contenedor de solo lectura y `no-new-privileges`. Ambos servicios necesitan todo el directorio de configuración con escritura. Sustituya el antiguo montaje de un único archivo de solo lectura del instalador por el montaje del directorio. El despliegue estándar no requiere cambiar recursivamente los permisos existentes ni preparar un directorio de addons.

**2.** Descargue las nuevas imágenes y vuelva a crear el contenedor del Server.
Desde la raíz del repositorio, elija los comandos de su despliegue actual:

CPU:

```bash
docker compose -f server/deploy/compose.cpu.yml pull server models
docker compose -f server/deploy/compose.cpu.yml up -d --no-build --force-recreate server
curl -fsS http://127.0.0.1:18097/v1/health
```

CUDA:

```bash
docker compose -f server/deploy/compose.cuda12.yml pull server models
docker compose -f server/deploy/compose.cuda12.yml up -d --no-build --force-recreate server
curl -fsS http://127.0.0.1:18098/v1/health
```

Si compila localmente, genere primero las imágenes 0.3.1 y use
`up -d --no-build --pull never --force-recreate server` en lugar de descargar
las imágenes. `docker compose restart` por sí solo no cambia a una imagen
nueva ni aplica cambios en los montajes.

**3.** El inicio aplica automáticamente las migraciones de la base de datos.
Espere a que `/v1/health` indique `ready` y la versión `0.3.1`, y compruebe
en **Sistema** el modelo y el proveedor de ejecución esperados. Confirme que
las Collections y personas existentes siguen presentes y pruebe una
búsqueda conocida. Si conserva el mismo modelo y contrato de embedding,
se mantienen las muestras, los embeddings y los identificadores de contrato
de las Collections; no es necesario volver a registrar a las personas.

**La detección de vida es opcional tras actualizar.** Tanto la configuración
distribuida como las configuraciones antiguas sin claves de addon la mantienen
desactivada, por lo que actualizar el Server no exige descargar el modelo de
vida. El Server nunca descarga modelos al iniciar. Para activarla, siga la
[configuración de detección de vida](#addon-opcional-de-prueba-de-vida): prepare
los [montajes y permisos Web](#montajes-y-permisos-para-descargas-web), seleccione
**Sistema → Detección de vida → Descargar y activar tras reiniciar**, espere a
que terminen correctamente la instalación y el guardado de la configuración,
y reinicie el Server manualmente. Los valores predeterminados son `normal`,
umbral `0.8` y `liveness_on_registration=false`. El modelo permanece en
`<models_dir>/addons/liveness.onnx`.

**Adoptar Raccoon es un cambio de modelo independiente.** Actualizar el Server
conserva el paquete de modelo actual. Para utilizar `raccoon_s` o `raccoon_l`,
instale el paquete elegido en un directorio de modelos separado siguiendo las
[instrucciones de instalación](#7-modelos-y-licencias), y configure un despliegue
para usarlo. Las Collections deben coincidir con el contrato de embedding del
nuevo modelo: cree Collections compatibles y vuelva a registrar a las personas,
o realice una migración de datos independiente. La Web UI no cambia el paquete
de modelo base.

**Compatibilidad de API y SDK desde 0.3.0:** Los resultados de modelos, Collections y
FaceSamples ya no incluyen `model_version`. La identidad del modelo usa
`model_id` y la compatibilidad de Collections usa `embedding_contract_id`.
Adapte los clientes que requieran el campo eliminado y utilice el SDK `0.3.1`
al actualizar el cliente Python distribuido. Si se evalúa la detección de vida,
`liveness` contiene los campos principales `status`, `is_live` y `live_score`,
con `reason` solo para `input_rejected`; si no se evalúa,
se omite. Consulte las [reglas de resultados y errores de vida](#resultados-de-la-prueba-de-vida)
antes de activarla para las solicitudes de reconocimiento.

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
