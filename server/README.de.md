# InsightFace Server

**Sprachen:** [English](README.md) · [中文](README.zh-CN.md) · [日本語](README.ja.md) · Deutsch · [Español](README.es.md) · [Français](README.fr.md) · [Русский](README.ru.md) · [Português](README.pt.md) · [한국어](README.ko.md)

> **Eine GPU. 50M+ Gesichtsvektoren. Rasante Suche mit INT8-Feature-Quantisierung ohne wesentlichen Genauigkeitsverlust.**

**Ein selbst gehosteter Gesichtserkennungsserver mit Web UI, verständlicher
REST API, SQLite und lokaler CPU- oder NVIDIA-GPU-Inferenz in einem Container.**

```text
Bild hochladen -> erkennen, Lebenderkennung durchführen, vergleichen, registrieren oder suchen
```

> **Modelllizenz:** Öffentliche vortrainierte InsightFace-Modelle sind in der
> Regel nur für nichtkommerzielle Forschung freigegeben. Kommerzielle Nutzung
> erfordert eine separate Genehmigung von
> [InsightFace](https://www.insightface.ai).

InsightFace Server ist für übliche Gesichtserkennungsabläufe eine einfachere,
datenschutzorientierte Alternative zu AWS Rekognition auf eigener
Infrastruktur. Bilder, Embeddings, Modelle und Indizes können im eigenen Netz
bleiben. Er ist **kein** AWS-kompatibler Ersatz und implementiert weder SigV4,
IAM, Region noch AWS-Ressourcensemantik.

Aktuelle Version: **0.3.1**, Linux x86_64.

| Laufzeit | Image |
| --- | --- |
| CPU | `ghcr.io/deepinsight/insightface-server:0.3.1-cpu` |
| NVIDIA GPU | `ghcr.io/deepinsight/insightface-server:0.3.1-cuda12` |

Die beweglichen Tags `cpu` und `cuda12` bezeichnen die neueste stabile Version
der jeweiligen Laufzeitfamilie. Ein mehrdeutiges `latest` gibt es nicht. Siehe
[Maintainer Guide — English](docs/maintainer-guide.md) für die Release-Regeln.

**Neuerungen und Upgrade auf 0.3.1:** Server und Modellinstaller laufen als root; ein einziges schreibbares Modell-Mount ersetzt das separate Addon-Mount. Compose erstellt das Modellverzeichnis bei Bedarf, ohne manuelle UID/GID- oder Rechtekonfiguration. Aktualisieren Sie auch Compose-Dateien und eigene Overrides; behalten Sie die vorhandenen Modell-, Konfigurations- und Datenpfade. Siehe [Upgrade-Anleitung](docs/user-guide.de.md#upgrade-auf-031).

Seit 0.3.0 werden `raccoon_s`, `raccoon_l`, ihre Modell-Manifeste, optionale Liveness mit Web-Installation und BMP-Eingaben unterstützt. Seitdem enthalten API- und SDK-Ergebnisse kein `model_version` mehr. Liveness bleibt bis zur Aktivierung ausgeschaltet.

![InsightFace Server Dashboard auf Englisch](docs/images/customer/dashboard-en.jpg)

## Funktionsübersicht

- SCRFD-Erkennung, fünf Landmarken, Ausrichtung, ArcFace-Embeddings, L2
  normalization, rohe cosine similarity und exakte 1:N-Personensuche.
- Mehrere Erkennungsauflösungen mit gemeinsamem NMS sowie `largest` und
  `center_largest` zur Ein-Gesicht-Auswahl.
- `Collection -> Person -> FaceSample`, modellgebundene Collections,
  Mehrbildregistrierung mit Teilerfolg, metadata und klaren Ablehnungsgründen.
- Registrierung mit `review_mode` `off`, `standard` oder `strict`; optional
  vorab berechnete `external_trusted` Embeddings.
- Exakte GPU-Suche mit FP32-, FP16-, BF16- und INT8-Vektorspeicherung.
- Mehrsprachige Web UI für Dashboard, Collections, Personen, Detect, Compare,
  Search, RTSP-Monitoring, Systemdiagnose und Hilfe.
- 31 snake_case-REST-Operationen unter `/v1`, einschließlich geschütztem
  `/v1/embeddings`, plus schlankem typisierten Python SDK.
- Serverseitige RTSP Monitors mit begrenzten In-Memory-Ereignissen, mehreren
  Clients und optionalem `preview.mjpeg`; das Schließen des Browsers beendet
  die Überwachung nicht.
- SQLite als dauerhafte Quelle, rekonstruierbare exakte In-Memory-Indizes,
  ein einziges schreibbares Modell-Mount und ein schreibbares Konfigurationsverzeichnis, persistentes `/data`, Migrationen,
  Healthchecks und strikte CUDA-Prüfung ohne stillen CPU-Fallback.
- JPEG-, PNG-, WebP- und BMP-Eingaben; Originaluploads werden standardmäßig nicht
  gespeichert.

Liveness ist standardmäßig deaktiviert: `inference.addons = []` und `addons.auto_download = []` in `server/config/server.toml`. **System → Lebenderkennung** lädt das Modell herunter, prüft es und speichert die Aktivierung für den nächsten manuellen Neustart. Ein geprüfter Cache wird wiederverwendet. Beim Start wird nichts heruntergeladen; fehlt ein aktiviertes Modell, stoppt der Start mit Installationshinweis. Ausgewertete Gesichter liefern die Kernfelder `status`, `is_live` und `live_score`; nur `input_rejected` ergänzt eine verständliche Erklärung in `reason`. Der Standardmodus ist `normal`; bei der Registrierung wird Liveness standardmäßig übersprungen (`liveness_on_registration = false`). Siehe [Konfiguration, Web-Berechtigungen und Upgrade](docs/user-guide.de.md#optionales-liveness-addon).

### GPU-Suchleistung auf der RTX 5090

Auf einer NVIDIA GeForce RTX 5090 (32.607 MiB) speicherte der native exakte
CUDA-Flat-Index mit INT8 bis zu **58,9M 512-dimensionale Bildvektoren**.

| GPU-Datentyp | Maximale Bildvektoren | 10M Top-5 p50 | 10M serielle QPS |
| --- | ---: | ---: | ---: |
| FP32 | 15,8M | 12,84 ms | 77,85 |
| FP16 | 30,7M | 6,83 ms | 146,32 |
| BF16 | 30,7M | 6,83 ms | 146,33 |
| INT8 | **58,9M** | **3,84 ms** | **260,81** |

INT8 erreichte gegenüber FP32 die 3,73-fache gemessene Kapazität und den
3,35-fachen 10M-Top-5-Durchsatz. Dies sind reine GPU-Messungen auf derselben
RTX 5090 mit Driver 580.105.08 und CUDA 12.9. Die Kapazität ist die isolierte
Grenze des nativen Index ohne geladene ONNX-Modelle oder Serverlast. Der
Geschwindigkeitstest nutzt exakt 10M Bildvektoren, eine vollständige exakte
GPU-residente Top-5-Suche, eine Anfrage gleichzeitig, 10 Warm-ups und 100
Messungen. Innerhalb der jeweiligen Speicherung ist die Suche exakt;
Quantisierung kann dennoch Scores gegenüber FP32 verändern. Produktion
benötigt VRAM-Reserve für Modelle, Requests, Parallelität, Index-Neuaufbau und
den Allocator.

### Multiethnische ICCV21-MFR-MR-ALL-Genauigkeit

Wir haben die nativen Suchprofile auf dem multiethnischen MR-Testdatensatz von
[ICCV21-MFR](../challenges/iccv21-mfr/) mit dem MR-ALL-All-Pairs-1:1-Protokoll
bei FAR `1e-6` evaluiert. Alle Profile verwenden dieselben einmalig über die
Server API extrahierten, L2-normalisierten 512-dimensionalen
`buffalo_l`-Embeddings; nur Vektorspeicherung und Rechendarstellung der Suche
ändern sich.

| Suchprofil | MR-ALL bei FAR 1e-6 | Cosine-Schwelle | Differenz zu FP32 |
| --- | ---: | ---: | ---: |
| FP32 | 91,249107 % | 0,407787 | — |
| FP16 | 91,249197 % | 0,407787 | +0,000090 Prozentpunkte |
| BF16 | 91,248502 % | 0,407787 | -0,000605 Prozentpunkte |
| **INT8** | **91,248005 %** | **0,407739** | **-0,001102 Prozentpunkte** |

**INT8 zeigt in diesem Benchmark keinen wesentlichen Genauigkeitsverlust:**
Mit der challenge-üblichen Darstellung auf zwei Dezimalstellen erreichen FP32
und INT8 jeweils **91,25 % MR-ALL**; die ungerundete Differenz beträgt nur
0,0011 Prozentpunkte. Gleichzeitig bleiben die oben gezeigten Vorteile von
3,73-facher gemessener Kapazität und 3,35-fachem 10M-Top-5-Durchsatz erhalten.
Verglichen wird die Präzision von Vektorspeicherung und Suche, nicht die
INT8-Modellinferenz.

![Collection-Verwaltung auf Englisch](docs/images/customer/collections-en.jpg)

![RTSP Monitor auf Englisch; private Adresse unkenntlich](docs/images/customer/monitoring-en.jpg)

## Schnellstart

Voraussetzungen:

- Linux x86_64 mit Docker Engine und Docker Compose;
- für CUDA eine unterstützte NVIDIA GPU, NVIDIA Driver und NVIDIA Container
  Toolkit.

Auf dem Host sind Python, OpenCV, ONNX Runtime, CUDA Toolkit und cuDNN nicht
erforderlich. Öffentliche Images enthalten keine Modelle, Kundendaten, API Keys
oder Produktionskonfiguration.

Installieren Sie in einem vollständigen InsightFace-Checkout ein Modell nach
`server/.models`:

Führen Sie die Befehle im Repository-Stamm aus; `server/config/server.toml` muss vorhanden sein. Server und Modellinstaller laufen als root (`0:0`). Compose erstellt `server/.models`, falls es fehlt, und bindet es einmal schreibbar unter `/models` ein. Das Unterverzeichnis `addons` entsteht erst bei einem Addon-Download. UID/GID-Exporte sowie manuelle Verzeichnis- oder Rechtevorbereitung sind nicht erforderlich. Der normale Serverstart lädt keine Modelle herunter; Liveness ist standardmäßig ausgeschaltet.

```bash
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license
```

Optional können Sie Liveness bei der Modellinstallation konfigurieren; verwenden Sie dafür stattdessen:

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

Server muss dafür nicht laufen. Bei einer Neuinstallation aktiviert das nächste `up -d` Liveness; bei einem bereits laufenden Server ist `docker compose -f server/deploy/compose.cpu.yml restart server` erforderlich. `up -d` allein lädt gespeicherte Einstellungen nicht neu. Für CUDA verwenden Sie `compose.cuda12.yml`.

Das Modellwerkzeug unterstützt alle sieben Pakete: `buffalo_l`, `buffalo_m`,
`buffalo_s`, `buffalo_sc`, `antelopev2`, `raccoon_s` und `raccoon_l`. Es
schreibt `manifest.json` und die signierte `MODEL.LICENSE`;
`models verify` prüft das installierte Paket. Die Modellbedingungen sind von
der Server-Quelllizenz getrennt.

CPU starten:

```bash
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

Stattdessen CUDA 12 starten:

```bash
docker compose -f server/deploy/compose.cuda12.yml pull
docker compose -f server/deploy/compose.cuda12.yml \
  run --rm models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cuda12.yml up -d
curl -fsS http://127.0.0.1:18098/v1/health
```

Öffnen Sie für CPU `http://SERVER:18097/` oder für CUDA
`http://SERVER:18098/`. Erstellen Sie eine Collection, registrieren Sie eine
Person mit einem oder mehreren Bildern und suchen Sie mit einem anderen Bild.
`docker compose ... down` ohne `-v` erhält das Datenvolume.

Die mitgelieferten Compose-Dateien deaktivieren Authentifizierung für isolierte
Tests standardmäßig. Vor der Freigabe für andere Benutzer oder Netze:

```bash
export INSIGHTFACE_AUTH_ENABLED=true
export INSIGHTFACE_API_KEY='durch-ein-langes-zufaelliges-geheimnis-ersetzen'
docker compose -f server/deploy/compose.cpu.yml up -d
```

Den vollständigen ersten Ablauf erklärt das
[Benutzerhandbuch](docs/user-guide.de.md).

## Aus Quellcode bauen

Sie können direkt aus einem vollständigen lokalen Quellcodeverzeichnis bauen,
auch mit nicht committeten Änderungen oder ohne `.git`-Verzeichnis. Git-Commits
und Pushes sind keine Voraussetzung für den Build.

Die Dockerfiles kopieren `server/` und ausgewählte Inferenzmodule aus
`python-package/insightface/`; daher ist das vollständige Quellcodeverzeichnis der
Build-Kontext.

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

`--pull never` erzwingt das lokal gebaute Image. Der Build lädt weiterhin
festgeschriebene Basis-Images und Abhängigkeiten; die Modellinstallation lädt
das separat akzeptierte Modellpaket.

## Kernverhalten

- Similarity ist der rohe Cosinuswert, keine Wahrscheinlichkeit. Thresholds
  liegen in `0.0..1.0`, Standard ist `0.4`.
- Eine Collection bindet Modell und embedding contract. Bei Abweichung bleibt
  sie sichtbar, Registrierung/Suche liefert `collection_model_mismatch`.
- Das Detection Profile beim Start wird in neue Collections kopiert; danach
  kann ihr Profil unabhängig für folgende Requests geändert werden.
- Optionale Gesichtsspeicherung enthält einen auf 112x112 skalierten
  bounding-box JPEG crop, nicht das Original und nicht den ausgerichteten
  Erkennungseingang; standardmäßig aus.
- SQLite-Commits sind maßgeblich. Indexänderungen sind vor einer erfolgreichen
  Registrierungs-/Löschantwort abgeschlossen; nach Neustart erfolgt der Aufbau
  aus SQLite.
- Responses enthalten `x-request-id`; Listen verwenden undurchsichtige
  signierte cursor.

Exakte Felder, Defaults, Lebenszyklen und Fehlerverhalten stehen ausschließlich
in den unten verlinkten Detaildokumenten.

## API und SDK

Hauptgruppen:

- System: `/v1/health`, `/v1/system`, `/v1/models`;
- zustandslose Gesichtsoperationen: `/v1/detect`, `/v1/compare`,
  `/v1/embeddings`;
- CRUD für Collection, Person und FaceSample;
- Personensuche in einer Collection;
- Konfiguration, Status, Ereignisse und Vorschau von RTSP Monitors.

Alle Parameter, Responses, Fehler und Beispiele enthält der
[REST-API-Leitfaden](docs/api.de.md). Interaktives OpenAPI bleibt unter `/docs`.

```python
from insightface_server import Client

with Client("http://localhost:18097", api_key=None) as client:
    faces = client.detect("photo.jpg")
    matches = client.search("employees", "unknown.jpg", limit=5)
```

Installation, Eingaben, Methoden und vollständige SDK-Abläufe stehen im
[Benutzerhandbuch](docs/user-guide.de.md).

## Sicherheit

Das mitgelieferte Deployment läuft als root mit den Docker-Standard-Capabilities und kann in Modell-, Konfigurations- und Daten-Mounts schreiben. Das übrige Container-Dateisystem bleibt schreibgeschützt; `no-new-privileges` bleibt aktiv. Beschränken Sie den Hostzugriff und die eingebundenen Verzeichnisse.

Gesichtsbilder und Embeddings sind biometrische Daten. Aktivieren Sie bei
Netzbetrieb die Authentifizierung, terminieren Sie HTTPS an einem vertrauenswürdigen
Reverse Proxy, beschränken Sie Docker- und Volume-Zugriff, lassen Sie breites
CORS aus und definieren Sie Backup, Aufbewahrung, Löschung, Einwilligung und
Incident Response. Bilder, Embeddings, RTSP-Zugangsdaten und API Keys gehören
nicht in Logs.

Der Server enthält kein TLS, Benutzerkonten, RBAC, Cloud-IAM oder
Rechtskonformitätsmodul. Betrieb und Sicherheit stehen im
[Benutzerhandbuch](docs/user-guide.de.md).

## Umfang der ersten Phase

Nicht enthalten sind AWS-/CompreFace-Kompatibilität, CUDA 11, Jetson, ARM64,
Windows Container, TensorRT, Kubernetes, verteilte Worker, persistente
Monitor-Ereignisse oder Aufnahme/NVR sowie Deepfake- oder
demografische Analysen.

## Dokumentation

- [Benutzerhandbuch](docs/user-guide.de.md) — Installation, Konfiguration,
  Modelle, Web UI, SDK, GPU, Sicherheit, Backup und Fehlerbehebung.
- [REST-API-Leitfaden](docs/api.de.md) — alle öffentlichen Endpunkte, Felder,
  Verhalten, Ergebnisse, Fehler, Pagination und Beispiele.
- [Maintainer Guide — English](docs/maintainer-guide.md) — Architektur,
  Suchinternes, Tests, Beiträge und Container-Releases.

GitHub und die Web-UI-Hilfe verwenden dieselben lokalisierten Markdown-Dateien;
nur die Darstellung unterscheidet sich.

## Lizenz

Der einzige Einstieg für Lizenzinformationen ist
[LICENSING.md](LICENSING.md). Server-Quellcode und Python SDK stehen unter MIT
License; diese Erklärung gilt nicht für Modelldateien, Modellgewichte,
Datensätze oder Drittkomponenten. Öffentliche InsightFace-Modelle sind ohne
separate Genehmigung in der Regel nur für nichtkommerzielle Forschung bestimmt.
Kommerzielle Lizenzierung: <https://www.insightface.ai>.
