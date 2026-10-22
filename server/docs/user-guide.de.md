# InsightFace Server Benutzerhandbuch

**Sprachen:** [English](user-guide.md) · [中文](user-guide.zh-CN.md) · [日本語](user-guide.ja.md) · Deutsch · [Español](user-guide.es.md) · [Français](user-guide.fr.md) · [Русский](user-guide.ru.md) · [Português](user-guide.pt.md) · [한국어](user-guide.ko.md)

Dieses Handbuch führt Erstanwender vom leeren Checkout bis zur ersten erfolgreichen Personensuche. Dieselben Funktionen stehen über Web UI, `/v1` und Python SDK bereit. Alle HTTP-Felder und Ergebnisse beschreibt der [API-Leitfaden](api.de.md).

Modelle werden durch `model_id` identifiziert; Antworten enthalten kein separates `model_version`.

Beim Server-Upgrade mit demselben Erkennungsmodell und Merkmalsvertrag bleiben `embedding_contract_id`, Samples und Embeddings vorhandener Collections erhalten. Ein Wechsel des Erkennungsmodells ist eine separate Migration; bei abweichendem Vertrag liefern Registrierung und Suche `collection_model_mismatch`.

Für Liveness siehe [Konfiguration, Modellinstallation und Ergebnisse](#optionales-liveness-addon). Die einzelnen Arbeitsschritte erklären zusätzlich die Auswirkungen.

## Vom Start bis zur ersten Suche

CPU benötigt Linux x86_64, Docker Engine und Docker Compose. CUDA benötigt zusätzlich einen kompatiblen NVIDIA-Treiber und NVIDIA Container Toolkit; CUDA, cuDNN, ORT, Python und OpenCV müssen nicht auf dem Host installiert sein.

Führen Sie die Befehle im Repository-Stamm aus; `server/config/server.toml` muss vorhanden sein. Server und Modellinstaller laufen als root (`0:0`). Compose erstellt `server/.models`, falls es fehlt, und bindet es einmal schreibbar unter `/models` ein. Das Unterverzeichnis `addons` entsteht erst bei einem Addon-Download. UID/GID-Exporte sowie manuelle Verzeichnis- oder Rechtevorbereitung sind nicht erforderlich. Der normale Serverstart lädt keine Modelle herunter; Liveness ist standardmäßig ausgeschaltet.

```bash
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml run --rm models install buffalo_l
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

Optional können Sie Liveness bei der Modellinstallation konfigurieren; verwenden Sie dafür stattdessen:

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

Server muss dafür nicht laufen. Bei einer Neuinstallation aktiviert das nächste `up -d` Liveness; bei einem bereits laufenden Server ist `docker compose -f server/deploy/compose.cpu.yml restart server` erforderlich. `up -d` allein lädt gespeicherte Einstellungen nicht neu. Für CUDA verwenden Sie `compose.cuda12.yml`.

Für GPU verwenden Sie `compose.cuda12.yml` und Port `18098`. Vor dem Download erscheint die Modelllizenz. Öffentliche InsightFace-Modelle sind ohne separate kommerzielle Lizenz nur für nichtkommerzielle Forschung bestimmt.

Die mitgelieferte Compose-Konfiguration deaktiviert Authentifizierung standardmäßig für isolierte Tests. Für andere Benutzer oder Netze setzen Sie vor dem Start `INSIGHTFACE_AUTH_ENABLED=true` und einen langen `INSIGHTFACE_API_KEY`. Danach: Dashboard prüfen, Collection erstellen, Person registrieren und mit einem anderen Bild suchen. Stoppen Sie mit `docker compose ... down` ohne `-v`, damit das Datenvolume erhalten bleibt.

## Optionales Liveness-Addon

Liveness ist in `server/config/server.toml` standardmäßig deaktiviert: `inference.addons` und `addons.auto_download` sind `[]`. Alte Konfigurationen ohne diese Schlüssel bleiben deaktiviert.

**Über die Kommandozeile aktivieren, auch vor dem ersten Serverstart:**

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

`--enable-liveness` prüft zuerst, ob die vorhandene Konfiguration aktualisiert werden kann. Danach installiert und verifiziert es Basispaket, konfigurierte Installations-Addons und Liveness und ergänzt `liveness` in `inference.addons` sowie `addons.auto_download`. Andere Einträge, Kommentare und Einstellungen bleiben erhalten. Geprüfte Caches werden wiederverwendet; auch bei einem Cache-Treffer wird die Aktivierung gespeichert. Ein fehlgeschlagener Download lässt die Konfiguration unverändert. Ein Speicherfehler liefert eine klare Fehlermeldung und einen von null verschiedenen Exit-Status; gültige Caches bleiben für einen erneuten Versuch nutzbar.

Beide Compose-Dienste binden das gesamte vorhandene `server/config` schreibbar unter `/etc/insightface` ein, mit `create_host_path: false`. So aktualisiert der Installer die Hostkonfiguration atomar, ohne dass Server läuft. Verzeichnis und `server.toml` müssen vorhanden sein.

Server muss dafür nicht laufen. Bei einer Neuinstallation aktiviert das nächste `up -d` Liveness; bei einem bereits laufenden Server ist `docker compose -f server/deploy/compose.cpu.yml restart server` erforderlich. `up -d` allein lädt gespeicherte Einstellungen nicht neu. Für CUDA verwenden Sie `compose.cuda12.yml`.

Ohne `--enable-liveness` bleibt `models install` unverändert und schreibt keine Konfiguration; Liveness bleibt standardmäßig deaktiviert. `models addons install liveness` lädt und verifiziert nur das Addon, aktiviert es aber nicht. Alternativ können Sie unten **System → Lebenderkennung** verwenden.

Wählen Sie unter **System → Lebenderkennung** die Aktion **Herunterladen und nach Neustart aktivieren**. Erst nach SHA-256-Prüfung wird `liveness` beiden Listen hinzugefügt; andere Einträge, Kommentare und Einstellungen bleiben erhalten. Geprüfte Dateien werden wiederverwendet. **Starten Sie den Server manuell neu**, damit die Änderung wirkt. Fehler sind sichtbar und wiederholbar; fehlgeschlagene Downloads aktivieren Liveness nicht.

System zeigt die geprüfte Installation (`installed`), den aktuellen Betrieb (`enabled`), die gespeicherte Einstellung für den nächsten Start (`configured_enabled`) und den Neustartbedarf (`restart_required`) getrennt. Download und Speichern ändern die laufende Inferenz noch nicht. Zum Deaktivieren setzen Sie in derselben Datei `inference.addons=[]` und `addons.auto_download=[]` und starten manuell neu. Die Web-Aktion ändert die Registrierungseinstellung nicht; deren Standard bleibt `liveness_on_registration=false`.

**Erweitert: Liveness manuell konfigurieren.** Diese Einstellungen ersetzen das Aktivierungsflag oder die Web-Aktion; installieren Sie das Modell vor dem Neustart.

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

### Modellinstallation und Start

`inference.addons` steuert die Laufzeit, `addons.auto_download` den Zusatzdownload beim Installieren eines Basispakets. Mit `["liveness"]` wird das Addon auch bei bereits gecachtem Basispaket installiert. Beim Serverstart erfolgt kein Download. Installer und Server lesen dieselbe Datei.

```bash
docker compose -f server/deploy/compose.cpu.yml run --rm models addons install liveness
docker compose -f server/deploy/compose.cpu.yml run --rm models addons verify liveness
docker compose -f server/deploy/compose.cpu.yml up -d --force-recreate
```

Ein aktiviertes, fehlendes Modell stoppt den Start mit `addon_model_missing`; ein ungültiges Modell mit `addon_model_invalid`. Das Addon wird nicht stillschweigend deaktiviert.

### Mounts und Berechtigungen für Web-Downloads

Die mitgelieferten Compose-Dateien binden das gesamte `/models` schreibbar ein; ein separates Mount für `/models/addons` ist nicht nötig. Server und Installer verwenden root (`0:0`), und die Installation erstellt `addons` bei Bedarf. Das gesamte vorhandene Verzeichnis `server/config` wird in beiden Diensten schreibbar unter `/etc/insightface` eingebunden, damit die Web-Aktion und `--enable-liveness` `server.toml` atomar speichern können. Mit diesen Vorgaben sind keine UID/GID-Exporte, `chgrp`- oder `chmod`-Schritte erforderlich. Bewusst schreibgeschützte Dateien oder Mounts verhindern weiterhin die Web-Verwaltung; prüfen Sie bei eigenen Mounts die Zugriffsrechte.

Das Modell-Mount verwendet in beiden Diensten `create_host_path: true`. Die Dienste behalten die Docker-Standard-Capabilities; `cap_drop: [ALL]` wird nicht gesetzt. Das übrige Container-Dateisystem bleibt schreibgeschützt und `no-new-privileges` aktiv. Root kann Dateien in schreibbaren Mounts ändern; neue Downloads können auf dem Host root gehören.

Bei eigenen Mounts verwenden Sie deren tatsächliche Hostpfade; für CUDA nehmen Sie `compose.cuda12.yml`. Alte schreibgeschützte Mounts funktionieren weiterhin bei deaktivierter Liveness. Ist die Web-Aktion nicht verfügbar, zeigt sie den Grund; alternativ können Sie das Modell per CLI installieren und die Konfiguration selbst bearbeiten. Nach erfolgreichem Speichern im Web übernimmt `docker compose -f server/deploy/compose.cpu.yml restart server` die Änderung. Änderungen an Mounts oder Proxy-Variablen erfordern eine Neuerstellung des Containers.

Setzen Sie bei Bedarf vor der Containererstellung `HTTP_PROXY`, `HTTPS_PROXY` und `NO_PROXY`; Compose reicht sie an Server und Modellwerkzeug weiter. Nutzen Sie eine vom Container erreichbare LAN-Adresse des Proxys: `127.0.0.1` im Container bezeichnet nicht den Mac. Die Aktion nutzt die bestehende API-Key-Authentifizierung; ohne Authentifizierung können erreichbare API-Clients sie ebenfalls ausführen. Sie lädt nur das festgelegte veröffentlichte Liveness-Modell, keine beliebige URL, und wechselt kein Basismodellpaket.

### Liveness-Ergebnisse

| Ergebnis | `status` | `is_live` | `live_score` |
| --- | --- | --- | --- |
| Prüfung bestanden | `ok` | `true` | `[0, 1]` |
| Nicht lebend | `ok` | `false` | `[0, 1]` |
| Eingabe abgelehnt | `input_rejected` | `null` | `null` |

Nur eine unzureichende Fläche des Originalbilds um das ausgerichtete Gesicht führt zu `input_rejected`. Dieses Ergebnis enthält zusätzlich `liveness.reason` als verständliche Erklärung; bei echten Gesichtern und Fälschungen fehlt `reason`. FaceAnalysis und die API liefern diesen Text immer auf Englisch; nur die Web-Oberfläche übersetzt die Anzeige. Verwenden Sie für Programmentscheidungen `status` und `is_live`, nicht den Wortlaut von `reason`. Ältere gespeicherte Ergebnisse können ohne `reason` vorliegen; Clients können dann einen allgemeinen Hinweis zur abgelehnten Eingabe anzeigen.

```json
{
  "status": "input_rejected",
  "is_live": null,
  "live_score": null,
  "reason": "Insufficient image area around the face for liveness detection. Move the face toward the center, step back from the camera, or use a less tightly cropped image."
}
```

`normal` erkennt nur Gesichter mit bestandener Liveness-Prüfung. `observe` protokolliert das Ergebnis und setzt die Erkennung fort. Ohne Prüfung fehlt `liveness`. Die drei Kernfelder sind `status`, `is_live` und `live_score`: Bei bestanden/fake gilt `status: ok` mit Wahrheitswert und Score; bei abgelehnter Eingabe gilt `status: input_rejected`, die anderen Werte sind `null`.

Detect liefert auch negative Ergebnisse mit HTTP 200. In `normal` liefern Embeddings, Vergleich und Suche HTTP 422 `liveness_fake` oder `liveness_input_rejected` mit `error.details.liveness`; Vergleiche ergänzen `details.side`. Laufzeitfehler liefern HTTP 503 `liveness_unavailable`. Laufzeitfehler brechen den Vorgang sowohl in `normal` als auch in `observe` ab; sie werden nicht als `input_rejected` zurückgegeben.

Für neue Personen und zusätzliche FaceSamples wird Liveness standardmäßig übersprungen: `[inference].liveness_on_registration=false` führt das Modell nicht aus und lässt `liveness` in neuen Samples weg. Mit `true` gilt bei aktiviertem Addon die Richtlinie `normal`/`observe`; Ablehnungen enthalten `reason` und `liveness`. Die Qualitätsprüfung über `review_mode` und die Validierung externer Embeddings bleiben aktiv. `review_mode=off` und `external_trusted` umgehen eine aktivierte Registrierungsprüfung nicht. Requests können diese Startkonfiguration nicht überschreiben. Bereits gespeicherte Ergebnisse bleiben sichtbar.

RTSP unterscheidet `liveness_blocked` von `unknown` und zählt `liveness_blocked_faces`. Blockierte Gesichter erzeugen keine Personen-/Unbekannt-Eintrittsereignisse und setzen die Bestätigungsfolge zurück. Bei Inferenzfehlern werden alte Identitätsanzeigen gelöscht.

`liveness_compare_scope` wählt für `/v1/compare` zwischen `both` (Standard), `source` und `target`. Die Prüfung besteht bei `live_score >= liveness_threshold`.

Das Modell liegt auf dem Host unter `server/.models/addons/liveness.onnx`, im Container unter `/models/addons/liveness.onnx`. `addons` in `/v1/models` und `/v1/system` zeigt die aktuell aktiven Addons.

[Vollständiger API-Vertrag](api.de.md#optionales-liveness-addon).

## 1. Anmelden und Bereitschaft prüfen

Öffnen Sie für CPU `http://SERVER:18097/` oder für CUDA 12 `http://SERVER:18098/`. Wenn Authentifizierung aktiv ist, tragen Sie unter **API-Schlüssel konfigurieren** den vom Betreiber erhaltenen Key ein. Er verbleibt nur im Speicher des Tabs und wird beim Neuladen oder Schließen gelöscht.

Prüfen Sie unter **Übersicht** oder **System**, dass Dienst, Datenbank, Modelle und Provider bereit sind. Eine CUDA-Instanz muss `CUDAExecutionProvider` melden und fällt nicht still auf CPU zurück.

Das Dashboard zeigt unter dem Modellnamen immer Liveness aktiviert oder deaktiviert. System trennt Installation, aktuellen Betrieb und ausstehenden Neustart.

## 2. Collection anlegen

Unter **Sammlungen** → **Neue Sammlung** setzen Sie eine stabile ID, Name,
Standard-Cosinus-Schwelle (`0.4`), ein verfügbares Suchprofil, Kapazität und
maximale FaceSamples pro Person. Die Speicherung eines auf 112×112 skalierten
`bounding-box crop` als JPEG ist standardmäßig aus; es ist nicht der
ausgerichtete Erkennungseingang.

Eine Collection ist an Modell-ID, Digest, Dimension und Vorverarbeitung gebunden. Nach einem Modellwechsel bleibt sie sichtbar, aber Registrierung und Suche werden bei abweichendem Vertrag ausdrücklich abgelehnt.

Das Erkennungsprofil kopiert beim Anlegen die Systemwerte und kann später für Eingabegrößen, Erkennungs-/NMS-Schwelle und Ein-Gesicht-Strategie geändert werden. `largest` priorisiert die Fläche; `center_largest` maximiert `Fläche - 2,0 × quadrierter Pixelabstand zwischen Box- und Bildmitte`. Die Erkennungskonfidenz gehört nicht zu diesem Wert.

## 3. Person registrieren

Wählen Sie unter **Personen** eine Collection und **Person registrieren**. Geben Sie optional ID, Name, externe ID, JSON-Metadaten und ein oder mehrere JPEG-, PNG-, WebP- oder BMP-Bilder an.

- `off`: verwendet die Ein-Gesicht-Strategie der Collection; mehrere Gesichter sind erlaubt.
- `standard`: genau ein nutzbares Gesicht sowie Prüfungen von Größe, Erkennungswert, Schärfe, Helligkeit und Pose.
- `strict`: zusätzlich muss die beste Ähnlichkeit innerhalb der Person höher sein als die beste Ähnlichkeit zu anderen Personen.

Stapelregistrierung kann teilweise erfolgreich sein und meldet den Ablehnungsgrund pro Bild. Originale werden nicht gespeichert. `external_trusted` akzeptiert ein L2-normalisiertes Embedding; das Bild bleibt für Erkennung und Qualitätsprüfung erforderlich, das Embedding wird aber nicht erneut extrahiert.

Neue Personen und zusätzliche FaceSamples überspringen Liveness standardmäßig (`liveness_on_registration=false`). Bei aktivierter Prüfung lehnt `normal` fake/ungeeignete Eingaben ab; `observe` speichert das Ergebnis und fährt fort. Die Qualitätsprüfung richtet sich weiterhin nach dem gewählten `review_mode`. Ablehnungen zeigen den tatsächlichen `reason` und das Liveness-Ergebnis getrennt.

## 4. Erkennen, vergleichen und suchen

**Erkennen** zeigt Boxen, fünf Landmarken, Erkennungswert und Qualität; kein Gesicht ist eine erfolgreiche leere Liste. **Vergleichen** wählt mit dem System- oder Collection-Profil je ein Gesicht und liefert Cosinus-`similarity`, `threshold` und `matched`. Ähnlichkeit ist keine Wahrscheinlichkeit.

Unter **Suchen** wählen Sie Collection und Bild. Der Person-Score ist die höchste Ähnlichkeit aller FaceSamples dieser Person. Ergebnisse sind absteigend sortiert; kein Treffer ist eine leere Liste. Neue Samples werden zuerst in SQLite bestätigt und vor der Erfolgsantwort in den Speicherindex eingefügt. Beim Neustart wird der Index aus SQLite neu aufgebaut.

Bei aktivierter Prüfung enthält jedes bewertete Gesicht `liveness.status`, `liveness.is_live` und `liveness.live_score`. Auch fake und `input_rejected` liefern HTTP 200, ohne Erkennungsmerkmale zu extrahieren. `input_rejected` bedeutet, dass um das Gesicht zu wenig Bildfläche vorhanden ist; `liveness.reason` erklärt, wie sich die Aufnahme verbessern lässt. Fehlt `liveness`, wurde es nicht bewertet.

`liveness_compare_scope` (`both`, `source`, `target`) bestimmt die vor der Erkennung geprüften Seiten. `normal` liefert bei Ablehnung HTTP 422 `liveness_fake` / `liveness_input_rejected`, `error.details.liveness` und `error.details.side`, ohne Ähnlichkeit. `observe` vergleicht weiter und liefert Ergebnisse an den bewerteten Gesichtern.

Mit Liveness in `normal` führen fake/ungeeignete Suchbilder zu HTTP 422 `liveness_fake` / `liveness_input_rejected` und `error.details.liveness`; die Suche läuft nicht. Das unterscheidet sich von einer erfolgreichen leeren Trefferliste. `observe` sucht weiter und liefert Liveness am Suchgesicht.

## 5. RTSP-Kameraüberwachung

Erstellen Sie unter **Kameraüberwachung** einen dauerhaften Monitor und konfigurieren Sie RTSP-Quelle, Collection, Inferenzrate, optionalen Schwellwert und Ereignisregeln. Die Vorschau ist standardmäßig aus; Erkennung und Ereignisse laufen trotzdem. Bei aktivierter Vorschau zeichnet die Web UI grüne registrierte und orange unbekannte Gesichter aus den `/state`-Daten über rohe Bilder.

Der Monitor läuft unabhängig vom Browser; aktivierte Aufgaben werden nach Server-Neustart wiederhergestellt. Einstellungen liegen in SQLite und RTSP-Zugangsdaten verschlüsselt unter `/data`, Videobilder und Ereignisse werden jedoch nicht gespeichert. Ereignisse bleiben nur im begrenzten RAM-Puffer. Der Decoder behält den neuesten Frame und überspringt veraltete Frames statt sie aufzustauen.

Mit Liveness in `normal` erhalten blockierte Gesichter `status: liveness_blocked` und ein separates Liveness-Ergebnis. Sie zählen zu `liveness_blocked_faces`, nicht zu `unknown_faces`, und erzeugen keine Eintrittsereignisse. `observe` erkennt weiter. Eingabeablehnung und fake werden getrennt angezeigt.

## 6. Daten, Backup und Sicherheit

Persistieren und sichern Sie `/data`, das Modellverzeichnis und die Konfiguration. Das mitgelieferte Deployment verwendet root mit den Docker-Standard-Capabilities und einem schreibbaren `/models`; dadurch kann der Dienst die eingebundenen Modelle, Konfiguration und Daten ändern. Das übrige Container-Dateisystem bleibt schreibgeschützt und `no-new-privileges` bleibt aktiv; `privileged` ist nicht erforderlich. Beschränken Sie Docker- und Hostzugriff und binden Sie keine fremden Hostverzeichnisse ein. Sichern Sie SQLite und optional gespeicherte Gesichtsbilder gemeinsam vor Massenänderungen. API Keys werden gehasht; ein neuer `INSIGHTFACE_API_KEY` beim Start mit demselben Volume rotiert den aktiven Key. Bilder, Embeddings und Keys gehören nicht in Logs.

Der OpenAPI-Schema-Explorer für Entwickler liegt unter `/docs`; aufgabenbezogene API-Anleitungen stehen in dieser Hilfe. Nennen Sie bei Fehlern `x-request-id`. Prüfen Sie bei `401` den Key, bei `409 collection_model_mismatch` den Modellvertrag und bei `422 face_not_found` das Eingabebild.

## 7. Modelle und Lizenzen

Die Images enthalten keine Modelle. Der normale Start bleibt offline; der
einmalige Dienst `models` installiert nach `server/.models`:

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models verify buffalo_l
```

Unterstützt sind `buffalo_l` (`det_10g.onnx` + `w600k_r50.onnx`),
`buffalo_m`, `buffalo_s`, `buffalo_sc`, `antelopev2`, `raccoon_s` und
`raccoon_l`. Die Installation erzeugt
`manifest.json` und die signierte `MODEL.LICENSE`. Ohne `--accept-license`
fragt ein interaktives Terminal vor dem Download nach einer Bestätigung.
Nichtinteraktive Befehle benötigen die Option und beenden sich ohne sie,
ohne etwas herunterzuladen. Öffentliche
vortrainierte InsightFace-Modelle sind ohne separate kommerzielle Lizenz nur
für nichtkommerzielle Forschung bestimmt.

`raccoon_s` und `raccoon_l` werden unterstützt. Der Server installiert aus jedem Paket nur Detektion und Erkennung; der Raccoon-Verifier wird nicht geladen. Die Identität ist der Modellname, ohne separate Modellversionsnummer. Die Liveness-Aktion im Web wechselt kein Basismodell. Bei einem anderen Erkennungsmodell benötigen Sie eine passende Collection; vorhandene Merkmale werden nicht als Merkmale des neuen Modells umgedeutet.

## 8. Startkonfiguration und Suche

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

`server/config/server.toml` wird einmal beim Start gelesen; Änderungen benötigen
einen Neustart. Standardwerte sind `input_sizes=[[96,96],[512,512]]`,
Detektionsschwelle `0.50`, NMS `0.40`, `single_face_selection="largest"` und
höchstens 100 Gesichter. SCRFD führt jede Auflösung aus, projiziert alle
Kandidaten in das Originalbild und wendet einmal globales NMS an.
`max_concurrency="auto"` bedeutet CPU 4 und CUDA 8.
`[web].disabled=true` lässt nur `/v1` und `/openapi.json` aktiv.

System zeigt nur verfügbare Suchprofile. Ein Profil ist nach Erstellung einer
Collection fest und kann nicht pro Search gewechselt werden:

- `fp32_v1`: Standard für CPU/CUDA;
- `fp16_v1`: CUDA;
- `bf16_v1`: unterstützte CPU oder SM80+ CUDA;
- `int8_x736_v1`: empfohlenes INT8 für CPU/CUDA, INT32-Akkumulation;
- `int8_x1000_v1`: Kompatibilität bestehender Collections.

Alle Profile durchsuchen jeden FaceSample vollständig und sind keine
ANN-Indizes; öffentliche Werte bleiben raw cosine. `capacity_rows` ist
standardmäßig `100000`, der Guardrail `10000000`,
`max_faces_per_person=20`. Bei 512 Dimensionen benötigt nur der Vektor ungefähr
2.048 Byte FP32, 1.024 Byte FP16/BF16 oder 512 Byte INT8 pro Zeile.

## 9. SDK, eigener Build und Datenbetrieb

Das Python SDK akzeptiert Pfade, bytes und file-like objects und bietet typisierte
Methoden für Detect, Compare, Collection, Registrierung, Search und Monitor.
Den vollständigen HTTP-Vertrag beschreibt der [API-Leitfaden](api.de.md).

Sie können direkt aus einem vollständigen lokalen Quellcodeverzeichnis bauen,
auch mit nicht committeten Änderungen oder ohne `.git`-Verzeichnis. Git-Commits
und Pushes sind keine Voraussetzung für den Build.

```bash
make -C server build-cpu
make -C server build-cuda12
```

Veröffentlichen Sie nach bestandenen Tests dasselbe geprüfte Image. Werden
anschließend nur derselbe Quellcode committet oder Commits geordnet, ist kein
neuer Build nötig. Änderungen an Dateien im Image, etwa Code, Frontend-Dateien
oder mitgelieferter Benutzerhilfe, erfordern einen neuen Build und eine erneute
Validierung.

Für lokale Images verwenden Compose-Befehle `--pull never`. Unveränderliche
Tags sind `0.3.1-cpu` und `0.3.1-cuda12`; `cpu` und `cuda12` zeigen auf die
jeweils neueste stabile Variante, ein `latest` gibt es absichtlich nicht.
Vor einem Upgrade Schreibzugriffe stoppen und `/data` samt Crop-Speicher
SQLite-sicher sichern. `docker compose down -v` nicht verwenden, weil es das
Datenvolume löscht.

### Upgrade auf 0.3.1

0.3.1 vereinfacht das Deployment: Server und Modellinstaller laufen als root, Compose erstellt das Modellverzeichnis bei Bedarf, und ein einzelnes schreibbares `/models` ersetzt das separate Addon-Mount.

Seit 0.3.0 werden `raccoon_s`, `raccoon_l`, ihre Modell-Manifeste, optionale Liveness, Web-Installation von Addons und BMP-Eingaben unterstützt. Der Server nutzt Raccoons Detektion und Erkennung; der Verifier wird nicht geladen. Diese Funktionen und API-Antwortverträge bleiben in 0.3.1 unverändert.

**1.** Aktualisieren Sie den Server-Quellcode und die Compose-Dateien auf 0.3.1,
behalten Sie dabei Ihre Einstellungen in `server/config/server.toml` und
Ihre Deployment-Overrides. Erhalten Sie den bisherigen Modellpfad, den
Namen des `/data`-Volumes, den Crop-Speicher, die Ports und die Einstellungen
für den API Key. Setzen Sie bei eigenen Compose-Dateien die Images beider
Dienste `server` und `models` passend zur Umgebung auf `0.3.1-cpu` oder
`0.3.1-cuda12`. Verwenden Sie für die folgenden Befehle dieselben
Compose-Dateien, Overrides und denselben Projektnamen wie bisher.

Aktualisieren Sie vor dem Start auch eigene Compose-Overrides, nicht nur die Image-Tags: Beide Dienste benötigen `user: "0:0"`, ohne `cap_drop: [ALL]` und ohne die bisherigen UID/GID- oder `group_add`-Vorgaben. Verwenden Sie für beide Dienste ein einziges schreibbares Modell-Bind-Mount unter `/models` mit `create_host_path: true`; entfernen Sie das separate `/models/addons`-Mount und `x-addons-path`. Behalten Sie den tatsächlichen Modellpfad und vorhandene Dateien einschließlich Addons. Konfigurations-Mounts verwenden weiterhin `create_host_path: false`, daher müssen `server/config` und `server.toml` vorhanden bleiben. Behalten Sie das Datenvolume und erstellen Sie vor der Umstellung eine Sicherung von Daten, Modellen und Konfiguration. Entfernen Sie root- oder mountbezogene Altwerte aus eigenen Overrides; ein neues Image allein übernimmt diese Änderungen nicht.

Behalten Sie das schreibgeschützte Container-Dateisystem und `no-new-privileges` bei. Beide Dienste benötigen das gesamte Konfigurationsverzeichnis schreibbar. Ersetzen Sie das bisherige schreibgeschützte Einzeldatei-Mount des Installers durch das Verzeichnis-Mount. Für das Standarddeployment sind weder eine rekursive Änderung bestehender Dateirechte noch die Vorbereitung eines Addon-Verzeichnisses erforderlich.

**2.** Laden Sie die neuen Images und erstellen Sie den Server-Container neu.
Wählen Sie im Repository-Stamm die Befehle für Ihr bestehendes Deployment:

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

Bei einem lokalen Build bauen Sie zuerst die 0.3.1-Images und verwenden
`up -d --no-build --pull never --force-recreate server`, statt Images
herunterzuladen. `docker compose restart` allein wechselt weder auf ein
neues Image noch übernimmt es geänderte Mounts.

**3.** Beim Start werden die Datenbankmigrationen automatisch angewendet. Warten
Sie, bis `/v1/health` den Status `ready` und die Version `0.3.1` meldet, und
prüfen Sie unter **System** das erwartete Modell und den Ausführungsprovider.
Kontrollieren Sie, dass Ihre bestehenden Collections und Personen vorhanden
sind, und führen Sie eine bekannte Suche aus. Bei unverändertem Modell und
Embedding-Vertrag bleiben Samples, Embeddings und die Vertrags-IDs der
Collections erhalten; eine erneute Registrierung ist nicht erforderlich.

**Die Lebenderkennung bleibt nach dem Upgrade optional.** Sowohl die
mitgelieferte Konfiguration als auch ältere Konfigurationen ohne Addon-Schlüssel
lassen sie deaktiviert. Ein reines Server-Upgrade benötigt daher keinen
Liveness-Download. Beim Serverstart werden keine Modelle heruntergeladen.
Folgen Sie zum Aktivieren der [Liveness-Einrichtung](#optionales-liveness-addon):
Bereiten Sie die [Web-Mounts und Berechtigungen](#mounts-und-berechtigungen-für-web-downloads)
vor, wählen Sie **System → Lebenderkennung → Herunterladen und nach Neustart aktivieren**,
warten Sie auf die erfolgreiche Installation und das Speichern der
Konfiguration und starten Sie den Server anschließend manuell neu. Die
Standardwerte sind `normal`, Schwelle `0.8` und
`liveness_on_registration=false`. Das Modell liegt weiterhin unter
`<models_dir>/addons/liveness.onnx`.

**Raccoon einzusetzen ist ein separater Modellwechsel.** Das Server-Upgrade
behält Ihr aktuelles Modellpaket. Um `raccoon_s` oder `raccoon_l` einzusetzen,
installieren Sie das gewünschte Paket nach der
[Modellinstallationsanleitung](#7-modelle-und-lizenzen) in einem separaten
Modellverzeichnis und konfigurieren Sie ein Deployment dafür. Collections
müssen zum Embedding-Vertrag des neuen Modells passen. Erstellen Sie passende
Collections und registrieren Sie die Personen erneut, oder führen Sie eine
separate Datenmigration durch. Die Web UI wechselt keine Basismodellpakete.

**API- und SDK-Kompatibilität seit 0.3.0:** Ergebnisse für Modelle, Collections und
FaceSamples enthalten kein `model_version` mehr. `model_id` identifiziert das
Modell, `embedding_contract_id` bestimmt die Collection-Kompatibilität.
Aktualisieren Sie Clients, die das entfernte Feld voraussetzen, und verwenden
Sie beim Upgrade des mitgelieferten Python-Clients das SDK `0.3.1`. Wird
Liveness ausgewertet, enthält `liveness` die Kernfelder `status`, `is_live` und
`live_score`, bei `input_rejected` zusätzlich `reason`; ohne Auswertung fehlt das Feld. Lesen Sie vor der Aktivierung
für Erkennungsanfragen die [Liveness-Ergebnisse und Fehlerregeln](#liveness-ergebnisse).

## 10. GPU, Netzwerk und Fehlerbehebung

Das CUDA-Image enthält CUDA Runtime 12.9.1, cuDNN 9.24.0 und
`onnxruntime-gpu==1.27.0`. Turing/Ampere/Ada/Hopper benötigen mindestens R535,
Blackwell/RTX 50 mindestens 570.26; für neue Installationen wird ein stabiler
R580 oder neuer empfohlen. Der Start prüft GPU, Compute Capability, Driver,
CUDA/cuDNN/ORT, Provider, echte Modell-Sessions und Warm-up und verweigert
stillen CPU-Fallback.

Bei Netzzugriff HTTPS an einem vertrauenswürdigen Reverse Proxy terminieren,
CORS-Ursprünge sowie Rate/Body/Timeout begrenzen und `/data` und Backups als
biometrische Daten schützen. Bilder, Embeddings und Keys nie protokollieren.
Phase eins besitzt nur einen undifferenzierten API Key und ist kein
Mandanten-Berechtigungssystem.
