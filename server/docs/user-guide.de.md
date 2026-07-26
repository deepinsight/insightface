# InsightFace Server Benutzerhandbuch

**Sprachen:** [English](user-guide.md) · [中文](user-guide.zh-CN.md) · [日本語](user-guide.ja.md) · Deutsch · [Español](user-guide.es.md) · [Français](user-guide.fr.md) · [Русский](user-guide.ru.md) · [Português](user-guide.pt.md) · [한국어](user-guide.ko.md)

Dieses Handbuch führt Erstanwender vom leeren Checkout bis zur ersten erfolgreichen Personensuche. Dieselben Funktionen stehen über Web UI, `/v1` und Python SDK bereit. Alle HTTP-Felder und Ergebnisse beschreibt der [API-Leitfaden](api.de.md).

## Vom Start bis zur ersten Suche

CPU benötigt Linux x86_64, Docker Engine und Docker Compose. CUDA benötigt zusätzlich einen kompatiblen NVIDIA-Treiber und NVIDIA Container Toolkit; CUDA, cuDNN, ORT, Python und OpenCV müssen nicht auf dem Host installiert sein.

```bash
mkdir -p server/.models
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml run --rm models install buffalo_l
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

Für GPU verwenden Sie `compose.cuda12.yml` und Port `18098`. Vor dem Download erscheint die Modelllizenz. Öffentliche InsightFace-Modelle sind ohne separate kommerzielle Lizenz nur für nichtkommerzielle Forschung bestimmt.

Die mitgelieferte Compose-Konfiguration deaktiviert Authentifizierung standardmäßig für isolierte Tests. Für andere Benutzer oder Netze setzen Sie vor dem Start `INSIGHTFACE_AUTH_ENABLED=true` und einen langen `INSIGHTFACE_API_KEY`. Danach: Dashboard prüfen, Collection erstellen, Person registrieren und mit einem anderen Bild suchen. Stoppen Sie mit `docker compose ... down` ohne `-v`, damit das Datenvolume erhalten bleibt.

## 1. Anmelden und Bereitschaft prüfen

Öffnen Sie für CPU `http://SERVER:18097/` oder für CUDA 12 `http://SERVER:18098/`. Wenn Authentifizierung aktiv ist, tragen Sie unter **API-Schlüssel konfigurieren** den vom Betreiber erhaltenen Key ein. Er verbleibt nur im Speicher des Tabs und wird beim Neuladen oder Schließen gelöscht.

Prüfen Sie unter **Übersicht** oder **System**, dass Dienst, Datenbank, Modelle und Provider bereit sind. Eine CUDA-Instanz muss `CUDAExecutionProvider` melden und fällt nicht still auf CPU zurück.

## 2. Collection anlegen

Unter **Sammlungen** → **Neue Sammlung** setzen Sie eine stabile ID, Name,
Standard-Cosinus-Schwelle (`0.4`), ein verfügbares Suchprofil, Kapazität und
maximale FaceSamples pro Person. Die Speicherung eines auf 112×112 skalierten
`bounding-box crop` als JPEG ist standardmäßig aus; es ist nicht der
ausgerichtete Erkennungseingang.

Eine Collection ist an Modell-ID, Version, Digest, Dimension und Vorverarbeitung gebunden. Nach einem Modellwechsel bleibt sie sichtbar, aber Registrierung und Suche werden bei abweichendem Vertrag ausdrücklich abgelehnt.

Das Erkennungsprofil kopiert beim Anlegen die Systemwerte und kann später für Eingabegrößen, Erkennungs-/NMS-Schwelle und Ein-Gesicht-Strategie geändert werden. `largest` priorisiert die Fläche; `center_largest` maximiert `Fläche - 2,0 × quadrierter Pixelabstand zwischen Box- und Bildmitte`. Die Erkennungskonfidenz gehört nicht zu diesem Wert.

## 3. Person registrieren

Wählen Sie unter **Personen** eine Collection und **Person registrieren**. Geben Sie optional ID, Name, externe ID, JSON-Metadaten und ein oder mehrere JPEG-, PNG- oder WebP-Bilder an.

- `off`: verwendet die Ein-Gesicht-Strategie der Collection; mehrere Gesichter sind erlaubt.
- `standard`: genau ein nutzbares Gesicht sowie Prüfungen von Größe, Erkennungswert, Schärfe, Helligkeit und Pose.
- `strict`: zusätzlich muss die beste Ähnlichkeit innerhalb der Person höher sein als die beste Ähnlichkeit zu anderen Personen.

Stapelregistrierung kann teilweise erfolgreich sein und meldet den Ablehnungsgrund pro Bild. Originale werden nicht gespeichert. `external_trusted` akzeptiert ein L2-normalisiertes Embedding; das Bild bleibt für Erkennung und Qualitätsprüfung erforderlich, das Embedding wird aber nicht erneut extrahiert.

## 4. Erkennen, vergleichen und suchen

**Erkennen** zeigt Boxen, fünf Landmarken, Erkennungswert und Qualität; kein Gesicht ist eine erfolgreiche leere Liste. **Vergleichen** wählt mit dem System- oder Collection-Profil je ein Gesicht und liefert Cosinus-`similarity`, `threshold` und `matched`. Ähnlichkeit ist keine Wahrscheinlichkeit.

Unter **Suchen** wählen Sie Collection und Bild. Der Person-Score ist die höchste Ähnlichkeit aller FaceSamples dieser Person. Ergebnisse sind absteigend sortiert; kein Treffer ist eine leere Liste. Neue Samples werden zuerst in SQLite bestätigt und vor der Erfolgsantwort in den Speicherindex eingefügt. Beim Neustart wird der Index aus SQLite neu aufgebaut.

## 5. RTSP-Kameraüberwachung

Erstellen Sie unter **Kameraüberwachung** einen dauerhaften Monitor und konfigurieren Sie RTSP-Quelle, Collection, Inferenzrate, optionalen Schwellwert und Ereignisregeln. Die Vorschau ist standardmäßig aus; Erkennung und Ereignisse laufen trotzdem. Bei aktivierter Vorschau zeichnet die Web UI grüne registrierte und orange unbekannte Gesichter aus den `/state`-Daten über rohe Bilder.

Der Monitor läuft unabhängig vom Browser; aktivierte Aufgaben werden nach Server-Neustart wiederhergestellt. Einstellungen liegen in SQLite und RTSP-Zugangsdaten verschlüsselt unter `/data`, Videobilder und Ereignisse werden jedoch nicht gespeichert. Ereignisse bleiben nur im begrenzten RAM-Puffer. Der Decoder behält den neuesten Frame und überspringt veraltete Frames statt sie aufzustauen.

## 6. Daten, Backup und Sicherheit

Persistieren Sie `/data` und mounten Sie `/models` schreibgeschützt. Sichern Sie SQLite und optional gespeicherte Gesichtsbilder gemeinsam vor Massenänderungen. API Keys werden gehasht; ein neuer `INSIGHTFACE_API_KEY` beim Start mit demselben Volume rotiert den aktiven Key. Bilder, Embeddings und Keys gehören nicht in Logs.

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
`buffalo_m`, `buffalo_sc` und `antelopev2`. Die Installation erzeugt
`manifest.json` und die signierte `MODEL.LICENSE`. Ohne `--accept-license`
zeigt das Werkzeug die Bedingungen und lädt nichts herunter. Öffentliche
vortrainierte InsightFace-Modelle sind ohne separate kommerzielle Lizenz nur
für nichtkommerzielle Forschung bestimmt.

## 8. Startkonfiguration und Suche

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

Aus einem vollständigen Repository kann jeder die Images bauen:

```bash
make -C server build-cpu
make -C server build-cuda12
```

Für lokale Images verwenden Compose-Befehle `--pull never`. Unveränderliche
Tags sind `0.2.0-cpu` und `0.2.0-cuda12`; `cpu` und `cuda12` zeigen auf die
jeweils neueste stabile Variante, ein `latest` gibt es absichtlich nicht.
Vor einem Upgrade Schreibzugriffe stoppen und `/data` samt Crop-Speicher
SQLite-sicher sichern. `docker compose down -v` nicht verwenden, weil es das
Datenvolume löscht.

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
