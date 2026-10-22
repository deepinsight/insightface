# InsightFace Server REST-API-Leitfaden

**Sprachen:** [English](api.md) · [中文](api.zh-CN.md) · [日本語](api.ja.md) · Deutsch · [Español](api.es.md) · [Français](api.fr.md) · [Русский](api.ru.md) · [Português](api.pt.md) · [한국어](api.ko.md)

Dieses Dokument erklärt Zweck, Eingaben, Serververhalten, Erfolg und typische Fehler aller öffentlichen Endpunkte. Start und Modellinstallation stehen im [Benutzerhandbuch](user-guide.de.md); das exakte laufende Schema finden Sie unter `/docs` und `/openapi.json`.


Modelle werden durch `model_id` identifiziert; Antworten enthalten kein separates `model_version`. Bestehende Collections behalten ihre `embedding_contract_id`.

Für Liveness siehe [Konfiguration, Modellinstallation und Ergebnisse](#optionales-liveness-addon). Die einzelnen Arbeitsschritte erklären zusätzlich die Auswirkungen.

## Gemeinsame Regeln

- Basispfad `/v1`, JSON in `snake_case`, Bilder als JPEG/PNG/WebP/BMP multipart.
- Die mitgelieferte Compose-Konfiguration deaktiviert Authentifizierung für isolierte Tests. Ist sie aktiv, benötigen alle Endpunkte außer health `Authorization: Bearer <api_key>`. Bei deaktivierter Authentifizierung den Header ganz weglassen.
- Jede Antwort trägt `x-request-id`; JSON wiederholt ihn als `request_id`.
- Confidence/quality/threshold liegen in `0..1`. Similarity ist keine Wahrscheinlichkeit, sondern roher Cosinus in `[-1,1]`; Standard `0.4`, Treffer bei `similarity >= threshold`.
- Cursor sind undurchsichtig und nur unverändert mit demselben Pfad, Collection-, Person- und Filterkontext wiederzuverwenden.
- Übliche Statuscodes: 400 Eingabe, 401 Auth, 404 nicht gefunden, 409 Konflikt, 413 Größe, 422 Bild/Gesicht, 429 Limit, 503 Timeout/Modell/Index.

```bash
BASE_URL=http://127.0.0.1:18097
AUTH_HEADER="Authorization: Bearer ${INSIGHTFACE_API_KEY}"
curl -fsS "${BASE_URL}/v1/health"
```

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

Unter **System → Lebenderkennung** können Sie das Modell herunterladen und für den nächsten Start aktivieren. Erst nach SHA-256-Prüfung wird `liveness` beiden Listen hinzugefügt; andere Einträge, Kommentare und Einstellungen bleiben erhalten. Geprüfte Dateien werden wiederverwendet. **Starten Sie den Server manuell neu**, damit die Änderung wirkt. Fehler sind sichtbar und wiederholbar; fehlgeschlagene Downloads aktivieren Liveness nicht.

[Mounts und Berechtigungen für Web-Downloads](user-guide.de.md#mounts-und-berechtigungen-für-web-downloads).

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

Führen Sie die Befehle vom Repository-Stamm mit vorhandener `server/config/server.toml` aus. Die mitgelieferten Compose-Dateien lassen den Installer als root laufen, erstellen das Modellverzeichnis bei Bedarf und binden `/models` schreibbar ein; der Addon-Download erstellt `addons` selbst. Eine UID/GID- oder manuelle Rechtevorbereitung ist nicht nötig. Details: [Ersteinrichtung im Benutzerhandbuch](user-guide.de.md).

```bash
docker compose -f server/deploy/compose.cpu.yml run --rm models addons install liveness
docker compose -f server/deploy/compose.cpu.yml run --rm models addons verify liveness
docker compose -f server/deploy/compose.cpu.yml up -d --force-recreate
```

Ein aktiviertes, fehlendes Modell stoppt den Start mit `addon_model_missing`; ein ungültiges Modell mit `addon_model_invalid`. Das Addon wird nicht stillschweigend deaktiviert.

### Liveness-Ergebnisse

| Ergebnis | `status` | `is_live` | `live_score` |
| --- | --- | --- | --- |
| Lebendes Gesicht | `ok` | `true` | `[0, 1]` |
| Täuschungsversuch | `ok` | `false` | `[0, 1]` |
| Ungeeignete Eingabe | `input_rejected` | `null` | `null` |

Nur eine unzureichende Fläche des Originalbilds um das ausgerichtete Gesicht führt zu `input_rejected`. Dieses Ergebnis enthält zusätzlich `liveness.reason` als verständliche Erklärung; bei echten Gesichtern und Fälschungen fehlt `reason`. FaceAnalysis und die API liefern diesen Text immer auf Englisch; nur die Web-Oberfläche übersetzt die Anzeige. Verwenden Sie für Programmentscheidungen `status` und `is_live`, nicht den Wortlaut von `reason`. Ältere gespeicherte Ergebnisse können ohne `reason` vorliegen; Clients können dann einen allgemeinen Hinweis zur abgelehnten Eingabe anzeigen.

```json
{
  "status": "input_rejected",
  "is_live": null,
  "live_score": null,
  "reason": "Insufficient image area around the face for liveness detection. Move the face toward the center, step back from the camera, or use a less tightly cropped image."
}
```

`normal` erkennt nur Gesichter mit bestandener Liveness-Prüfung. `observe` liefert das Ergebnis und setzt die Erkennung fort. Ohne Prüfung fehlt `liveness`. Die drei Kernfelder sind `status`, `is_live` und `live_score`: Bei bestandener Prüfung oder erkanntem Täuschungsversuch gilt `status: ok` mit Wahrheitswert und Score; bei abgelehnter Eingabe gilt `status: input_rejected`, die anderen Werte sind `null`.

`/v1/detect` liefert auch negative Ergebnisse mit HTTP 200. In `normal` liefern Embeddings, Vergleich und Suche HTTP 422 `liveness_fake` oder `liveness_input_rejected` mit `error.details.liveness`; Vergleiche ergänzen `details.side`. Laufzeitfehler liefern HTTP 503 `liveness_unavailable`. Laufzeitfehler brechen den Vorgang sowohl in `normal` als auch in `observe` ab; sie werden nicht als `input_rejected` zurückgegeben.

Für neue Personen und zusätzliche FaceSamples wird Liveness standardmäßig übersprungen: `[inference].liveness_on_registration=false` führt das Modell nicht aus und lässt `liveness` in neuen Samples weg. Mit `true` gilt bei aktiviertem Addon die Richtlinie `normal`/`observe`; Ablehnungen enthalten `reason` und `liveness`. Die Qualitätsprüfung über `review_mode` und die Validierung externer Embeddings bleiben aktiv. `review_mode=off` und `external_trusted` umgehen eine aktivierte Registrierungsprüfung nicht. Requests können diese Startkonfiguration nicht überschreiben. Bereits gespeicherte Ergebnisse bleiben sichtbar.

RTSP unterscheidet `liveness_blocked` von `unknown` und zählt `liveness_blocked_faces`. Blockierte Gesichter erzeugen keine Personen-/Unbekannt-Eintrittsereignisse und setzen die Bestätigungsfolge zurück. Bei Inferenzfehlern werden alte Identitätsanzeigen gelöscht.

`liveness_compare_scope` legt die geprüften Seiten von `/v1/compare` fest: `both` (Standard) für beide, `source` für das Quellbild, `target` für das Zielbild. Als lebend gilt ein Gesicht bei `live_score >= liveness_threshold`.

`models addons install liveness` speichert das veröffentlichte Modell unter `/models/addons/liveness.onnx`, auf dem Compose-Host unter `server/.models/addons/liveness.onnx`. Startfehler sind `addon_model_missing` und `addon_model_invalid`. `/v1/models` und `/v1/system` melden aktive Zusatzmodelle unter `addons`.

[Konfiguration und Arbeitsabläufe](user-guide.de.md#optionales-liveness-addon).

## System

### `GET /v1/health`

**Zweck/Eingabe:** Öffentliche Readiness, keine Parameter und keine Authentifizierung. **Ergebnis:** Prüft Start und SQLite quick_check; 200 mit `status`, `auth_enabled`, `request_id`. **Fehler:** `503 not_ready`.

### `GET /v1/system`

**Zweck/Eingabe:** Sichere Betriebsdiagnose, keine Parameter. **Ergebnis:** 200 mit OS/CPU/GPU, Driver, CUDA/cuDNN/ORT, Provider, Modell, DB, Mounts, Zählern, Suchbackend, sicherer Konfiguration und Inferenz-Parallelität; keine Secrets/Bilder/Embeddings. **Fehler:** 401, 503.

### `GET /v1/models`

`addons` meldet aktive Addons getrennt vom Basismodell. Prüfen Sie den Eintrag `liveness` und die wirksamen Einstellungen in `safe_config` der Systemantwort. Diese Endpunkte sind schreibgeschützt und installieren keine Modelle.

**Zweck/Eingabe:** Verifizierte Detector-/Recognizer-Modelle, Provider und Lizenz lesen; keine Parameter. **Ergebnis:** 200 `models`, `execution_provider`, `license`. **Fehler:** 401.

Die Basispakete `raccoon_s` und `raccoon_l` unterstützen CPU und CUDA und werden vor dem Start mit dem Modellwerkzeug installiert. Dieser Endpunkt zeigt laufende Modellkomponenten, keinen Downloadkatalog. Die folgende Web-Aktion verwaltet nur Liveness. Collections sind an Erkennungsmodell und Vorverarbeitung gebunden: Ein anderes Basispaket konvertiert vorhandene Merkmale nicht und kann `409 collection_model_mismatch` auslösen. Das alleinige Aktivieren von Liveness ändert diesen Vertrag nicht.

### `GET /v1/addons/liveness`

**Verwendung:** Installationszustand und Einstellungen für den nächsten Start lesen, ohne Download oder Änderung. Dies ist eine Verwaltungs-API, keine eigenständige Liveness-Inferenz.

**Ergebnis:** HTTP 200. `enabled` beschreibt den laufenden Prozess. `installed` bedeutet, dass die Datei die veröffentlichte SHA-256-Prüfung besteht; es bedeutet keine Aktivierung. `configured_enabled` liest die Auswahl für den nächsten Start aus der aktuellen Konfigurationsdatei. `restart_required` zeigt eine Abweichung von `enabled`. Bis zum Neustart beschreibt `safe_config` aus `/v1/system` weiterhin den laufenden Prozess.

`state` ist `idle` (kein geprüftes Modell), `downloading` (Vorbereitung läuft), `ready` (geprüftes Modell vorhanden) oder `error` (Vorbereitungs-, Datei- oder Konfigurationsfehler). `ready` allein bestätigt weder eine gespeicherte Aktivierung noch einen abgeschlossenen Neustart.

`can_enable` zeigt, ob die Web-Vorbereitung verfügbar ist. Andernfalls liefern `unavailable_code` einen stabilen Grundcode und `unavailable_reason` eine Erklärung; sonst sind beide `null`. `error` ist `null` oder ein Objekt mit `code` und `message`. `model_path` ist der lokale Modellpfad; `config_file` der gewählte TOML-Pfad oder `null`. Die Antwort enthält außerdem `request_id`.

Mögliche Werte von `unavailable_code`: `config_file_missing` (keine Konfigurationsdatei gewählt), `config_file_not_regular` (keine reguläre Datei), `config_file_mount` (einzeln eingebundene Datei), `config_not_writable` (Konfiguration nicht schreibbar), `addon_directory_not_writable` (Addon-Verzeichnis nicht schreibbar), `addon_config_invalid` (ungültige Konfiguration), `addon_model_invalid` (ungültiges Modell) und `server_stopping` (Server fährt herunter).

```json
{
  "enabled": false,
  "installed": true,
  "configured_enabled": true,
  "restart_required": true,
  "can_enable": true,
  "unavailable_code": null,
  "unavailable_reason": null,
  "state": "ready",
  "error": null,
  "model_path": "/models/addons/liveness.onnx",
  "config_file": "/etc/insightface/server.toml",
  "request_id": "3ed21e89-4595-4eed-a699-1df42ca62032"
}
```

### `POST /v1/addons/liveness/enable`

**Verwendung:** Liveness herunterladen und für den nächsten Start konfigurieren. Ein leeres JSON-Objekt `{}` mit `Content-Type: application/json` senden. Modell-URLs oder andere Parameter werden nicht akzeptiert.

```bash
curl -sS "${BASE_URL}/v1/addons/liveness" -H "${AUTH_HEADER}"
curl -sS "${BASE_URL}/v1/addons/liveness/enable" -H "${AUTH_HEADER}" \
  -H 'Content-Type: application/json' -d '{}'
```

**Ergebnis:** HTTP 202 liefert dieselben Statusfelder wie GET und bestätigt die Annahme des Auftrags, keine Aktivierung. `GET /v1/addons/liveness` bis zum Abschluss abfragen. Wiederholte Anfragen teilen den laufenden Auftrag; das Schließen des Browsers bricht ihn nicht ab.

Erst nach Download und SHA-256-Prüfung wird `liveness` in `[inference].addons` und `[addons].auto_download` der `config_file` ergänzt. Andere Werte und Kommentare bleiben erhalten, geprüfte Dateien werden wiederverwendet. Bei `installed=true`, `configured_enabled=true` und `restart_required=true` den Server manuell neu starten. Danach meldet der neue Prozess `enabled=true` und `restart_required=false`. Es gibt weder Nachladen im laufenden Betrieb noch eine API zum Wechsel des Basismodells.

**Fehler:** Anfragefehler verwenden das normale Fehlerformat: `400 invalid_addon_request` bei einem anderen Inhalt als `{}`, `401 unauthorized` bei fehlender Authentifizierung, `403 origin_not_allowed` bei unzulässiger Browserherkunft, `409 addon_management_unavailable` bei ungeeigneten Pfaden, Rechten oder Einstellungen und `415 json_required` bei einem anderen Inhaltstyp als JSON. Browser müssen dieselbe oder eine ausdrücklich per CORS erlaubte Herkunft verwenden.

Ein angenommener Auftrag kann später fehlschlagen: GET liefert weiterhin HTTP 200, aber `state=error` und `error.code`. `addon_download_failed` lässt die Konfiguration unverändert; Netzwerk/Proxy des Servers prüfen. Bei `addon_config_save_failed` Konfiguration oder Verzeichnisrechte korrigieren; geprüfte Downloads bleiben nutzbar. `addon_config_invalid` bezeichnet ungültiges TOML auf dem Datenträger. Bei `addon_model_invalid` die beschädigte Datei ersetzen oder entfernen; sie wird nie still überschrieben. Bei `addon_job_in_progress` arbeitet ein anderer Prozess: warten und aktualisieren. Erst nach Behebung der Ursache POST erneut senden.

## Zustandslose Bildoperationen

### `POST /v1/detect`

Bei aktivierter Prüfung enthält jedes bewertete Gesicht `liveness.status`, `liveness.is_live` und `liveness.live_score`. Auch erkannte Täuschungsversuche und `input_rejected` liefern HTTP 200, ohne Erkennungsmerkmale zu extrahieren. `input_rejected` bedeutet, dass um das Gesicht zu wenig Bildfläche vorhanden ist; `liveness.reason` erklärt, wie sich die Aufnahme verbessern lässt. Fehlt `liveness`, wurde es nicht bewertet.

**Eingabe:** multipart `image` erforderlich, `max_faces` 1–100, optional `collection_id`. **Verhalten/Ergebnis:** Mehrfachauflösung, gemeinsame NMS, Flächensortierung; 200 `faces` mit Boxen/5 Punkten/Score/Qualität und `processing_ms`. Kein Gesicht ist eine leere Erfolgsliste. **Fehler:** 400 alter min_score, 404 Collection, 413, 422 invalid_image, 503.

```bash
curl -sS "${BASE_URL}/v1/detect" -H "${AUTH_HEADER}" -F 'image=@group.jpg' -F 'max_faces=10'
```

### `POST /v1/compare`

`liveness_compare_scope` (`both`, `source`, `target`) bestimmt die vor der Erkennung geprüften Seiten. `normal` liefert bei Ablehnung HTTP 422 `liveness_fake` / `liveness_input_rejected`, `error.details.liveness` und `error.details.side`, ohne Ähnlichkeit. `observe` vergleicht weiter und liefert Ergebnisse an den bewerteten Gesichtern.

**Eingabe:** multipart `source`, `target`, optional `threshold` 0–1 und `collection_id`. **Ergebnis:** Wählt je ein Gesicht und liefert 200 `matched`, Cosinus-`similarity`, effektiven threshold, beide Faces und Laufzeit. **Fehler:** 404, 413, 422 invalid_image/face_not_found, 503.

### `POST /v1/embeddings`

Mit Liveness in `normal` führen Täuschungsversuche oder ungeeignete Eingaben zu HTTP 422 `liveness_fake` / `liveness_input_rejected` und `error.details.liveness`; es wird kein Embedding extrahiert. `observe` liefert Embedding und Liveness am Gesicht.

**Eingabe:** multipart `image`, optional `collection_id`. **Ergebnis:** 200 mit ausgewähltem Face, L2-normalisiertem Embedding, Modell und Laufzeit. Für normale Registrierung nicht nötig; Embeddings werden nicht geloggt. **Fehler:** 400 alter face_selection, 404, 413, 422, 503.

## Collections

### `POST /v1/collections`

**Eingabe:** JSON `id` und `name`; optional description, threshold (Standard 0.4), metadata, save_face_crops, `detection` und `search` mit profile/capacity/max_faces_per_person/load_policy. **Verhalten/Ergebnis:** Bindet Modell, Vorverarbeitung und Suchvertrag; 201 mit vollständig aufgelöster `collection`. **Fehler:** 400 Konfiguration, 409 exists, 503 Index.

```bash
curl -sS "${BASE_URL}/v1/collections" -H "${AUTH_HEADER}" -H 'Content-Type: application/json' -d '{"id":"employees","name":"Employees","threshold":0.4}'
```

### `GET /v1/collections`

**Eingabe:** query `limit` 1–100 (Standard 50), optionaler Cursor. **Ergebnis:** 200 `collections`, nullable `next_cursor`. **Fehler:** 400 invalid_cursor, 401.

### `GET /v1/collections/{collection_id}`

**Eingabe:** Collection-ID im Pfad. **Ergebnis:** 200 `collection`, Person-/Face-Zähler, `embedding_contract_id`. **Fehler:** 404.

### `PATCH /v1/collections/{collection_id}`

**Eingabe:** Pfad-ID; JSON name/description/threshold/metadata/save_face_crops, Suchkapazität/max/load und Detection. Null, unbekannte Felder, Modellbindung und search profile sind nicht änderbar. **Ergebnis:** 200 vollständige Collection; neue Detection gilt ab dem nächsten Request. **Fehler:** 400, 404, 409, 503.

### `DELETE /v1/collections/{collection_id}`

**Eingabe:** Pfad-ID, query `force=false`; für nicht leere Collection true. **Ergebnis:** 204 ohne Body. **Fehler:** 404, 409 collection_not_empty, 503.

## Personen und FaceSamples

### `POST /v1/collections/{collection_id}/persons`

Neue Personen und zusätzliche FaceSamples überspringen Liveness standardmäßig (`liveness_on_registration=false`). Bei aktivierter Prüfung lehnt `normal` Täuschungsversuche oder ungeeignete Eingaben ab; `observe` speichert das Ergebnis und fährt fort. Die Qualitätsprüfung folgt dem gewählten `review_mode`. Ablehnungen zeigen den tatsächlichen `reason` und das Liveness-Ergebnis getrennt.

**Eingabe:** Pfad Collection; multipart wiederholtes `images`, optional id/name/external_id, JSON-String metadata, `review_mode=off|standard|strict`, `embedding_mode=server|external_trusted`; extern benötigt Vektoren und contract ID. **Verhalten/Ergebnis:** Prüft jedes Bild; 201 `person`, akzeptierte `faces`, `rejected_images`; Teilerfolg erlaubt, alle abgelehnt ergibt 422 ohne Person. **Fehler:** 400, 404, 409 ID/Vertrag/Kapazität, 413, 422, 503.

```bash
curl -sS "${BASE_URL}/v1/collections/employees/persons" -H "${AUTH_HEADER}" -F 'id=alice' -F 'review_mode=off' -F 'images=@alice.jpg'
```

### `GET /v1/collections/{collection_id}/persons`

**Eingabe:** Collection-ID, query limit/cursor/`search` über ID, Name oder externe ID. **Ergebnis:** 200 `persons`, `next_cursor`. **Fehler:** 400 cursor, 404.

### `GET /v1/collections/{collection_id}/persons/{person_id}`

**Eingabe:** Collection- und Person-ID. **Ergebnis:** 200 `person` mit face_count. **Fehler:** 404.

### `PATCH /v1/collections/{collection_id}/persons/{person_id}`

**Eingabe:** Pfad-IDs; JSON name/external_id/object metadata. **Ergebnis:** 200 aktualisierte Person. **Fehler:** 400, 404, 409 external_id_exists.

### `DELETE /v1/collections/{collection_id}/persons/{person_id}`

**Eingabe:** Pfad-IDs. **Ergebnis:** Löscht Person, alle FaceSamples, Embeddings und Crops, synchronisiert Index, 204. **Fehler:** 404, 503.

### `POST /v1/collections/{collection_id}/persons/{person_id}/faces`

Neue Personen und zusätzliche FaceSamples überspringen Liveness standardmäßig (`liveness_on_registration=false`). Bei aktivierter Prüfung lehnt `normal` Täuschungsversuche oder ungeeignete Eingaben ab; `observe` speichert das Ergebnis und fährt fort. Die Qualitätsprüfung folgt dem gewählten `review_mode`. Ablehnungen zeigen den tatsächlichen `reason` und das Liveness-Ergebnis getrennt.

**Eingabe:** Pfad-IDs; wiederholte images und dieselben Review-/Embedding-Felder wie Person-Erstellung. **Ergebnis:** 201 `faces`, `rejected_images`, Teilerfolg möglich. **Fehler:** Registrierungsfehler plus 404 Person.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces`

**Eingabe:** Pfad-IDs; query limit 1–100 und Cursor. **Ergebnis:** 200 Face-Metadaten, `has_crop`, `next_cursor`, ohne Embedding/Bildbytes. **Fehler:** 400 cursor, 404.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}/image`

**Eingabe:** drei Pfad-IDs. **Ergebnis:** Falls gespeichert 200 `image/jpeg`, 112×112 Crop, `Cache-Control:no-store`; Request-ID nur im Header. **Fehler:** 401, 404 Face/face_image_not_found.

### `DELETE /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}`

**Eingabe:** drei Pfad-IDs. **Ergebnis:** Löscht Embedding/Crop/Indexzeile, 204. **Fehler:** 404, 503.

## Suche

### `POST /v1/collections/{collection_id}/search`

Mit Liveness in `normal` führen Täuschungsversuche oder ungeeignete Suchbilder zu HTTP 422 `liveness_fake` / `liveness_input_rejected` und `error.details.liveness`; die Suche läuft nicht. Das unterscheidet sich von einer erfolgreichen leeren Trefferliste. `observe` sucht weiter und liefert Liveness am Suchgesicht.

**Eingabe:** Collection-ID; multipart `image`, `limit` 1–100 (Standard 5), optional threshold, sonst Collection-Wert. **Verhalten/Ergebnis:** Sucht das gewählte Face über alle Samples, pro Person gilt der höchste Wert; 200 `searched_face`, sortierte `matches`, threshold und Laufzeit. Kein Treffer ist leere Liste. **Fehler:** 404, 409 Modell, 413, 422 Bild/Gesicht, 503 Index/Timeout.

```bash
curl -sS "${BASE_URL}/v1/collections/employees/search" -H "${AUTH_HEADER}" -F 'image=@query.jpg' -F 'limit=5'
```

## RTSP-Monitore

Monitor-Konfigurationen liegen dauerhaft in SQLite; aktivierte Aufgaben werden nach einem Server-Neustart wiederhergestellt. Videobilder werden nicht gespeichert, Ereignisse nur in einem begrenzten RAM-Ringpuffer.

### `POST /v1/monitors`

**Zweck:** Einen dauerhaften RTSP-Erkennungsmonitor anlegen. **Eingabe:** JSON mit ID, Name, `source`, Collection, `inference_fps` (Standard 2), optionalem Schwellwert, Puffer/Event-Regeln und `preview_enabled` (Standard false). **Ergebnis:** 201 mit redigiertem `monitor`; Zugangsdaten werden verschlüsselt gespeichert. **Fehler:** 400, 404, 409, 429.

### `GET /v1/monitors`

**Zweck:** Monitore seitenweise auflisten. **Eingabe:** `limit` 1–100 (Standard 50) und der unveränderte, undurchsichtige `cursor` der vorigen Antwort. **Ergebnis:** 200 mit `monitors` und `next_cursor`, niemals mit Zugangsdaten. **Fehler:** 400 `invalid_cursor`, 401.

### `GET /v1/monitors/{monitor_id}`

**Zweck:** Konfiguration und Laufzeitübersicht eines Monitors lesen. **Eingabe:** `monitor_id` im Pfad. **Ergebnis:** 200 mit Event-Regeln, redigierter Quelle, Preview-Einstellung und Status. **Fehler:** 401, 404 `monitor_not_found`.

### `PATCH /v1/monitors/{monitor_id}`

**Zweck:** Alle veränderbaren Felder teilweise ändern und über `enabled` starten/stoppen. **Eingabe:** JSON-Teilobjekt; auch `event_policy` ist partiell, ein null-Schwellwert erbt den Collection-Wert. **Ergebnis:** 200 mit vollständigem Monitor; Quelle/Collection/Rate/Regeln starten die Aufgabe neu. **Fehler:** 400, 404, 429.

### `DELETE /v1/monitors/{monitor_id}`

**Zweck:** Einen Monitor dauerhaft entfernen. **Eingabe:** `monitor_id` im Pfad. **Ergebnis:** Decoder, Inferenz und RTSP-Verbindung werden gestoppt, RAM-Ereignisse verworfen, HTTP 204; die Collection bleibt bestehen. **Fehler:** 401, 404.

### `GET /v1/monitors/{monitor_id}/state`

Mit Liveness in `normal` erhalten blockierte Gesichter `status: liveness_blocked` und ein separates Liveness-Ergebnis. Sie zählen zu `liveness_blocked_faces`, nicht zu `unknown_faces`, und erzeugen keine Eintrittsereignisse. `observe` erkennt weiter. Eingabeablehnung und Täuschungsversuch werden getrennt angezeigt.

**Zweck:** Den aktuellen Zustand für Headless-Clients abfragen. **Eingabe:** Monitor-ID im Pfad. **Ergebnis:** 200 mit Verbindung, effektiver FPS, Laufzeit, übersprungenen Frames, aktuellen Treffern/unbekannten Gesichtern, Preview, Reconnects und Fehlern, ohne Embeddings. **Fehler:** 401, 404.

### `GET /v1/monitors/{monitor_id}/events`

**Zweck:** Flüchtige Enter/Exit/Error/Recovery-Ereignisse abrufen. **Eingabe:** `limit` 1–1000 und der letzte undurchsichtige `cursor`. **Ergebnis:** 200 mit `events`, `next_cursor`, `truncated` und `stream_reset`; Neustarts löschen Ereignisse. **Fehler:** 400 `invalid_cursor`, 401, 404.

### `GET /v1/monitors/{monitor_id}/preview.mjpeg`

**Zweck:** Die standardmäßig deaktivierte rohe MJPEG-Vorschau öffnen. **Eingabe:** Pfad-ID und normaler Bearer-Header; kein API-Key in der URL. **Ergebnis:** Lang laufendes `multipart/x-mixed-replace`, nur bei Zuschauern codiert; Boxen zeichnet der Client aus `/state`. **Fehler:** 401, 404, 409 `preview_disabled`, 503.

## Retry-Regel

GET darf wiederholt werden. DELETE erst nach Statusprüfung. Bei unklarem Netzwerkergebnis einer Person-/Face-Erstellung vor erneutem POST per ID lesen. Nur 429 und temporäre 503 mit begrenztem exponentiellem Backoff plus Jitter wiederholen; 4xx-Eingaben korrigieren.
