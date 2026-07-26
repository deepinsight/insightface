# InsightFace Server REST-API-Leitfaden

**Sprachen:** [English](api.md) · [中文](api.zh-CN.md) · [日本語](api.ja.md) · Deutsch · [Español](api.es.md) · [Français](api.fr.md) · [Русский](api.ru.md) · [Português](api.pt.md) · [한국어](api.ko.md)

Dieses Dokument erklärt Zweck, Eingaben, Serververhalten, Erfolg und typische Fehler aller öffentlichen Endpunkte. Start und Modellinstallation stehen im [Benutzerhandbuch](user-guide.de.md); das exakte laufende Schema finden Sie unter `/docs` und `/openapi.json`.

## Gemeinsame Regeln

- Basispfad `/v1`, JSON in `snake_case`, Bilder als JPEG/PNG/WebP multipart.
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

## System

### `GET /v1/health`

**Zweck/Eingabe:** Öffentliche Readiness, keine Parameter und keine Authentifizierung. **Ergebnis:** Prüft Start und SQLite quick_check; 200 mit `status`, `auth_enabled`, `request_id`. **Fehler:** `503 not_ready`.

### `GET /v1/system`

**Zweck/Eingabe:** Sichere Betriebsdiagnose, keine Parameter. **Ergebnis:** 200 mit OS/CPU/GPU, Driver, CUDA/cuDNN/ORT, Provider, Modell, DB, Mounts, Zählern, Suchbackend, sicherer Konfiguration und Inferenz-Parallelität; keine Secrets/Bilder/Embeddings. **Fehler:** 401, 503.

### `GET /v1/models`

**Zweck/Eingabe:** Verifizierte Detector-/Recognizer-Modelle, Provider und Lizenz lesen; keine Parameter. **Ergebnis:** 200 `models`, `execution_provider`, `license`. **Fehler:** 401.

## Zustandslose Bildoperationen

### `POST /v1/detect`

**Eingabe:** multipart `image` erforderlich, `max_faces` 1–100, optional `collection_id`. **Verhalten/Ergebnis:** Mehrfachauflösung, gemeinsame NMS, Flächensortierung; 200 `faces` mit Boxen/5 Punkten/Score/Qualität und `processing_ms`. Kein Gesicht ist eine leere Erfolgsliste. **Fehler:** 400 alter min_score, 404 Collection, 413, 422 invalid_image, 503.

```bash
curl -sS "${BASE_URL}/v1/detect" -H "${AUTH_HEADER}" -F 'image=@group.jpg' -F 'max_faces=10'
```

### `POST /v1/compare`

**Eingabe:** multipart `source`, `target`, optional `threshold` 0–1 und `collection_id`. **Ergebnis:** Wählt je ein Gesicht und liefert 200 `matched`, Cosinus-`similarity`, effektiven threshold, beide Faces und Laufzeit. **Fehler:** 404, 413, 422 invalid_image/face_not_found, 503.

### `POST /v1/embeddings`

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

**Eingabe:** Pfad-IDs; wiederholte images und dieselben Review-/Embedding-Felder wie Person-Erstellung. **Ergebnis:** 201 `faces`, `rejected_images`, Teilerfolg möglich. **Fehler:** Registrierungsfehler plus 404 Person.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces`

**Eingabe:** Pfad-IDs; query limit 1–100 und Cursor. **Ergebnis:** 200 Face-Metadaten, `has_crop`, `next_cursor`, ohne Embedding/Bildbytes. **Fehler:** 400 cursor, 404.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}/image`

**Eingabe:** drei Pfad-IDs. **Ergebnis:** Falls gespeichert 200 `image/jpeg`, 112×112 Crop, `Cache-Control:no-store`; Request-ID nur im Header. **Fehler:** 401, 404 Face/face_image_not_found.

### `DELETE /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}`

**Eingabe:** drei Pfad-IDs. **Ergebnis:** Löscht Embedding/Crop/Indexzeile, 204. **Fehler:** 404, 503.

## Suche

### `POST /v1/collections/{collection_id}/search`

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

**Zweck:** Den aktuellen Zustand für Headless-Clients abfragen. **Eingabe:** Monitor-ID im Pfad. **Ergebnis:** 200 mit Verbindung, effektiver FPS, Laufzeit, übersprungenen Frames, aktuellen Treffern/unbekannten Gesichtern, Preview, Reconnects und Fehlern, ohne Embeddings. **Fehler:** 401, 404.

### `GET /v1/monitors/{monitor_id}/events`

**Zweck:** Flüchtige Enter/Exit/Error/Recovery-Ereignisse abrufen. **Eingabe:** `limit` 1–1000 und der letzte undurchsichtige `cursor`. **Ergebnis:** 200 mit `events`, `next_cursor`, `truncated` und `stream_reset`; Neustarts löschen Ereignisse. **Fehler:** 400 `invalid_cursor`, 401, 404.

### `GET /v1/monitors/{monitor_id}/preview.mjpeg`

**Zweck:** Die standardmäßig deaktivierte rohe MJPEG-Vorschau öffnen. **Eingabe:** Pfad-ID und normaler Bearer-Header; kein API-Key in der URL. **Ergebnis:** Lang laufendes `multipart/x-mixed-replace`, nur bei Zuschauern codiert; Boxen zeichnet der Client aus `/state`. **Fehler:** 401, 404, 409 `preview_disabled`, 503.

## Retry-Regel

GET darf wiederholt werden. DELETE erst nach Statusprüfung. Bei unklarem Netzwerkergebnis einer Person-/Face-Erstellung vor erneutem POST per ID lesen. Nur 429 und temporäre 503 mit begrenztem exponentiellem Backoff plus Jitter wiederholen; 4xx-Eingaben korrigieren.
