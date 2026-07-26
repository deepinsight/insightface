# Guide d’utilisation de l’API REST InsightFace Server

**Langues :** [English](api.md) · [中文](api.zh-CN.md) · [日本語](api.ja.md) · [Deutsch](api.de.md) · [Español](api.es.md) · Français · [Русский](api.ru.md) · [Português](api.pt.md) · [한국어](api.ko.md)

Ce guide décrit l’objectif, les entrées, le traitement serveur, le résultat et les erreurs de chaque route publique. Pour installer et effectuer une première recherche, consultez le [guide utilisateur](user-guide.fr.md). Le schéma exact de l’instance est disponible sous `/docs` et `/openapi.json`.

## Règles communes

- Base `/v1`, JSON en `snake_case`, images JPEG/PNG/WebP en multipart.
- Le Compose fourni désactive l’authentification pour une évaluation isolée. Lorsqu’elle est active, tout sauf health exige `Authorization: Bearer <api_key>` ; sinon omettez totalement cet en-tête.
- Chaque réponse porte `x-request-id`, répété dans le JSON par `request_id`.
- confidence/quality/threshold utilisent `0..1`. Similarity n’est pas une probabilité : cosine brut `[-1,1]`, seuil par défaut `0.4`, correspondance si `similarity >= threshold`.
- Un cursor est opaque et ne se réutilise, inchangé, qu’avec la même route, Collection, Person et filtre.
- Codes usuels : 400 entrée, 401 auth, 404 absent, 409 conflit, 413 taille, 422 image/visage, 429 limite, 503 timeout/modèle/index.

```bash
BASE_URL=http://127.0.0.1:18097
AUTH_HEADER="Authorization: Bearer ${INSIGHTFACE_API_KEY}"
curl -fsS "${BASE_URL}/v1/health"
```

## Système

### `GET /v1/health`

**Usage/entrée :** readiness publique, sans paramètre ni auth. **Résultat :** vérifie démarrage et SQLite quick_check ; 200 avec `status`, `auth_enabled`, `request_id`. **Erreur :** `503 not_ready`.

### `GET /v1/system`

**Usage/entrée :** diagnostic sûr, sans paramètre. **Résultat :** 200 avec OS/CPU/GPU, Driver, CUDA/cuDNN/ORT, Provider, modèle, DB, montages, compteurs, recherche, configuration sûre et concurrence ; jamais de secrets, images ou embeddings. **Erreurs :** 401, 503.

### `GET /v1/models`

**Usage/entrée :** modèles detector/recognizer vérifiés, Provider et licence ; sans paramètre. **Résultat :** 200 `models`, `execution_provider`, `license`. **Erreur :** 401.

## Opérations faciales sans état

### `POST /v1/detect`

**Entrée :** multipart `image` requis, `max_faces` 1–100, `collection_id` facultatif. **Traitement/résultat :** fusion multi-résolution, NMS globale, tri par aire ; 200 `faces` avec boîtes/5 points/score/qualité et `processing_ms`. Aucun visage donne une liste vide valide. **Erreurs :** 400 ancien min_score, 404 Collection, 413, 422 invalid_image, 503.

```bash
curl -sS "${BASE_URL}/v1/detect" -H "${AUTH_HEADER}" -F 'image=@group.jpg' -F 'max_faces=10'
```

### `POST /v1/compare`

**Entrée :** multipart `source`, `target`, `threshold` facultatif 0–1 et `collection_id`. **Résultat :** choisit un visage par image ; 200 `matched`, cosine `similarity`, seuil effectif, deux visages et durée. **Erreurs :** 404, 413, 422 invalid_image/face_not_found, 503.

### `POST /v1/embeddings`

**Entrée :** multipart `image`, `collection_id` facultatif. **Résultat :** 200 avec visage choisi, embedding L2, modèle et durée. Inutile à l’inscription normale ; le vecteur n’est pas journalisé. **Erreurs :** 400 ancien face_selection, 404, 413, 422, 503.

## Collections

### `POST /v1/collections`

**Entrée :** JSON `id`, `name`; facultatifs description, threshold (0.4), metadata, save_face_crops, `detection`, `search` avec profile/capacity/max_faces_per_person/load_policy. **Traitement/résultat :** fixe modèle, prétraitement et contrat de recherche ; 201 avec `collection` résolue. **Erreurs :** 400 configuration, 409 exists, 503 index.

```bash
curl -sS "${BASE_URL}/v1/collections" -H "${AUTH_HEADER}" -H 'Content-Type: application/json' -d '{"id":"employees","name":"Employees","threshold":0.4}'
```

### `GET /v1/collections`

**Entrée :** query `limit` 1–100 (50), cursor facultatif. **Résultat :** 200 `collections`, `next_cursor` nullable. **Erreurs :** 400 invalid_cursor, 401.

### `GET /v1/collections/{collection_id}`

**Entrée :** ID Collection dans le chemin. **Résultat :** 200 `collection`, compteurs Person/Face et `embedding_contract_id`. **Erreur :** 404.

### `PATCH /v1/collections/{collection_id}`

**Entrée :** ID ; JSON name/description/threshold/metadata/save_face_crops, capacité/max/load de recherche et detection. Null, champs inconnus, modèle et search profile sont immuables. **Résultat :** 200 Collection complète ; detection s’applique à la requête suivante. **Erreurs :** 400, 404, 409, 503.

### `DELETE /v1/collections/{collection_id}`

**Entrée :** ID ; query `force=false`, true si non vide. **Résultat :** 204 sans corps. **Erreurs :** 404, 409 collection_not_empty, 503.

## Personnes et FaceSamples

### `POST /v1/collections/{collection_id}/persons`

**Entrée :** Collection ; multipart `images` répétable, id/name/external_id facultatifs, metadata JSON texte, `review_mode=off|standard|strict`, `embedding_mode=server|external_trusted`; le mode externe ajoute vecteurs et contract ID. **Traitement/résultat :** contrôle chaque image ; 201 `person`, `faces` acceptées et `rejected_images`, succès partiel autorisé ; tout rejeté donne 422 sans Person. **Erreurs :** 400, 404, 409 ID/contrat/capacité, 413, 422, 503.

```bash
curl -sS "${BASE_URL}/v1/collections/employees/persons" -H "${AUTH_HEADER}" -F 'id=alice' -F 'review_mode=off' -F 'images=@alice.jpg'
```

### `GET /v1/collections/{collection_id}/persons`

**Entrée :** Collection ; query limit/cursor/`search` sur ID, nom ou ID externe. **Résultat :** 200 `persons`, `next_cursor`. **Erreurs :** 400 cursor, 404.

### `GET /v1/collections/{collection_id}/persons/{person_id}`

**Entrée :** IDs Collection et Person. **Résultat :** 200 `person` avec face_count. **Erreur :** 404.

### `PATCH /v1/collections/{collection_id}/persons/{person_id}`

**Entrée :** IDs ; JSON name/external_id/metadata objet. **Résultat :** 200 Person mise à jour. **Erreurs :** 400, 404, 409 external_id_exists.

### `DELETE /v1/collections/{collection_id}/persons/{person_id}`

**Entrée :** IDs. **Résultat :** supprime Person, FaceSamples, embeddings et crops, synchronise l’index, 204. **Erreurs :** 404, 503.

### `POST /v1/collections/{collection_id}/persons/{person_id}/faces`

**Entrée :** IDs ; images répétables et mêmes champs review/embedding que la création. **Résultat :** 201 `faces`, `rejected_images`, succès partiel. **Erreurs :** inscription plus 404 Person.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces`

**Entrée :** IDs ; query limit 1–100 et cursor. **Résultat :** 200 métadonnées `faces`, `has_crop`, `next_cursor`, sans embedding ni octets. **Erreurs :** 400 cursor, 404.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}/image`

**Entrée :** trois IDs. **Résultat :** si conservé, 200 `image/jpeg`, crop 112×112, `Cache-Control:no-store`; request ID dans l’en-tête seulement. **Erreurs :** 401, 404 face/face_image_not_found.

### `DELETE /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}`

**Entrée :** trois IDs. **Résultat :** supprime embedding/crop/ligne d’index, 204. **Erreurs :** 404, 503.

## Recherche

### `POST /v1/collections/{collection_id}/search`

**Entrée :** Collection ; multipart `image`, `limit` 1–100 (5), threshold facultatif ou valeur Collection. **Traitement/résultat :** compare le visage choisi à tous les samples et garde le maximum par Person ; 200 `searched_face`, `matches` triés, seuil et durée. Aucun match est une liste vide. **Erreurs :** 404, 409 modèle, 413, 422 image/visage, 503 index/timeout.

```bash
curl -sS "${BASE_URL}/v1/collections/employees/search" -H "${AUTH_HEADER}" -F 'image=@query.jpg' -F 'limit=5'
```

## Moniteurs RTSP

La configuration d’un Monitor persiste dans SQLite et toute tâche activée est restaurée au redémarrage du serveur. Les images ne sont pas enregistrées ; les événements restent dans un tampon circulaire borné en mémoire.

### `POST /v1/monitors`

**Usage :** Créer un Monitor RTSP persistant. **Entrée :** JSON avec ID, nom, `source`, Collection, `inference_fps` (2), seuil facultatif, tampon/politique d’événements et `preview_enabled` (false). **Résultat :** 201 avec `monitor` masqué ; les identifiants sont chiffrés au repos. **Erreurs :** 400, 404, 409, 429.

### `GET /v1/monitors`

**Usage :** Lister les Monitors avec pagination. **Entrée :** `limit` 1–100 (50) et le `cursor` opaque de la réponse précédente, inchangé. **Résultat :** 200 avec `monitors` et `next_cursor`, jamais les identifiants. **Erreurs :** 400 `invalid_cursor`, 401.

### `GET /v1/monitors/{monitor_id}`

**Usage :** Lire la configuration et le résumé d’exécution d’un Monitor. **Entrée :** `monitor_id` dans le chemin. **Résultat :** 200 avec politique d’événements, source masquée, aperçu et état. **Erreurs :** 401, 404 `monitor_not_found`.

### `PATCH /v1/monitors/{monitor_id}`

**Usage :** Modifier partiellement sauf l’ID et démarrer/arrêter via `enabled`. **Entrée :** JSON partiel ; `event_policy` est aussi partiel et un seuil null hérite de la Collection. **Résultat :** 200 avec le Monitor complet ; source/Collection/fréquence/politique relancent la tâche. **Erreurs :** 400, 404, 429.

### `DELETE /v1/monitors/{monitor_id}`

**Usage :** Supprimer définitivement un Monitor. **Entrée :** `monitor_id` dans le chemin. **Résultat :** arrête décodage, inférence et RTSP, efface les événements mémoire puis renvoie 204 ; la Collection reste. **Erreurs :** 401, 404.

### `GET /v1/monitors/{monitor_id}/state`

**Usage :** Interroger l’état courant depuis un client sans interface. **Entrée :** ID du Monitor. **Résultat :** 200 avec connexion, FPS effectif, latence, images sautées, visages reconnus/inconnus, aperçu, reconnexions et erreur sûre, sans embeddings. **Erreurs :** 401, 404.

### `GET /v1/monitors/{monitor_id}/events`

**Usage :** Lire les événements volatils entrée/sortie/erreur/rétablissement. **Entrée :** `limit` 1–1000 et le dernier `cursor` opaque. **Résultat :** 200 avec `events`, `next_cursor`, `truncated` et `stream_reset` ; un redémarrage perd les événements. **Erreurs :** 400 `invalid_cursor`, 401, 404.

### `GET /v1/monitors/{monitor_id}/preview.mjpeg`

**Usage :** Ouvrir l’aperçu MJPEG brut, désactivé par défaut. **Entrée :** ID et en-tête Bearer normal ; jamais la clé dans l’URL. **Résultat :** long `multipart/x-mixed-replace`, encodé seulement avec spectateurs ; le client trace les cadres via `/state`. **Erreurs :** 401, 404, 409 `preview_disabled`, 503.

## Réessais

GET peut être réessayé. Vérifiez l’état avant de répéter DELETE. Si une création Person/Face a un résultat réseau incertain, lisez l’ID avant un nouveau POST. Réessayez seulement 429 et 503 transitoires avec backoff exponentiel borné et jitter ; corrigez les 4xx.
