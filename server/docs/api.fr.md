# Guide d’utilisation de l’API REST InsightFace Server

**Langues :** [English](api.md) · [中文](api.zh-CN.md) · [日本語](api.ja.md) · [Deutsch](api.de.md) · [Español](api.es.md) · Français · [Русский](api.ru.md) · [Português](api.pt.md) · [한국어](api.ko.md)

Ce guide décrit l’objectif, les entrées, le traitement serveur, le résultat et les erreurs de chaque route publique. Pour installer et effectuer une première recherche, consultez le [guide utilisateur](user-guide.fr.md). Le schéma exact de l’instance est disponible sous `/docs` et `/openapi.json`.


Les modèles sont identifiés par `model_id` ; les réponses ne contiennent pas de champ `model_version` distinct. Les Collections existantes conservent leur `embedding_contract_id`.

Pour utiliser la détection du vivant, consultez [configuration, installation et résultats](#addon-optionnel-de-détection-du-vivant). Chaque procédure explique aussi ses effets.

## Règles communes

- Base `/v1`, JSON en `snake_case`, images JPEG/PNG/WebP/BMP en multipart.
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

## Addon optionnel de détection du vivant

La détection du vivant est désactivée par défaut dans `server/config/server.toml` : `inference.addons` et `addons.auto_download` valent `[]`. Les anciennes configurations sans ces clés restent désactivées.

**Activer en ligne de commande, y compris avant le premier démarrage de Server :**

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

`--enable-liveness` vérifie d’abord que la configuration existante peut être mise à jour. Il installe et vérifie le paquet de base, les addons configurés pour l’installation et la détection du vivant, puis ajoute `liveness` à `inference.addons` et `addons.auto_download`, en conservant les autres entrées, commentaires et paramètres. Les caches vérifiés sont réutilisés, mais l’activation est enregistrée même si les modèles sont déjà en cache. Un échec de téléchargement ne modifie pas la configuration ; un échec d’enregistrement produit une erreur explicite et un code de sortie non nul. Les fichiers valides en cache restent réutilisables lors d’un nouvel essai.

Les deux services Compose montent tout le répertoire existant `server/config` en écriture dans `/etc/insightface`, avec `create_host_path: false`. L’installateur peut ainsi mettre à jour atomiquement la configuration de l’hôte sans Server en cours d’exécution. Le répertoire et `server.toml` doivent exister.

Server n’a pas besoin de fonctionner. Pour une nouvelle installation, le prochain `up -d` active la détection du vivant ; si Server fonctionne déjà, utilisez `docker compose -f server/deploy/compose.cpu.yml restart server`. `up -d` seul ne recharge pas les paramètres enregistrés. Pour CUDA, utilisez `compose.cuda12.yml`.

Sans `--enable-liveness`, `models install` garde son comportement et n’écrit pas la configuration ; la détection du vivant reste désactivée par défaut. `models addons install liveness` télécharge et vérifie seulement l’addon, sans l’activer. L’activation reste aussi disponible dans **Système → Détection du vivant**, comme indiqué ci-dessous.

Dans **Système → Détection du vivant**, téléchargez le modèle et activez-le pour le prochain démarrage. Après vérification SHA-256, `liveness` est ajouté aux deux listes ; les autres entrées, commentaires et réglages sont conservés. Un fichier déjà vérifié est réutilisé. **Redémarrez manuellement le Server** pour appliquer le changement. Les erreurs permettent de réessayer ; un téléchargement échoué n’active pas la détection.

[Montages et permissions pour les téléchargements Web](user-guide.fr.md#montages-et-permissions-pour-les-téléchargements-web).

**Avancé : configuration manuelle.** Ces paramètres remplacent le paramètre d’activation ou l’action Web ; installez le modèle avant de redémarrer.

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

### Installation du modèle et démarrage

`inference.addons` contrôle l’exécution et `addons.auto_download` le téléchargement complémentaire à l’installation d’un paquet de base. Avec `["liveness"]`, l’addon est installé même si le paquet de base est en cache. Aucun téléchargement au démarrage. L’installateur et le Server lisent le même fichier.

Exécutez les commandes depuis la racine du dépôt avec `server/config/server.toml` présent. Le Compose fourni exécute l’installateur en root, crée le répertoire des modèles si nécessaire et monte `/models` en écriture ; le téléchargement de l’addon crée `addons`. Aucune préparation manuelle des UID/GID ou permissions n’est nécessaire. Consultez la [configuration initiale du guide utilisateur](user-guide.fr.md).

```bash
docker compose -f server/deploy/compose.cpu.yml run --rm models addons install liveness
docker compose -f server/deploy/compose.cpu.yml run --rm models addons verify liveness
docker compose -f server/deploy/compose.cpu.yml up -d --force-recreate
```

Un modèle activé absent arrête le démarrage avec `addon_model_missing` ; un modèle invalide produit `addon_model_invalid`. L’addon n’est pas désactivé silencieusement.

### Résultats de détection du vivant

| Résultat | `status` | `is_live` | `live_score` |
| --- | --- | --- | --- |
| Visage vivant | `ok` | `true` | `[0, 1]` |
| Usurpation | `ok` | `false` | `[0, 1]` |
| Entrée rejetée | `input_rejected` | `null` | `null` |

Seule une surface insuffisante de l’image source autour du visage aligné produit `input_rejected`. Ce résultat ajoute `liveness.reason`, une explication destinée à l’utilisateur ; les résultats de visage vivant ou de falsification omettent `reason`. FaceAnalysis et l’API renvoient toujours ce texte en anglais ; seule l’interface Web traduit son affichage. La logique du programme doit utiliser `status` et `is_live`, sans analyser le texte de `reason`. Les anciens résultats enregistrés peuvent ne pas contenir `reason` ; le client peut alors afficher un message générique de rejet de l’entrée.

```json
{
  "status": "input_rejected",
  "is_live": null,
  "live_score": null,
  "reason": "Insufficient image area around the face for liveness detection. Move the face toward the center, step back from the camera, or use a less tightly cropped image."
}
```

`normal` reconnaît uniquement les visages qui passent le contrôle ; `observe` conserve le résultat et poursuit la reconnaissance. Sans évaluation, `liveness` est omis. Les trois champs principaux sont `status`, `is_live` et `live_score` : les résultats positifs ou d’usurpation utilisent `status: ok`, un booléen et un score ; une entrée rejetée utilise `status: input_rejected` et deux valeurs `null`.

`/v1/detect` renvoie HTTP 200 même pour un résultat négatif. En `normal`, embeddings, comparaison et recherche renvoient HTTP 422 `liveness_fake` ou `liveness_input_rejected` avec `error.details.liveness` ; la comparaison ajoute `details.side`. Une panne renvoie HTTP 503 `liveness_unavailable`. Les erreurs d’exécution interrompent l’opération en `normal` comme en `observe` ; elles ne sont pas converties en `input_rejected`.

La création de personnes et l’ajout de FaceSamples ignorent ce contrôle par défaut : `[inference].liveness_on_registration=false` n’exécute pas le modèle et omet `liveness` dans les nouveaux échantillons. Avec `true` et l’addon activé, la politique `normal`/`observe` s’applique ; les refus comprennent `reason` et `liveness`. La qualité selon `review_mode` et la validation des embeddings externes restent contrôlées. `review_mode=off` et `external_trusted` ne contournent pas un contrôle d’inscription activé. Les requêtes ne peuvent pas modifier cette configuration de démarrage. Les résultats déjà enregistrés restent consultables.

RTSP distingue `liveness_blocked` de `unknown` et compte `liveness_blocked_faces`. Les visages bloqués ne produisent aucun événement d’entrée de personne/inconnu et réinitialisent la confirmation. Une panne d’inférence efface les identités précédemment affichées.

`liveness_compare_scope` choisit les côtés évalués par `/v1/compare` : `both` (par défaut) pour les deux, `source` pour l’image source, `target` pour l’image cible. Le visage est considéré vivant si `live_score >= liveness_threshold`.

`models addons install liveness` enregistre le modèle publié dans `/models/addons/liveness.onnx`, soit `server/.models/addons/liveness.onnx` sur l’hôte Compose. Les erreurs de démarrage sont `addon_model_missing` et `addon_model_invalid`. `/v1/models` et `/v1/system` signalent les compléments actifs dans `addons`.

[Configuration et procédures](user-guide.fr.md#addon-optionnel-de-détection-du-vivant).

## Système

### `GET /v1/health`

**Usage/entrée :** readiness publique, sans paramètre ni auth. **Résultat :** vérifie démarrage et SQLite quick_check ; 200 avec `status`, `auth_enabled`, `request_id`. **Erreur :** `503 not_ready`.

### `GET /v1/system`

**Usage/entrée :** diagnostic sûr, sans paramètre. **Résultat :** 200 avec OS/CPU/GPU, Driver, CUDA/cuDNN/ORT, Provider, modèle, DB, montages, compteurs, recherche, configuration sûre et concurrence ; jamais de secrets, images ou embeddings. **Erreurs :** 401, 503.

### `GET /v1/models`

`addons` indique les addons actifs séparément du modèle de base. Vérifiez `liveness` et les réglages effectifs dans `safe_config` de la réponse système. Ces endpoints sont en lecture seule et n’installent pas de modèle.

**Usage/entrée :** modèles detector/recognizer vérifiés, Provider et licence ; sans paramètre. **Résultat :** 200 `models`, `execution_provider`, `license`. **Erreur :** 401.

Les paquets de base `raccoon_s` et `raccoon_l` fonctionnent sur CPU et CUDA et s’installent avec l’outil de modèles avant le démarrage. Cet endpoint liste les composants en cours d’utilisation, pas un catalogue de téléchargements. L’action Web ci-dessous gère uniquement la détection du vivant. Les Collections sont liées au modèle de reconnaissance et au prétraitement : changer de paquet ne convertit pas les vecteurs existants et peut produire `409 collection_model_mismatch`. Activer uniquement la détection du vivant ne change pas ce contrat.

### `GET /v1/addons/liveness`

**Usage:** Consulter l’installation et les réglages du prochain démarrage sans téléchargement ni modification. Cette API sert à l’administration, pas à une inférence autonome de détection du vivant.

**Résultat:** HTTP 200. `enabled` décrit le processus en cours. `installed` signifie que le fichier passe la vérification SHA-256 publiée, sans indiquer une activation. `configured_enabled` lit le choix du prochain démarrage dans le fichier actuel ; `restart_required` indique une différence avec `enabled`. Jusqu’au redémarrage, `safe_config` de `/v1/system` décrit toujours le processus en cours.

`state` vaut `idle` (aucun modèle vérifié), `downloading` (préparation en cours), `ready` (modèle vérifié disponible) ou `error` (erreur de préparation, de fichier ou de configuration). `ready` seul ne confirme ni l’enregistrement de l’activation ni la fin du redémarrage.

`can_enable` indique si la préparation Web est disponible. Sinon, `unavailable_code` donne un code de motif stable et `unavailable_reason` une explication ; autrement, les deux valent `null`. `error` vaut `null` ou contient `code` et `message`. `model_path` est le chemin local du modèle ; `config_file`, le chemin TOML choisi ou `null`. La réponse contient aussi `request_id`.

Les valeurs de `unavailable_code` sont `config_file_missing` (aucun fichier de configuration choisi), `config_file_not_regular` (fichier non ordinaire), `config_file_mount` (fichier monté seul), `config_not_writable` (configuration non modifiable), `addon_directory_not_writable` (répertoire du complément non modifiable), `addon_config_invalid` (configuration invalide), `addon_model_invalid` (modèle invalide) et `server_stopping` (arrêt en cours).

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

**Usage:** Télécharger le modèle et enregistrer son activation pour le prochain démarrage. Envoyer un objet JSON vide `{}` avec `Content-Type: application/json`. Les URL de modèles et autres paramètres ne sont pas acceptés.

```bash
curl -sS "${BASE_URL}/v1/addons/liveness" -H "${AUTH_HEADER}"
curl -sS "${BASE_URL}/v1/addons/liveness/enable" -H "${AUTH_HEADER}" \
  -H 'Content-Type: application/json' -d '{}'
```

**Résultat:** HTTP 202 renvoie les mêmes champs que GET et confirme l’acceptation du travail, pas l’activation. Interroger `GET /v1/addons/liveness` jusqu’à la fin. Les requêtes répétées partagent le travail en cours ; fermer le navigateur ne l’annule pas.

Après téléchargement et vérification SHA-256 seulement, le travail ajoute `liveness` à `[inference].addons` et `[addons].auto_download` dans `config_file`, en conservant les autres valeurs et commentaires. Un fichier vérifié est réutilisé. Quand `installed=true`, `configured_enabled=true` et `restart_required=true`, redémarrer manuellement le serveur. Le nouveau processus renvoie `enabled=true` et `restart_required=false`. Il n’y a ni rechargement à chaud ni API de changement du paquet de base.

**Erreurs:** Les erreurs de requête suivent le format habituel : `400 invalid_addon_request` pour un corps différent de `{}`, `401 unauthorized` si l’authentification échoue, `403 origin_not_allowed` pour une origine de navigateur interdite, `409 addon_management_unavailable` pour des chemins, permissions ou réglages incompatibles, et `415 json_required` si le contenu n’est pas JSON. Le navigateur doit utiliser la même origine que le serveur ou une origine explicitement autorisée par CORS.

Un travail accepté peut échouer plus tard : GET conserve HTTP 200 avec `state=error` et `error.code`. `addon_download_failed` ne change pas la configuration ; vérifier le réseau ou le proxy du serveur. Pour `addon_config_save_failed`, corriger la configuration ou les droits du répertoire ; un modèle vérifié reste réutilisable. `addon_config_invalid` indique un TOML invalide sur disque. `addon_model_invalid` exige de remplacer ou supprimer le fichier invalide, jamais écrasé silencieusement. `addon_job_in_progress` indique un autre processus en préparation : attendre et actualiser. Corriger la cause avant de répéter POST.

## Opérations faciales sans état

### `POST /v1/detect`

Chaque visage évalué contient `liveness.status`, `liveness.is_live` et `liveness.live_score`. Les résultats d’usurpation et `input_rejected` renvoient aussi HTTP 200, sans extraction de caractéristiques de reconnaissance. `input_rejected` indique une surface d’image insuffisante autour du visage ; `liveness.reason` explique comment ajuster l’image. L’absence de `liveness` signifie aucune évaluation.

**Entrée :** multipart `image` requis, `max_faces` 1–100, `collection_id` facultatif. **Traitement/résultat :** fusion multi-résolution, NMS globale, tri par aire ; 200 `faces` avec boîtes/5 points/score/qualité et `processing_ms`. Aucun visage donne une liste vide valide. **Erreurs :** 400 ancien min_score, 404 Collection, 413, 422 invalid_image, 503.

```bash
curl -sS "${BASE_URL}/v1/detect" -H "${AUTH_HEADER}" -F 'image=@group.jpg' -F 'max_faces=10'
```

### `POST /v1/compare`

`liveness_compare_scope` (`both`, `source`, `target`) choisit les côtés évalués avant reconnaissance. En `normal`, un rejet renvoie HTTP 422 `liveness_fake` / `liveness_input_rejected`, `error.details.liveness` et `error.details.side`, sans similarité. `observe` continue et joint les résultats aux visages évalués.

**Entrée :** multipart `source`, `target`, `threshold` facultatif 0–1 et `collection_id`. **Résultat :** choisit un visage par image ; 200 `matched`, cosine `similarity`, seuil effectif, deux visages et durée. **Erreurs :** 404, 413, 422 invalid_image/face_not_found, 503.

### `POST /v1/embeddings`

Avec la vérification en `normal`, une usurpation ou une entrée inadaptée renvoie HTTP 422 `liveness_fake` / `liveness_input_rejected` et `error.details.liveness` ; aucun embedding n’est extrait. `observe` renvoie l’embedding avec le résultat du visage.

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

La création de Person et l’ajout de FaceSamples ignorent cette vérification par défaut (`liveness_on_registration=false`). Si elle est activée, `normal` rejette les usurpations et les entrées inadaptées ; `observe` conserve le résultat et continue. Le contrôle qualité suit le `review_mode` choisi. La liste affiche séparément le véritable `reason` et le résultat de détection du vivant.

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

La création de Person et l’ajout de FaceSamples ignorent cette vérification par défaut (`liveness_on_registration=false`). Si elle est activée, `normal` rejette les usurpations et les entrées inadaptées ; `observe` conserve le résultat et continue. Le contrôle qualité suit le `review_mode` choisi. La liste affiche séparément le véritable `reason` et le résultat de détection du vivant.

**Entrée :** IDs ; images répétables et mêmes champs review/embedding que la création. **Résultat :** 201 `faces`, `rejected_images`, succès partiel. **Erreurs :** inscription plus 404 Person.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces`

**Entrée :** IDs ; query limit 1–100 et cursor. **Résultat :** 200 métadonnées `faces`, `has_crop`, `next_cursor`, sans embedding ni octets. **Erreurs :** 400 cursor, 404.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}/image`

**Entrée :** trois IDs. **Résultat :** si conservé, 200 `image/jpeg`, crop 112×112, `Cache-Control:no-store`; request ID dans l’en-tête seulement. **Erreurs :** 401, 404 face/face_image_not_found.

### `DELETE /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}`

**Entrée :** trois IDs. **Résultat :** supprime embedding/crop/ligne d’index, 204. **Erreurs :** 404, 503.

## Recherche

### `POST /v1/collections/{collection_id}/search`

Avec la vérification en `normal`, une usurpation ou une requête inadaptée renvoie HTTP 422 `liveness_fake` / `liveness_input_rejected` et `error.details.liveness` ; la recherche ne démarre pas. Ce cas diffère d’une liste de correspondances vide réussie. `observe` continue et renvoie le résultat du visage recherché.

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

Avec la vérification en `normal`, les visages bloqués ont `status: liveness_blocked` et un résultat séparé. Ils comptent dans `liveness_blocked_faces`, pas dans `unknown_faces`, et ne génèrent pas d’événement d’entrée. `observe` continue la reconnaissance. Les entrées rejetées et les usurpations sont affichées distinctement.

**Usage :** Interroger l’état courant depuis un client sans interface. **Entrée :** ID du Monitor. **Résultat :** 200 avec connexion, FPS effectif, latence, images sautées, visages reconnus/inconnus, aperçu, reconnexions et erreur sûre, sans embeddings. **Erreurs :** 401, 404.

### `GET /v1/monitors/{monitor_id}/events`

**Usage :** Lire les événements volatils entrée/sortie/erreur/rétablissement. **Entrée :** `limit` 1–1000 et le dernier `cursor` opaque. **Résultat :** 200 avec `events`, `next_cursor`, `truncated` et `stream_reset` ; un redémarrage perd les événements. **Erreurs :** 400 `invalid_cursor`, 401, 404.

### `GET /v1/monitors/{monitor_id}/preview.mjpeg`

**Usage :** Ouvrir l’aperçu MJPEG brut, désactivé par défaut. **Entrée :** ID et en-tête Bearer normal ; jamais la clé dans l’URL. **Résultat :** long `multipart/x-mixed-replace`, encodé seulement avec spectateurs ; le client trace les cadres via `/state`. **Erreurs :** 401, 404, 409 `preview_disabled`, 503.

## Réessais

GET peut être réessayé. Vérifiez l’état avant de répéter DELETE. Si une création Person/Face a un résultat réseau incertain, lisez l’ID avant un nouveau POST. Réessayez seulement 429 et 503 transitoires avec backoff exponentiel borné et jitter ; corrigez les 4xx.
