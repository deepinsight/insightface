# Guide utilisateur InsightFace Server

**Langues :** [English](user-guide.md) · [中文](user-guide.zh-CN.md) · [日本語](user-guide.ja.md) · [Deutsch](user-guide.de.md) · [Español](user-guide.es.md) · Français · [Русский](user-guide.ru.md) · [Português](user-guide.pt.md) · [한국어](user-guide.ko.md)

Ce guide accompagne un nouvel utilisateur depuis un répertoire vide jusqu’à la première recherche réussie. Les mêmes fonctions existent dans l’interface Web, `/v1` et le SDK Python. Tous les champs et résultats HTTP sont décrits dans le [guide API](api.fr.md).

Les modèles sont identifiés par `model_id` ; les réponses ne contiennent pas de champ `model_version` distinct.

Une mise à jour du Server avec le même modèle de reconnaissance et le même contrat conserve `embedding_contract_id`, échantillons et embeddings des Collections existantes. Changer de modèle constitue une migration distincte ; un contrat différent provoque `collection_model_mismatch` à l’inscription et à la recherche.

Pour utiliser la détection du vivant, consultez [configuration, installation et résultats](#addon-optionnel-de-détection-du-vivant). Chaque procédure explique aussi ses effets.

## De zéro à la première recherche

La version CPU nécessite Linux x86_64, Docker Engine et Docker Compose. CUDA exige en plus un pilote NVIDIA compatible et NVIDIA Container Toolkit ; il n’est pas nécessaire d’installer CUDA, cuDNN, ORT, Python ou OpenCV sur l’hôte.

Exécutez les commandes depuis la racine du dépôt avec `server/config/server.toml` présent. Server et l’installateur de modèles s’exécutent en root (`0:0`). Compose crée `server/.models` si nécessaire et le monte une seule fois en écriture dans `/models`. Le sous-répertoire `addons` est créé lors du téléchargement d’un addon. Aucun export UID/GID ni préparation manuelle des répertoires ou permissions n’est nécessaire. Le démarrage normal ne télécharge pas de modèles ; la détection du vivant est désactivée par défaut.

```bash
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml run --rm models install buffalo_l
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

Pour configurer facultativement la détection du vivant pendant l’installation du modèle, utilisez cette commande à la place :

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

Server n’a pas besoin de fonctionner. Pour une nouvelle installation, le prochain `up -d` active la détection du vivant ; si Server fonctionne déjà, utilisez `docker compose -f server/deploy/compose.cpu.yml restart server`. `up -d` seul ne recharge pas les paramètres enregistrés. Pour CUDA, utilisez `compose.cuda12.yml`.

Pour le GPU, utilisez `compose.cuda12.yml` et le port `18098`. L’installateur affiche la licence avant téléchargement ; les modèles publics InsightFace sont réservés à la recherche non commerciale sans licence commerciale distincte.

Le Compose fourni désactive l’authentification par défaut pour une évaluation isolée. Avant toute exposition réseau, définissez `INSIGHTFACE_AUTH_ENABLED=true` et un long `INSIGHTFACE_API_KEY`. Vérifiez ensuite le Dashboard, créez une Collection, inscrivez une Person et recherchez-la avec une autre image. Arrêtez avec `docker compose ... down` sans `-v` pour conserver le volume.

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

Dans **Système → Détection du vivant**, choisissez **Télécharger et activer après redémarrage**. Après vérification SHA-256, `liveness` est ajouté aux deux listes ; les autres entrées, commentaires et réglages sont conservés. Un fichier déjà vérifié est réutilisé. **Redémarrez manuellement le Server** pour appliquer le changement. Les erreurs permettent de réessayer ; un téléchargement échoué n’active pas la détection.

Système distingue l’installation vérifiée (`installed`), l’exécution actuelle (`enabled`), la configuration enregistrée pour le prochain démarrage (`configured_enabled`) et le redémarrage nécessaire (`restart_required`). Le téléchargement ou l’enregistrement ne modifie pas l’inférence en cours. Pour désactiver, enregistrez `inference.addons=[]` et `addons.auto_download=[]` dans le même fichier, puis redémarrez manuellement. L’action Web ne modifie pas le réglage d’inscription ; sa valeur par défaut reste `liveness_on_registration=false`.

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

```bash
docker compose -f server/deploy/compose.cpu.yml run --rm models addons install liveness
docker compose -f server/deploy/compose.cpu.yml run --rm models addons verify liveness
docker compose -f server/deploy/compose.cpu.yml up -d --force-recreate
```

Un modèle activé absent arrête le démarrage avec `addon_model_missing` ; un modèle invalide produit `addon_model_invalid`. L’addon n’est pas désactivé silencieusement.

### Montages et permissions pour les téléchargements Web

Les fichiers Compose fournis montent tout `/models` en écriture ; aucun montage séparé de `/models/addons` n’est nécessaire. Server et l’installateur utilisent root (`0:0`), et l’installation crée `addons` si nécessaire. Tout le répertoire existant `server/config` est monté en écriture dans `/etc/insightface` dans les deux services pour permettre à l’action Web et à `--enable-liveness` d’enregistrer `server.toml` de manière atomique. Cette configuration ne nécessite ni exports UID/GID, ni `chgrp`, ni `chmod`. Les fichiers ou montages volontairement en lecture seule empêchent toujours la gestion Web ; vérifiez les permissions des montages personnalisés.

Le montage de modèles de chaque service utilise `create_host_path: true`. Les services conservent les capabilities Docker par défaut ; `cap_drop: [ALL]` n’est pas appliqué. Le reste du système de fichiers reste en lecture seule et `no-new-privileges` demeure actif. Root peut modifier les fichiers des montages en écriture ; les nouveaux téléchargements peuvent appartenir à root sur l’hôte.

Pour un déploiement personnalisé, utilisez les chemins réellement montés ; pour CUDA, utilisez `compose.cuda12.yml`. Les anciens montages en lecture seule restent utilisables avec la détection du vivant désactivée. L’action Web explique son indisponibilité ; vous pouvez aussi installer par CLI et modifier la configuration manuellement. Après enregistrement dans le Web, appliquez avec `docker compose -f server/deploy/compose.cpu.yml restart server`. Un changement de montage ou de variables proxy nécessite de recréer le conteneur.

Si un proxy est nécessaire, définissez `HTTP_PROXY`, `HTTPS_PROXY` et `NO_PROXY` avant de créer le conteneur ; Compose les transmet au Server et à l’outil de modèles. Utilisez une adresse LAN accessible depuis le conteneur : son `127.0.0.1` ne désigne pas le Mac. L’action utilise l’authentification API Key existante ; sans authentification, un client ayant accès à l’API peut aussi l’exécuter. Elle télécharge uniquement le modèle de détection du vivant publié et fixé, sans URL arbitraire ni changement de modèle de base.

### Résultats de détection du vivant

| Résultat | `status` | `is_live` | `live_score` |
| --- | --- | --- | --- |
| Vérification réussie | `ok` | `true` | `[0, 1]` |
| Non vivant | `ok` | `false` | `[0, 1]` |
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

`normal` reconnaît uniquement les visages qui passent le contrôle ; `observe` conserve le résultat et poursuit la reconnaissance. Sans évaluation, `liveness` est omis. Les trois champs principaux sont `status`, `is_live` et `live_score` : succès/fake utilise `status: ok`, un booléen et un score ; une entrée rejetée utilise `status: input_rejected` et deux valeurs `null`.

Detect renvoie HTTP 200 même pour un résultat négatif. En `normal`, embeddings, comparaison et recherche renvoient HTTP 422 `liveness_fake` ou `liveness_input_rejected` avec `error.details.liveness` ; la comparaison ajoute `details.side`. Une panne renvoie HTTP 503 `liveness_unavailable`. Les erreurs d’exécution interrompent l’opération en `normal` comme en `observe` ; elles ne sont pas converties en `input_rejected`.

La création de personnes et l’ajout de FaceSamples ignorent ce contrôle par défaut : `[inference].liveness_on_registration=false` n’exécute pas le modèle et omet `liveness` dans les nouveaux échantillons. Avec `true` et l’addon activé, la politique `normal`/`observe` s’applique ; les refus comprennent `reason` et `liveness`. La qualité selon `review_mode` et la validation des embeddings externes restent contrôlées. `review_mode=off` et `external_trusted` ne contournent pas un contrôle d’inscription activé. Les requêtes ne peuvent pas modifier cette configuration de démarrage. Les résultats déjà enregistrés restent consultables.

RTSP distingue `liveness_blocked` de `unknown` et compte `liveness_blocked_faces`. Les visages bloqués ne produisent aucun événement d’entrée de personne/inconnu et réinitialisent la confirmation. Une panne d’inférence efface les identités précédemment affichées.

`liveness_compare_scope` sélectionne `both` (par défaut), `source` ou `target` pour `/v1/compare`. La vérification réussit si `live_score >= liveness_threshold`.

Le modèle est stocké dans `server/.models/addons/liveness.onnx` sur l’hôte et `/models/addons/liveness.onnx` dans le conteneur. `addons` dans `/v1/models` et `/v1/system` indique les addons actifs.

[Contrat API complet](api.fr.md#addon-optionnel-de-détection-du-vivant).

## 1. Connexion et état

Ouvrez `http://SERVEUR:18097/` pour le CPU ou `http://SERVEUR:18098/` pour CUDA 12. Si l’authentification est active, choisissez **Configurer la clé API**, collez la clé fournie et appliquez-la à l’onglet. Elle reste uniquement en mémoire et disparaît au rechargement ou à la fermeture.

Dans **Tableau de bord** ou **Système**, vérifiez que service, base, modèles et Provider sont prêts. CUDA doit afficher `CUDAExecutionProvider` et ne bascule jamais silencieusement sur CPU.

Le tableau de bord affiche toujours la détection du vivant activée ou désactivée sous le modèle. Système distingue installation, fonctionnement actuel et redémarrage requis.

## 2. Créer une Collection

Dans **Collections** → **Nouvelle collection**, définissez un ID stable, un nom,
le seuil cosinus (`0.4` au départ), un profil disponible, la capacité et le
nombre maximal de FaceSamples par personne. La conservation JPEG d’un
`bounding-box crop` redimensionné en 112×112 est désactivée par défaut ; ce
n’est pas l’entrée alignée du modèle de reconnaissance.

La Collection est liée à l’ID, le digest, la dimension et le prétraitement du modèle. Après un changement de modèle, elle reste visible mais inscription et recherche sont refusées si le contrat diffère.

Le profil de détection copie les valeurs système à la création, puis permet de modifier tailles d’entrée, seuils détection/NMS et stratégie mono-visage. `largest` privilégie la surface ; `center_largest` maximise `surface - 2,0 × distance en pixels au carré entre le centre du cadre et celui de l’image`. La confiance de détection ne participe pas à ce score.

## 3. Inscrire une Person

Dans **Personnes**, sélectionnez la Collection puis **Inscrire une personne**. Saisissez éventuellement ID, nom, ID externe, metadata JSON et une ou plusieurs images JPEG, PNG, WebP ou BMP.

- `off` : utilise la stratégie mono-visage de la Collection et autorise plusieurs visages ;
- `standard` : impose un visage exploitable et contrôle taille, détection, netteté, luminosité et pose ;
- `strict` : impose aussi que la meilleure similarité interne soit supérieure à la meilleure similarité externe.

Un lot accepte un succès partiel et détaille chaque rejet. Les originaux ne sont pas stockés. `external_trusted` accepte un embedding normalisé L2 ; l’image reste obligatoire pour détection et qualité, mais le vecteur n’est pas réextrait.

La création de Person et l’ajout de FaceSamples ignorent cette vérification par défaut (`liveness_on_registration=false`). Si elle est activée, `normal` rejette fake/entrée inadaptée ; `observe` conserve le résultat et continue. Le contrôle qualité suit le `review_mode` sélectionné. La liste affiche séparément le véritable `reason` et le résultat de détection du vivant.

## 4. Détecter, comparer et rechercher

**Détecter** affiche boîtes, cinq points, score et qualité ; aucun visage renvoie une liste vide valide. **Comparer** utilise le profil système ou Collection pour choisir un visage par image et renvoie `similarity` cosinus, `threshold` et `matched`. La similarité n’est pas une probabilité.

Dans **Rechercher**, choisissez Collection et image. Le score d’une personne est la meilleure similarité de ses FaceSamples. Les résultats sont triés par ordre décroissant ; aucun résultat donne une liste vide. Chaque échantillon est d’abord validé dans SQLite puis ajouté à l’index avant la réponse. Au redémarrage, l’index est reconstruit depuis SQLite.

Chaque visage évalué contient `liveness.status`, `liveness.is_live` et `liveness.live_score`. Fake et `input_rejected` renvoient aussi HTTP 200, sans extraction de caractéristiques de reconnaissance. `input_rejected` indique une surface d’image insuffisante autour du visage ; `liveness.reason` explique comment ajuster l’image. L’absence de `liveness` signifie aucune évaluation.

`liveness_compare_scope` (`both`, `source`, `target`) choisit les côtés évalués avant reconnaissance. En `normal`, un rejet renvoie HTTP 422 `liveness_fake` / `liveness_input_rejected`, `error.details.liveness` et `error.details.side`, sans similarité. `observe` continue et joint les résultats aux visages évalués.

Avec la vérification en `normal`, fake/requête inadaptée renvoie HTTP 422 `liveness_fake` / `liveness_input_rejected` et `error.details.liveness` ; la recherche ne démarre pas. Ce cas diffère d’une liste de correspondances vide réussie. `observe` continue et renvoie le résultat du visage recherché.

## 5. Surveillance de caméra RTSP

Dans **Surveillance caméra**, créez un Monitor persistant et configurez source RTSP, Collection, fréquence, seuil facultatif et politique d’événements. L’aperçu est désactivé par défaut ; reconnaissance et événements continuent sans lui. Lorsqu’il est actif, la Web UI trace les inscrits en vert et les inconnus en orange depuis `/state` sur des images brutes.

Le Monitor fonctionne indépendamment du navigateur et les tâches actives sont restaurées après redémarrage. La configuration est dans SQLite et les identifiants RTSP sont chiffrés dans `/data`, mais images et événements ne sont pas enregistrés. Les événements restent seulement dans un tampon mémoire borné. Le décodeur garde l’image la plus récente et ignore les anciennes au lieu de les empiler.

Avec la vérification en `normal`, les visages bloqués ont `status: liveness_blocked` et un résultat séparé. Ils comptent dans `liveness_blocked_faces`, pas dans `unknown_faces`, et ne génèrent pas d’événement d’entrée. `observe` continue la reconnaissance. Entrée rejetée et fake sont affichés distinctement.

## 6. Données et sécurité

Conservez et sauvegardez `/data`, le répertoire des modèles et la configuration. Le déploiement fourni utilise root avec les capabilities Docker par défaut et `/models` en écriture ; le service peut modifier les modèles, la configuration et les données montés. Le reste du système de fichiers du conteneur demeure en lecture seule et `no-new-privileges` reste actif ; `privileged` n’est pas nécessaire. Limitez l’accès à Docker et à l’hôte, et ne montez pas de répertoires hôtes étrangers au service. Avant les opérations en masse, sauvegardez SQLite et les recadrages ensemble. Les clés sont hachées ; démarrer le même volume avec un nouvel `INSIGHTFACE_API_KEY` change la clé active. Ne journalisez ni images, ni embeddings, ni clés.

L’explorateur de schéma OpenAPI destiné aux développeurs se trouve sous `/docs` ; les instructions API orientées tâches sont dans cette aide. Fournissez `x-request-id` lors d’un incident. `401` concerne la clé, `409 collection_model_mismatch` le contrat modèle, `422 face_not_found` l’absence de visage exploitable.

## 7. Modèles et licences

Les images ne contiennent aucun modèle. Le démarrage normal reste hors ligne ;
le service ponctuel `models` installe dans `server/.models` :

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models verify buffalo_l
```

Les paquets pris en charge sont `buffalo_l` (`det_10g.onnx` +
`w600k_r50.onnx`), `buffalo_m`, `buffalo_s`, `buffalo_sc`, `antelopev2`,
`raccoon_s` et `raccoon_l`. L’installation
crée `manifest.json` et le fichier signé `MODEL.LICENSE`. Sans
`--accept-license`, un terminal interactif demande confirmation avant de
télécharger. Les commandes non interactives exigent cette option et s’arrêtent
sans téléchargement si elle est absente. Les modèles préentraînés publics InsightFace sont réservés à la
recherche non commerciale sans licence commerciale séparée.

`raccoon_s` et `raccoon_l` sont pris en charge. Le Server installe uniquement la détection et la reconnaissance de chaque paquet ; le vérificateur Raccoon n’est pas chargé. Le nom identifie le modèle, sans numéro de version indépendant. L’action Web de détection du vivant ne change pas le modèle de base. Pour un autre modèle de reconnaissance, utilisez une Collection compatible ; les anciens embeddings ne deviennent pas les caractéristiques du nouveau modèle.

## 8. Configuration de démarrage et recherche

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

`server/config/server.toml` est lu une seule fois au démarrage ; toute
modification exige un redémarrage. Valeurs initiales :
`input_sizes=[[96,96],[512,512]]`, seuil de détection `0.50`, NMS `0.40`,
`single_face_selection="largest"` et 100 visages au maximum. SCRFD exécute
chaque résolution, reprojette tous les candidats sur l’image source et effectue
un seul NMS global. `max_concurrency="auto"` signifie CPU 4 et CUDA 8.
`[web].disabled=true` ne conserve que `/v1` et `/openapi.json`.

System n’annonce que les profils réellement disponibles. Le profil est fixé à
la création de la Collection et n’est pas sélectionnable par requête :

- `fp32_v1` : CPU/CUDA standard ;
- `fp16_v1` : CUDA ;
- `bf16_v1` : CPU compatible ou CUDA SM80+ ;
- `int8_x736_v1` : INT8 recommandé CPU/CUDA, accumulation INT32 ;
- `int8_x1000_v1` : compatibilité des Collections existantes.

Tous parcourent chaque FaceSample et ne sont pas des index ANN ; le score public
reste raw cosine. `capacity_rows=100000`, garde-fou `10000000` et
`max_faces_per_person=20`. Pour 512 dimensions, le vecteur seul représente
environ 2 048 octets FP32, 1 024 FP16/BF16 ou 512 INT8 par ligne.

## 9. SDK, construction et exploitation des données

Le SDK Python accepte chemin, bytes et objet fichier et fournit des méthodes
typées pour Detect, Compare, Collections, inscription, Search et Monitors.
Consultez le contrat HTTP dans le [guide API](api.fr.md).

Vous pouvez construire directement depuis un répertoire local contenant toutes
les sources, même avec des modifications non committées ou sans répertoire
`.git`. Les commits et les push Git ne sont pas des prérequis à la construction.

```bash
make -C server build-cpu
make -C server build-cuda12
```

Une fois les tests réussis, publiez la même image que celle testée. Committer
ensuite les mêmes sources ou organiser les commits ne nécessite pas de
reconstruction. Toute modification des fichiers inclus dans l’image, tels que
le code, les ressources du frontend ou l’aide utilisateur intégrée, nécessite
une nouvelle construction et une nouvelle validation.

Ajoutez `--pull never` aux commandes Compose pour employer l’image locale. Les
tags immuables sont `0.3.1-cpu` et `0.3.1-cuda12`; `cpu` et `cuda12` suivent la
dernière version stable et aucun `latest` n’est publié. Avant mise à niveau,
arrêtez les écritures et sauvegardez `/data` et les crops avec une méthode sûre
pour SQLite. N’utilisez pas `docker compose down -v`, qui supprime le volume.

### Mise à niveau vers 0.3.1

0.3.1 simplifie le déploiement : Server et l’installateur utilisent root, Compose crée le répertoire des modèles si nécessaire et un unique `/models` en écriture remplace le montage séparé des addons.

Depuis 0.3.0, `raccoon_s`, `raccoon_l`, leurs manifestes, la détection du vivant facultative, l’installation Web d’addons et les images BMP sont pris en charge. Server utilise la détection et la reconnaissance Raccoon ; le vérificateur n’est pas chargé. Ces fonctionnalités et contrats de réponse API restent inchangés en 0.3.1.

**1.** Mettez le code du Server et les fichiers Compose à la version 0.3.1 en
conservant vos réglages dans `server/config/server.toml` et vos surcharges
de déploiement. Gardez le chemin actuel des modèles, le nom du volume
`/data`, le stockage des recadrages, les ports et les réglages de clé API.
Dans les fichiers Compose personnalisés, mettez les images des deux services
`server` et `models` à `0.3.1-cpu` ou `0.3.1-cuda12` selon l’environnement.
Pour les commandes ci-dessous, utilisez les mêmes fichiers Compose,
surcharges et nom de projet que pour votre déploiement habituel.

Avant le démarrage, mettez aussi à jour vos surcharges Compose, pas seulement les tags : les deux services doivent utiliser `user: "0:0"`, sans `cap_drop: [ALL]` ni anciens réglages UID/GID ou `group_add`. Utilisez pour chacun un seul bind mount en écriture dans `/models` avec `create_host_path: true` ; supprimez le montage séparé de `/models/addons` et `x-addons-path`. Conservez le chemin réel des modèles et leurs fichiers, addons compris. Les montages de configuration gardent `create_host_path: false` : `server/config` et `server.toml` doivent donc rester présents. Conservez le volume de données et sauvegardez données, modèles et configuration avant le changement. Retirez les anciens réglages d’utilisateur et de montages de vos surcharges ; changer seulement l’image ne les corrige pas.

Conservez le système de fichiers du conteneur en lecture seule et `no-new-privileges`. Les deux services ont besoin de tout le répertoire de configuration en écriture. Remplacez l’ancien montage du fichier seul en lecture seule de l’installateur par celui du répertoire. Le déploiement standard ne nécessite ni modification récursive des permissions existantes, ni préparation d’un répertoire d’addons.

**2.** Téléchargez les nouvelles images et recréez le conteneur Server. Depuis la
racine du dépôt, choisissez les commandes de votre déploiement actuel :

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

Si vous compilez localement, construisez d’abord les images 0.3.1 et utilisez
`up -d --no-build --pull never --force-recreate server` au lieu de télécharger
les images. `docker compose restart` seul ne passe pas à une nouvelle image
et n’applique pas les modifications de montage.

**3.** Le démarrage applique automatiquement les migrations de base de données.
Attendez que `/v1/health` indique `ready` et la version `0.3.1`, puis vérifiez
dans **Système** le modèle et le fournisseur d’exécution attendus. Confirmez
la présence des Collections et des personnes existantes, puis effectuez une
recherche connue. Si le modèle et le contrat d’embedding restent identiques,
les échantillons, embeddings et identifiants de contrat des Collections sont
conservés ; aucune réinscription n’est nécessaire.

**La détection du vivant reste facultative après la mise à niveau.** La
configuration fournie et les anciennes configurations sans clés d’addon la
laissent désactivée. Une simple mise à niveau ne nécessite donc aucun
téléchargement du modèle de détection du vivant. Le Server ne télécharge jamais
de modèle au démarrage. Pour l’activer, suivez les
[instructions de configuration](#addon-optionnel-de-détection-du-vivant) :
préparez les [montages et permissions Web](#montages-et-permissions-pour-les-téléchargements-web),
choisissez **Système → Détection du vivant → Télécharger et activer après redémarrage**,
attendez la réussite de l’installation et de l’enregistrement de la
configuration, puis redémarrez manuellement le Server. Les valeurs par défaut
sont `normal`, un seuil de `0.8` et `liveness_on_registration=false`. Le modèle
reste dans `<models_dir>/addons/liveness.onnx`.

**Adopter Raccoon constitue un changement de modèle distinct.** La mise à
niveau du Server conserve le paquet de modèle actuel. Pour utiliser
`raccoon_s` ou `raccoon_l`, installez le paquet choisi dans un répertoire de
modèles séparé en suivant les
[instructions d’installation](#7-modèles-et-licences), puis configurez un
déploiement pour l’utiliser. Les Collections doivent correspondre au contrat
d’embedding du nouveau modèle : créez des Collections compatibles et
réinscrivez les personnes, ou effectuez une migration de données distincte.
La Web UI ne change pas le paquet de modèle de base.

**Compatibilité API et SDK depuis 0.3.0 :** Les résultats des modèles, Collections et
FaceSamples ne contiennent plus `model_version`. L’identité du modèle utilise
`model_id`, et la compatibilité des Collections utilise `embedding_contract_id`.
Adaptez les clients qui exigent le champ supprimé et utilisez le SDK `0.3.1`
lors de la mise à niveau du client Python fourni. Lorsqu’une vérification du
vivant est effectuée, `liveness` contient les champs principaux `status`, `is_live` et
`live_score`, avec `reason` uniquement pour `input_rejected` ; sinon, ce champ est omis. Consultez les
[résultats et erreurs de détection du vivant](#résultats-de-détection-du-vivant)
avant de l’activer pour les requêtes de reconnaissance.

## 10. GPU, réseau et dépannage

L’image CUDA contient CUDA Runtime 12.9.1, cuDNN 9.24.0 et
`onnxruntime-gpu==1.27.0`. Turing/Ampere/Ada/Hopper demandent R535 ou plus,
Blackwell/RTX 50 demandent 570.26 ou plus ; une R580 stable ou plus récente est
recommandée. Au démarrage, GPU, Compute Capability, Driver, CUDA/cuDNN/ORT,
Provider, Sessions réelles et warm-up sont vérifiés ; aucun repli CPU silencieux
n’est permis.

Pour une exposition réseau, terminez HTTPS sur un reverse proxy fiable,
restreignez CORS, débit, taille et délais, puis protégez `/data` et les backups
comme données biométriques. Ne journalisez jamais images, embeddings ou clés.
La phase un ne possède qu’une API Key sans rôles et n’est pas un système
d’autorisation multi-tenant.
