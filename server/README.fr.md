# InsightFace Server

**Langues :** [English](README.md) · [中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · Français · [Русский](README.ru.md) · [Português](README.pt.md) · [한국어](README.ko.md)

> **Un seul GPU. 50M+ de vecteurs faciaux. Recherche ultrarapide par quantification INT8 des caractéristiques, sans perte de précision significative.**

**Un serveur de reconnaissance faciale autohébergé réunissant Web UI, REST API
simple, SQLite et inférence locale CPU ou GPU NVIDIA dans un seul conteneur.**

```text
téléverser une image -> détecter, effectuer un contrôle du vivant, comparer, inscrire ou rechercher
```

> **Licence des modèles :** les modèles publics préentraînés InsightFace sont
> généralement réservés à la recherche non commerciale. L’usage commercial
> exige une autorisation distincte d’[InsightFace](https://www.insightface.ai).

InsightFace Server est une alternative plus simple et orientée confidentialité
à AWS Rekognition pour les usages courants sur votre propre infrastructure. Les
images, embeddings, modèles et index peuvent rester dans votre réseau. Ce n’est
pas un remplacement compatible AWS et il n’implémente ni SigV4, IAM, Region, ni
la sémantique des ressources AWS.

Version actuelle : **0.3.1**, Linux x86_64.

| Environnement | Image |
| --- | --- |
| CPU | `ghcr.io/deepinsight/insightface-server:0.3.1-cpu` |
| GPU NVIDIA | `ghcr.io/deepinsight/insightface-server:0.3.1-cuda12` |

Les tags mobiles `cpu` et `cuda12` désignent la dernière version stable de
chaque famille. Aucun tag ambigu `latest` n’est fourni. Voir le
[Maintainer Guide — English](docs/maintainer-guide.md) pour la politique.

**Nouveautés et mise à niveau vers 0.3.1 :** Server et l’installateur s’exécutent en root ; un seul montage de modèles en écriture remplace le montage séparé des addons. Compose crée le répertoire si nécessaire, sans configuration manuelle des UID/GID ou permissions. Mettez aussi à jour les fichiers Compose et vos surcharges en conservant les chemins des modèles, de la configuration et des données. Consultez la [procédure de mise à niveau](docs/user-guide.fr.md#mise-à-niveau-vers-031).

Depuis 0.3.0, `raccoon_s`, `raccoon_l`, leurs manifestes, la détection du vivant facultative avec installation Web et les images BMP sont pris en charge. Depuis cette version, les résultats API et SDK omettent `model_version`. La détection du vivant reste désactivée tant qu’elle n’est pas activée.

![Dashboard InsightFace Server en anglais](docs/images/customer/dashboard-en.jpg)

## Fonctionnalités principales

- Détection SCRFD, cinq landmarks, alignement, embeddings ArcFace, L2
  normalization, cosine similarity brut et recherche Person 1:N exacte.
- Détection multirésolution avec un NMS global après fusion, et sélection
  `largest` ou `center_largest`.
- `Collection -> Person -> FaceSample`, Collections liées au modèle,
  inscription multi-image avec succès partiel, metadata et motifs explicites.
- `review_mode` d’inscription : `off`, `standard` ou `strict`; embeddings
  pré-calculés facultatifs via `external_trusted`.
- Recherche GPU exacte avec stockage vectoriel FP32, FP16, BF16 et INT8.
- Web UI multilingue pour Dashboard, Collections, People, Detect, Compare,
  Search, surveillance RTSP, System et Help.
- 31 opérations REST snake_case sous `/v1`, dont `/v1/embeddings` protégé, et
  un SDK Python léger et typé.
- RTSP Monitors persistants côté serveur, événements bornés en mémoire,
  plusieurs clients et `preview.mjpeg` facultatif ; fermer le navigateur
  n’arrête pas la surveillance.
- SQLite comme source durable, index exacts mémoire reconstruisibles,
  un seul montage de modèles en écriture et un répertoire de configuration accessible en écriture, `/data` persistant, migrations, health checks et
  validation CUDA stricte sans fallback CPU silencieux.
- Entrées JPEG, PNG, WebP et BMP ; les originaux ne sont pas conservés par défaut.

La détection du vivant est désactivée par défaut : `inference.addons = []` et `addons.auto_download = []` dans `server/config/server.toml`. **Système → Détection du vivant** télécharge et vérifie le modèle, puis enregistre l’activation pour le prochain redémarrage manuel ; un fichier vérifié est réutilisé. Aucun téléchargement au démarrage. Un modèle activé manquant arrête le démarrage avec des instructions d’installation. Chaque visage évalué renvoie les champs principaux `status`, `is_live` et `live_score` ; seul `input_rejected` ajoute une explication pour l’utilisateur dans `reason`. Le mode par défaut est `normal` ; l’inscription ignore la détection par défaut (`liveness_on_registration = false`). Voir [configuration, permissions Web et mise à niveau](docs/user-guide.fr.md#addon-optionnel-de-détection-du-vivant).

### Performances de recherche GPU sur RTX 5090

Sur une NVIDIA GeForce RTX 5090 (32 607 MiB), l’index CUDA flat exact natif a
stocké jusqu’à **58,9M vecteurs d’image de 512 dimensions en INT8**.

| Type de données GPU | Vecteurs d’image maximum | 10M Top-5 p50 | 10M QPS série |
| --- | ---: | ---: | ---: |
| FP32 | 15,8M | 12,84 ms | 77,85 |
| FP16 | 30,7M | 6,83 ms | 146,32 |
| BF16 | 30,7M | 6,83 ms | 146,33 |
| INT8 | **58,9M** | **3,84 ms** | **260,81** |

INT8 atteint 3,73 fois la capacité mesurée et 3,35 fois le débit Top-5 de FP32
sur 10M. Ces mesures GPU proviennent de la même RTX 5090 avec Driver 580.105.08
et CUDA 12.9. La capacité est la limite isolée de l’index natif, sans modèles
ONNX chargés ni charge Server. La vitesse utilise exactement 10M vecteurs
d’image, un balayage exact Top-5 résident GPU, une seule requête en vol, 10
warm-ups et 100 mesures. La recherche est exacte dans chaque représentation
stockée ; la quantification peut néanmoins modifier les scores par rapport à
FP32. La production doit réserver de la VRAM pour modèles, requêtes,
concurrence, reconstruction d’index et allocator.

### Précision MR-ALL multiethnique ICCV21-MFR

Nous avons évalué les profils de recherche natifs sur le jeu de test
multiethnique (MR) d’[ICCV21-MFR](../challenges/iccv21-mfr/) avec son protocole
MR-ALL 1:1 toutes paires à FAR `1e-6`. Tous les profils utilisent les mêmes
embeddings `buffalo_l` de 512 dimensions, normalisés L2 et extraits une seule
fois via la Server API ; seules changent les représentations de stockage et de
calcul de la recherche.

| Profil de recherche | MR-ALL à FAR 1e-6 | Seuil cosine | Écart par rapport à FP32 |
| --- | ---: | ---: | ---: |
| FP32 | 91,249107 % | 0,407787 | — |
| FP16 | 91,249197 % | 0,407787 | +0,000090 point |
| BF16 | 91,248502 % | 0,407787 | -0,000605 point |
| **INT8** | **91,248005 %** | **0,407739** | **-0,001102 point** |

**INT8 ne présente aucune perte de précision significative sur ce
benchmark :** avec l’affichage à deux décimales utilisé par le challenge, FP32
et INT8 obtiennent tous deux **91,25 % de MR-ALL** ; l’écart non arrondi n’est
que de 0,0011 point. Les avantages indiqués plus haut — capacité mesurée
3,73 fois supérieure et débit Top-5 sur 10M 3,35 fois supérieur — sont
conservés. Cette comparaison mesure la précision du stockage et de la recherche
vectorielle, pas l’inférence de modèle INT8.

![Gestion des Collections en anglais](docs/images/customer/collections-en.jpg)

![RTSP Monitor en anglais ; adresse privée masquée](docs/images/customer/monitoring-en.jpg)

## Démarrage rapide

Prérequis :

- Linux x86_64 avec Docker Engine et Docker Compose ;
- pour CUDA, GPU NVIDIA compatible, NVIDIA Driver et NVIDIA Container Toolkit.

L’hôte n’a pas besoin de Python, OpenCV, ONNX Runtime, CUDA Toolkit ou cuDNN.
Les images publiques n’incluent ni modèles, données client, API Keys, ni
configuration de production.

Depuis un checkout complet d’InsightFace, installez un modèle dans
`server/.models` :

Exécutez les commandes depuis la racine du dépôt avec `server/config/server.toml` présent. Server et l’installateur de modèles s’exécutent en root (`0:0`). Compose crée `server/.models` si nécessaire et le monte une seule fois en écriture dans `/models`. Le sous-répertoire `addons` est créé lors du téléchargement d’un addon. Aucun export UID/GID ni préparation manuelle des répertoires ou permissions n’est nécessaire. Le démarrage normal ne télécharge pas de modèles ; la détection du vivant est désactivée par défaut.

```bash
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license
```

Pour configurer facultativement la détection du vivant pendant l’installation du modèle, utilisez cette commande à la place :

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

Server n’a pas besoin de fonctionner. Pour une nouvelle installation, le prochain `up -d` active la détection du vivant ; si Server fonctionne déjà, utilisez `docker compose -f server/deploy/compose.cpu.yml restart server`. `up -d` seul ne recharge pas les paramètres enregistrés. Pour CUDA, utilisez `compose.cuda12.yml`.

L’outil accepte les sept paquets : `buffalo_l`, `buffalo_m`, `buffalo_s`,
`buffalo_sc`, `antelopev2`, `raccoon_s` et `raccoon_l`. Il écrit
`manifest.json` et le `MODEL.LICENSE` signé ; `models verify` contrôle le
paquet. Les conditions du modèle sont distinctes de la licence du code Server.

Démarrer CPU :

```bash
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

Démarrer CUDA 12 à la place :

```bash
docker compose -f server/deploy/compose.cuda12.yml pull
docker compose -f server/deploy/compose.cuda12.yml \
  run --rm models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cuda12.yml up -d
curl -fsS http://127.0.0.1:18098/v1/health
```

Ouvrez `http://SERVEUR:18097/` pour CPU ou `http://SERVEUR:18098/` pour CUDA.
Créez une Collection, inscrivez une Person avec une ou plusieurs photos, puis
recherchez avec une autre photo. `docker compose ... down` sans `-v` conserve
le volume.

Les fichiers Compose fournis désactivent l’authentification par défaut pour une
évaluation isolée. Avant d’exposer le service à d’autres utilisateurs ou
réseaux :

```bash
export INSIGHTFACE_AUTH_ENABLED=true
export INSIGHTFACE_API_KEY='remplacer-par-un-long-secret-aleatoire'
docker compose -f server/deploy/compose.cpu.yml up -d
```

Le [guide utilisateur](docs/user-guide.fr.md) décrit tout le premier parcours.

## Construire depuis les sources

Vous pouvez construire directement depuis un répertoire local contenant toutes
les sources, même avec des modifications non committées ou sans répertoire
`.git`. Les commits et les push Git ne sont pas des prérequis à la construction.

Les Dockerfiles copient `server/` et certains modules d’inférence de
`python-package/insightface/` ; le répertoire complet des sources est donc le contexte de build.

CPU :

```bash
make -C server build-cpu
docker compose -f server/deploy/compose.cpu.yml \
  run --rm --pull never models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cpu.yml \
  up -d --no-build --pull never
```

CUDA 12 :

```bash
make -C server build-cuda12
docker compose -f server/deploy/compose.cuda12.yml \
  run --rm --pull never models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cuda12.yml \
  up -d --no-build --pull never
```

`--pull never` garantit l’utilisation de l’image locale. Le build télécharge
toujours les images de base et dépendances verrouillées ; l’installation du
modèle télécharge séparément le paquet dont la licence a été acceptée.

## Comportement essentiel

- Similarity est le cosinus brut, pas une probabilité. Les thresholds utilisent
  `0.0..1.0`, valeur par défaut `0.4`.
- Une Collection fixe le modèle et l’embedding contract. En cas d’écart elle
  reste visible, mais inscription/recherche renvoie
  `collection_model_mismatch`.
- Le Detection Profile de démarrage est copié dans les nouvelles Collections ;
  leur profil peut ensuite changer indépendamment pour les requêtes suivantes.
- Le stockage facultatif conserve un bounding-box JPEG crop redimensionné en
  112x112, pas l’original ni l’entrée alignée de reconnaissance ; désactivé par
  défaut.
- Les commits SQLite font foi. L’index est synchronisé avant toute réponse
  d’inscription/suppression réussie et reconstruit depuis SQLite au redémarrage.
- Les réponses contiennent `x-request-id` ; les listes utilisent des cursor
  opaques et signés.

Les champs, defaults, cycles de vie et erreurs exacts sont maintenus uniquement
dans les documents détaillés ci-dessous.

## API et SDK

Groupes principaux :

- système : `/v1/health`, `/v1/system`, `/v1/models` ;
- visage sans état : `/v1/detect`, `/v1/compare`, `/v1/embeddings` ;
- CRUD Collection, Person et FaceSample ;
- recherche de Person dans une Collection ;
- configuration, état, événements et aperçu des RTSP Monitors.

Le [guide REST API](docs/api.fr.md) contient tous les paramètres, réponses,
erreurs et exemples. OpenAPI interactif reste disponible sous `/docs`.

```python
from insightface_server import Client

with Client("http://localhost:18097", api_key=None) as client:
    faces = client.detect("photo.jpg")
    matches = client.search("employees", "unknown.jpg", limit=5)
```

L’installation, les entrées, les méthodes et les parcours complets du SDK sont
dans le [guide utilisateur](docs/user-guide.fr.md).

## Sécurité

Le déploiement fourni utilise root avec les capabilities Docker par défaut et peut écrire dans les montages de modèles, de configuration et de données. Le reste du système de fichiers est en lecture seule et `no-new-privileges` reste actif. Limitez l’accès à l’hôte et les répertoires montés.

Les images faciales et embeddings sont des données biométriques. En réseau,
activez l’authentification, terminez HTTPS sur un reverse proxy fiable,
restreignez Docker et les volumes, laissez le CORS large désactivé et définissez
backup, rétention, suppression, consentement et réponse aux incidents. Ne
journalisez ni images, embeddings, identifiants RTSP, ni API Keys.

Le Server n’inclut ni TLS, comptes utilisateurs, RBAC, cloud IAM, ni couche de
conformité légale. L’exploitation et la sécurité sont dans le
[guide utilisateur](docs/user-guide.fr.md).

## Périmètre de la première phase

Sont exclus : compatibilité AWS/CompreFace, CUDA 11, Jetson, ARM64, Windows
Container, TensorRT, Kubernetes, Workers distribués, événements Monitor
persistants ou enregistrement/NVR, deepfake et attributs
démographiques.

## Documentation

- [Guide utilisateur](docs/user-guide.fr.md) — installation, configuration,
  modèles, Web UI, SDK, GPU, sécurité, backup et dépannage.
- [Guide REST API](docs/api.fr.md) — tous les endpoints, champs, comportements,
  résultats, erreurs, pagination et exemples.
- [Maintainer Guide — English](docs/maintainer-guide.md) — architecture,
  recherche interne, tests, contribution et releases des conteneurs.

GitHub et l’aide Web UI lisent les mêmes Markdown localisés ; seule la
présentation diffère.

## Licence

Le point d’entrée unique des licences est [LICENSING.md](LICENSING.md). Le code
Server et le SDK Python sont sous MIT License ; cette déclaration ne couvre ni
fichiers ou poids de modèles, ni datasets, ni composants tiers. Les modèles
publics InsightFace sont généralement limités à la recherche non commerciale
sans autorisation séparée. Licence commerciale :
<https://www.insightface.ai>.
