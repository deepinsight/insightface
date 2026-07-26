# Guide utilisateur InsightFace Server

**Langues :** [English](user-guide.md) · [中文](user-guide.zh-CN.md) · [日本語](user-guide.ja.md) · [Deutsch](user-guide.de.md) · [Español](user-guide.es.md) · Français · [Русский](user-guide.ru.md) · [Português](user-guide.pt.md) · [한국어](user-guide.ko.md)

Ce guide accompagne un nouvel utilisateur depuis un répertoire vide jusqu’à la première recherche réussie. Les mêmes fonctions existent dans l’interface Web, `/v1` et le SDK Python. Tous les champs et résultats HTTP sont décrits dans le [guide API](api.fr.md).

## De zéro à la première recherche

La version CPU nécessite Linux x86_64, Docker Engine et Docker Compose. CUDA exige en plus un pilote NVIDIA compatible et NVIDIA Container Toolkit ; il n’est pas nécessaire d’installer CUDA, cuDNN, ORT, Python ou OpenCV sur l’hôte.

```bash
mkdir -p server/.models
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml run --rm models install buffalo_l
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

Pour le GPU, utilisez `compose.cuda12.yml` et le port `18098`. L’installateur affiche la licence avant téléchargement ; les modèles publics InsightFace sont réservés à la recherche non commerciale sans licence commerciale distincte.

Le Compose fourni désactive l’authentification par défaut pour une évaluation isolée. Avant toute exposition réseau, définissez `INSIGHTFACE_AUTH_ENABLED=true` et un long `INSIGHTFACE_API_KEY`. Vérifiez ensuite le Dashboard, créez une Collection, inscrivez une Person et recherchez-la avec une autre image. Arrêtez avec `docker compose ... down` sans `-v` pour conserver le volume.

## 1. Connexion et état

Ouvrez `http://SERVEUR:18097/` pour le CPU ou `http://SERVEUR:18098/` pour CUDA 12. Si l’authentification est active, choisissez **Configurer la clé API**, collez la clé fournie et appliquez-la à l’onglet. Elle reste uniquement en mémoire et disparaît au rechargement ou à la fermeture.

Dans **Tableau de bord** ou **Système**, vérifiez que service, base, modèles et Provider sont prêts. CUDA doit afficher `CUDAExecutionProvider` et ne bascule jamais silencieusement sur CPU.

## 2. Créer une Collection

Dans **Collections** → **Nouvelle collection**, définissez un ID stable, un nom,
le seuil cosinus (`0.4` au départ), un profil disponible, la capacité et le
nombre maximal de FaceSamples par personne. La conservation JPEG d’un
`bounding-box crop` redimensionné en 112×112 est désactivée par défaut ; ce
n’est pas l’entrée alignée du modèle de reconnaissance.

La Collection est liée à l’ID, la version, le digest, la dimension et le prétraitement du modèle. Après un changement de modèle, elle reste visible mais inscription et recherche sont refusées si le contrat diffère.

Le profil de détection copie les valeurs système à la création, puis permet de modifier tailles d’entrée, seuils détection/NMS et stratégie mono-visage. `largest` privilégie la surface ; `center_largest` maximise `surface - 2,0 × distance en pixels au carré entre le centre du cadre et celui de l’image`. La confiance de détection ne participe pas à ce score.

## 3. Inscrire une Person

Dans **Personnes**, sélectionnez la Collection puis **Inscrire une personne**. Saisissez éventuellement ID, nom, ID externe, metadata JSON et une ou plusieurs images JPEG, PNG ou WebP.

- `off` : utilise la stratégie mono-visage de la Collection et autorise plusieurs visages ;
- `standard` : impose un visage exploitable et contrôle taille, détection, netteté, luminosité et pose ;
- `strict` : impose aussi que la meilleure similarité interne soit supérieure à la meilleure similarité externe.

Un lot accepte un succès partiel et détaille chaque rejet. Les originaux ne sont pas stockés. `external_trusted` accepte un embedding normalisé L2 ; l’image reste obligatoire pour détection et qualité, mais le vecteur n’est pas réextrait.

## 4. Détecter, comparer et rechercher

**Détecter** affiche boîtes, cinq points, score et qualité ; aucun visage renvoie une liste vide valide. **Comparer** utilise le profil système ou Collection pour choisir un visage par image et renvoie `similarity` cosinus, `threshold` et `matched`. La similarité n’est pas une probabilité.

Dans **Rechercher**, choisissez Collection et image. Le score d’une personne est la meilleure similarité de ses FaceSamples. Les résultats sont triés par ordre décroissant ; aucun résultat donne une liste vide. Chaque échantillon est d’abord validé dans SQLite puis ajouté à l’index avant la réponse. Au redémarrage, l’index est reconstruit depuis SQLite.

## 5. Surveillance de caméra RTSP

Dans **Surveillance caméra**, créez un Monitor persistant et configurez source RTSP, Collection, fréquence, seuil facultatif et politique d’événements. L’aperçu est désactivé par défaut ; reconnaissance et événements continuent sans lui. Lorsqu’il est actif, la Web UI trace les inscrits en vert et les inconnus en orange depuis `/state` sur des images brutes.

Le Monitor fonctionne indépendamment du navigateur et les tâches actives sont restaurées après redémarrage. La configuration est dans SQLite et les identifiants RTSP sont chiffrés dans `/data`, mais images et événements ne sont pas enregistrés. Les événements restent seulement dans un tampon mémoire borné. Le décodeur garde l’image la plus récente et ignore les anciennes au lieu de les empiler.

## 6. Données et sécurité

Persistez `/data` et montez `/models` en lecture seule. Sauvegardez ensemble SQLite et les recadrages avant une opération massive. Les clés sont hashées ; redémarrer le même volume avec un autre `INSIGHTFACE_API_KEY` fait tourner la clé active. Ne journalisez ni images, ni embeddings, ni clés.

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
`w600k_r50.onnx`), `buffalo_m`, `buffalo_sc` et `antelopev2`. L’installation
crée `manifest.json` et le fichier signé `MODEL.LICENSE`. Sans
`--accept-license`, l’outil affiche les conditions puis s’arrête sans
télécharger. Les modèles préentraînés publics InsightFace sont réservés à la
recherche non commerciale sans licence commerciale séparée.

## 8. Configuration de démarrage et recherche

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

Tout utilisateur peut construire depuis le dépôt complet :

```bash
make -C server build-cpu
make -C server build-cuda12
```

Ajoutez `--pull never` aux commandes Compose pour employer l’image locale. Les
tags immuables sont `0.2.0-cpu` et `0.2.0-cuda12`; `cpu` et `cuda12` suivent la
dernière version stable et aucun `latest` n’est publié. Avant mise à niveau,
arrêtez les écritures et sauvegardez `/data` et les crops avec une méthode sûre
pour SQLite. N’utilisez pas `docker compose down -v`, qui supprime le volume.

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
