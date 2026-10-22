import { SHARED_UI_MESSAGES, UI_CATALOGS } from "./ui-translations.mjs?v=0.3.1-r1";

export const LANGUAGES = Object.freeze([
  { code: "en", label: "English", htmlLang: "en" },
  { code: "zh", label: "中文", htmlLang: "zh-CN" },
  { code: "ja", label: "日本語", htmlLang: "ja" },
  { code: "de", label: "Deutsch", htmlLang: "de" },
  { code: "es", label: "Español", htmlLang: "es" },
  { code: "fr", label: "Français", htmlLang: "fr" },
  { code: "ru", label: "Русский", htmlLang: "ru" },
  { code: "pt", label: "Português", htmlLang: "pt" },
  { code: "ko", label: "한국어", htmlLang: "ko" },
]);

const catalogs = {
  zh: {
    "Skip to content": "跳到主要内容", "Primary navigation": "主导航", "InsightFace Server dashboard": "InsightFace Server 仪表盘",
    "Server console": "服务器控制台", Connecting: "正在连接", Dashboard: "仪表盘", Collections: "人员库", People: "人员",
    Detect: "检测", Compare: "比对", Search: "搜索", "Live camera": "实时摄像头", System: "系统", Help: "帮助",
    "API docs": "API 文档", "Configure API key": "配置 API Key", "Toggle navigation": "切换导航", "No API key": "未设置 API Key",
    "Refresh current page": "刷新当前页面", Refresh: "刷新", "LOCAL FACE INTELLIGENCE": "本地人脸智能",
    "Your face recognition service,": "您的人脸识别服务，", "at a glance.": "一目了然。", "Execution provider": "执行 Provider",
    "Database totals": "数据库统计", "Face samples": "人脸样本", Model: "模型", SERVICE: "服务", "Runtime status": "运行状态",
    "View diagnostics →": "查看诊断 →", "THIS TAB": "当前标签页", "Recent errors": "最近错误", Clear: "清除",
    "No recent errors": "没有最近错误", "IDENTITY SPACES": "身份空间", "＋ New collection": "＋ 新建人员库",
    CREATE: "创建", "New collection": "新建人员库", "Collection ID": "人员库 ID", Name: "名称", Description: "描述",
    "Default cosine threshold": "默认余弦阈值", "Search profile": "搜索配置", "Capacity (faces)": "容量（人脸）",
    "Maximum faces per person": "每人最多人脸数", "Load policy": "加载策略", "Server default": "服务器默认", Lazy: "延迟加载", Eager: "启动加载",
    "Save 112×112 face crops in SQLite": "在 SQLite 中保存 112×112 人脸裁剪图", "Metadata (JSON)": "元数据（JSON）",
    "Create collection": "创建人员库", "Filter collections": "筛选人员库", Collection: "人员库", Threshold: "阈值", Profile: "配置",
    Crops: "裁剪图", Faces: "人脸", Actions: "操作", "No collections yet": "还没有人员库", ENROLLMENT: "人员注册",
    "People & face samples": "人员与人脸样本", "＋ Register person": "＋ 注册人员", "Select a collection": "选择人员库",
    "Search people": "搜索人员", REGISTER: "注册", "New person": "新人员", "Person ID": "人员 ID", optional: "可选",
    "External ID": "外部 ID", "Enrollment review": "入库审查", "Off · largest face": "关闭 · 最大人脸", "Standard quality": "标准质量",
    "Strict identity": "严格身份", "Embedding source": "特征来源", "Extract from image": "从图片抽取", "Trusted external feature": "可信外部特征",
    "Drop registration photos here": "将注册照片拖到这里", "Register person": "注册人员", DIRECTORY: "人员目录", "No person selected": "未选择人员",
    STATELESS: "无状态", "Detect faces": "检测人脸", "Drop an image": "拖入图片", "Maximum faces": "最大人脸数", "Minimum score": "最低分数",
    "Image preview": "图片预览", "ONE-TO-ONE": "一对一", "Compare two faces": "比对两张人脸", "Source image": "源图片", "Target image": "目标图片",
    "Choose source": "选择源图片", "Choose target": "选择目标图片", "Match threshold": "匹配阈值", "Compare faces": "比对人脸",
    "Waiting for images": "等待图片", "ONE-TO-MANY": "一对多", "Search a collection": "搜索人员库", "Drop a query image": "拖入查询图片",
    "Top results": "返回数量", Default: "默认", "Query preview": "查询预览", "BROWSER DEMO": "浏览器演示", "Camera is off": "摄像头已关闭",
    Mode: "模式", "Detect only": "仅检测", "Frame rate": "帧率", "Start camera": "启动摄像头", "Stop camera": "停止摄像头",
    ADMINISTRATION: "管理", "Refresh system": "刷新系统", Server: "服务器", Provider: "Provider", Accelerator: "加速器",
    "API key in UI": "UI 中的 API Key", "Not configured": "未配置", RUNTIME: "运行时", "Hardware & software": "硬件与软件",
    STORAGE: "存储", "Database & mounts": "数据库与挂载", MODELS: "模型", "Loaded models": "已加载模型", AUTHENTICATION: "认证",
    "API key": "API Key", "Paste API key": "粘贴 API Key", "Clear key": "清除 Key", "Use for this tab": "在此标签页使用",
    "Edit collection": "编辑人员库", Cancel: "取消", "Save changes": "保存修改", README: "README", "User guide": "用户指南",
    "Open document": "打开文档", "Documentation": "文档", "Could not load documentation.": "无法加载文档。",
    "The key stays in this tab's memory and is never saved by the UI.": "Key 仅保留在当前标签页内存中，UI 不会保存它。",
    "Similarity is not a probability.": "相似度不是概率。", "No files selected": "未选择文件", "JPEG, PNG, WebP, or BMP": "JPEG、PNG、WebP 或 BMP",
    "Detector input sizes": "检测输入尺寸", "Detection threshold": "检测阈值", "NMS threshold": "NMS 阈值", "Single-face selection": "单脸挑选策略",
    "Largest face": "最大脸", "Center-weighted largest": "中心加权最大脸", "Detection profile": "检测配置", "Off · collection selection": "关闭 · 使用人员库策略",
  },
  ja: {
    "Skip to content": "メインコンテンツへ", "Primary navigation": "メインナビゲーション", "Server console": "サーバーコンソール", Connecting: "接続中",
    Dashboard: "ダッシュボード", Collections: "コレクション", People: "人物", Detect: "検出", Compare: "比較", Search: "検索",
    "Live camera": "ライブカメラ", System: "システム", Help: "ヘルプ", "API docs": "API ドキュメント", "Configure API key": "API キーを設定",
    "No API key": "API キーなし", "Refresh current page": "現在のページを更新", Refresh: "更新", "Execution provider": "実行プロバイダー",
    "Face samples": "顔サンプル", Model: "モデル", "Runtime status": "実行状態", "Recent errors": "最近のエラー", Clear: "クリア",
    "No recent errors": "最近のエラーはありません", "＋ New collection": "＋ 新規コレクション", "New collection": "新規コレクション",
    "Collection ID": "コレクション ID", Name: "名前", Description: "説明", "Default cosine threshold": "既定のコサインしきい値",
    "Search profile": "検索プロファイル", "Capacity (faces)": "容量（顔）", "Maximum faces per person": "人物ごとの最大顔数",
    "Load policy": "ロードポリシー", "Server default": "サーバー既定", Lazy: "遅延", Eager: "起動時", "Metadata (JSON)": "メタデータ（JSON）",
    "Create collection": "コレクションを作成", Collection: "コレクション", Threshold: "しきい値", Profile: "プロファイル", Faces: "顔",
    ENROLLMENT: "登録", "People & face samples": "人物と顔サンプル", "＋ Register person": "＋ 人物を登録", "Select a collection": "コレクションを選択",
    "Search people": "人物を検索", "New person": "新規人物", "Person ID": "人物 ID", optional: "任意", "External ID": "外部 ID",
    "Enrollment review": "登録レビュー", "Off · largest face": "オフ · 最大の顔", "Standard quality": "標準品質", "Strict identity": "厳格な本人確認",
    "Embedding source": "特徴量ソース", "Extract from image": "画像から抽出", "Trusted external feature": "信頼済み外部特徴量",
    "Drop registration photos here": "登録写真をここにドロップ", "Register person": "人物を登録", "No person selected": "人物が選択されていません",
    "Detect faces": "顔を検出", "Drop an image": "画像をドロップ", "Maximum faces": "最大顔数", "Minimum score": "最小スコア",
    "Compare two faces": "2つの顔を比較", "Source image": "ソース画像", "Target image": "ターゲット画像", "Choose source": "ソースを選択",
    "Choose target": "ターゲットを選択", "Match threshold": "一致しきい値", "Compare faces": "顔を比較", "Waiting for images": "画像待機中",
    "Search a collection": "コレクションを検索", "Drop a query image": "検索画像をドロップ", "Top results": "上位結果", Default: "既定",
    "Camera is off": "カメラはオフです", Mode: "モード", "Detect only": "検出のみ", "Frame rate": "フレームレート",
    "Start camera": "カメラを開始", "Stop camera": "カメラを停止", ADMINISTRATION: "管理", "Refresh system": "システムを更新",
    Server: "サーバー", Provider: "プロバイダー", Accelerator: "アクセラレーター", "Not configured": "未設定", RUNTIME: "ランタイム",
    "Hardware & software": "ハードウェアとソフトウェア", STORAGE: "ストレージ", "Database & mounts": "データベースとマウント",
    "Loaded models": "ロード済みモデル", AUTHENTICATION: "認証", "API key": "API キー", "Paste API key": "API キーを貼り付け",
    "Clear key": "キーをクリア", "Use for this tab": "このタブで使用", "Edit collection": "コレクションを編集", Cancel: "キャンセル",
    "Save changes": "変更を保存", README: "README", "User guide": "ユーザーガイド", Documentation: "ドキュメント", "Open document": "ドキュメントを開く",
    "Could not load documentation.": "ドキュメントを読み込めませんでした。", "Similarity is not a probability.": "類似度は確率ではありません。",
    "No files selected": "ファイル未選択", "JPEG, PNG, WebP, or BMP": "JPEG、PNG、WebP、または BMP",
    "Detector input sizes": "検出入力サイズ", "Detection threshold": "検出しきい値", "NMS threshold": "NMS しきい値", "Single-face selection": "単一顔の選択",
    "Largest face": "最大の顔", "Center-weighted largest": "中心重み付き最大", "Detection profile": "検出プロファイル", "Off · collection selection": "オフ · Collection の選択",
  },
  de: {
    "Skip to content": "Zum Inhalt springen", "Primary navigation": "Hauptnavigation", "Server console": "Serverkonsole", Connecting: "Verbindung wird hergestellt",
    Dashboard: "Übersicht", Collections: "Sammlungen", People: "Personen", Detect: "Erkennen", Compare: "Vergleichen", Search: "Suchen",
    "Live camera": "Live-Kamera", System: "System", Help: "Hilfe", "API docs": "API-Dokumentation", "Configure API key": "API-Schlüssel konfigurieren",
    "No API key": "Kein API-Schlüssel", "Refresh current page": "Aktuelle Seite neu laden", Refresh: "Aktualisieren", "Execution provider": "Ausführungs-Provider",
    "Face samples": "Gesichtsproben", Model: "Modell", "Runtime status": "Laufzeitstatus", "Recent errors": "Letzte Fehler", Clear: "Leeren",
    "No recent errors": "Keine aktuellen Fehler", "＋ New collection": "＋ Neue Sammlung", "New collection": "Neue Sammlung", "Collection ID": "Sammlungs-ID",
    Name: "Name", Description: "Beschreibung", "Default cosine threshold": "Standard-Cosinus-Schwelle", "Search profile": "Suchprofil",
    "Capacity (faces)": "Kapazität (Gesichter)", "Maximum faces per person": "Maximale Gesichter pro Person", "Load policy": "Ladestrategie",
    "Server default": "Serverstandard", Lazy: "Verzögert", Eager: "Beim Start", "Metadata (JSON)": "Metadaten (JSON)",
    "Create collection": "Sammlung erstellen", Collection: "Sammlung", Threshold: "Schwelle", Profile: "Profil", Faces: "Gesichter",
    "People & face samples": "Personen und Gesichtsproben", "＋ Register person": "＋ Person registrieren", "Select a collection": "Sammlung auswählen",
    "Search people": "Personen suchen", "New person": "Neue Person", "Person ID": "Personen-ID", optional: "optional", "External ID": "Externe ID",
    "Enrollment review": "Registrierungsprüfung", "Off · largest face": "Aus · größtes Gesicht", "Standard quality": "Standardqualität", "Strict identity": "Strikte Identität",
    "Embedding source": "Embedding-Quelle", "Extract from image": "Aus Bild extrahieren", "Trusted external feature": "Vertrauenswürdiges externes Merkmal",
    "Drop registration photos here": "Registrierungsfotos hier ablegen", "Register person": "Person registrieren", "No person selected": "Keine Person ausgewählt",
    "Detect faces": "Gesichter erkennen", "Drop an image": "Bild hier ablegen", "Maximum faces": "Maximale Gesichter", "Minimum score": "Mindestwert",
    "Compare two faces": "Zwei Gesichter vergleichen", "Source image": "Quellbild", "Target image": "Zielbild", "Choose source": "Quelle wählen",
    "Choose target": "Ziel wählen", "Match threshold": "Übereinstimmungsschwelle", "Compare faces": "Gesichter vergleichen", "Waiting for images": "Warte auf Bilder",
    "Search a collection": "Sammlung durchsuchen", "Drop a query image": "Suchbild hier ablegen", "Top results": "Top-Ergebnisse", Default: "Standard",
    "Camera is off": "Kamera ist aus", Mode: "Modus", "Detect only": "Nur erkennen", "Frame rate": "Bildrate", "Start camera": "Kamera starten",
    "Stop camera": "Kamera stoppen", ADMINISTRATION: "VERWALTUNG", "Refresh system": "System aktualisieren", Server: "Server", Provider: "Provider",
    Accelerator: "Beschleuniger", "Not configured": "Nicht konfiguriert", RUNTIME: "LAUFZEIT", "Hardware & software": "Hardware und Software",
    STORAGE: "SPEICHER", "Database & mounts": "Datenbank und Einbindungen", "Loaded models": "Geladene Modelle", AUTHENTICATION: "AUTHENTIFIZIERUNG",
    "API key": "API-Schlüssel", "Paste API key": "API-Schlüssel einfügen", "Clear key": "Schlüssel löschen", "Use for this tab": "Für diesen Tab verwenden",
    "Edit collection": "Sammlung bearbeiten", Cancel: "Abbrechen", "Save changes": "Änderungen speichern", README: "README", "User guide": "Benutzerhandbuch",
    Documentation: "Dokumentation", "Open document": "Dokument öffnen", "Could not load documentation.": "Dokumentation konnte nicht geladen werden.",
    "Similarity is not a probability.": "Ähnlichkeit ist keine Wahrscheinlichkeit.", "No files selected": "Keine Dateien ausgewählt", "JPEG, PNG, WebP, or BMP": "JPEG, PNG, WebP oder BMP",
    "Detector input sizes": "Detektor-Eingabegrößen", "Detection threshold": "Erkennungsschwelle", "NMS threshold": "NMS-Schwelle", "Single-face selection": "Ein-Gesicht-Auswahl",
    "Largest face": "Größtes Gesicht", "Center-weighted largest": "Zentrumsgewichtet größtes", "Detection profile": "Erkennungsprofil", "Off · collection selection": "Aus · Collection-Auswahl",
  },
  es: {
    "Skip to content": "Ir al contenido", "Primary navigation": "Navegación principal", "Server console": "Consola del servidor", Connecting: "Conectando",
    Dashboard: "Panel", Collections: "Colecciones", People: "Personas", Detect: "Detectar", Compare: "Comparar", Search: "Buscar",
    "Live camera": "Cámara en vivo", System: "Sistema", Help: "Ayuda", "API docs": "Documentación API", "Configure API key": "Configurar clave API",
    "No API key": "Sin clave API", "Refresh current page": "Actualizar página actual", Refresh: "Actualizar", "Execution provider": "Proveedor de ejecución",
    "Face samples": "Muestras faciales", Model: "Modelo", "Runtime status": "Estado de ejecución", "Recent errors": "Errores recientes", Clear: "Limpiar",
    "No recent errors": "No hay errores recientes", "＋ New collection": "＋ Nueva colección", "New collection": "Nueva colección", "Collection ID": "ID de colección",
    Name: "Nombre", Description: "Descripción", "Default cosine threshold": "Umbral coseno predeterminado", "Search profile": "Perfil de búsqueda",
    "Capacity (faces)": "Capacidad (rostros)", "Maximum faces per person": "Máximo de rostros por persona", "Load policy": "Política de carga",
    "Server default": "Predeterminado del servidor", Lazy: "Diferida", Eager: "Al iniciar", "Metadata (JSON)": "Metadatos (JSON)",
    "Create collection": "Crear colección", Collection: "Colección", Threshold: "Umbral", Profile: "Perfil", Faces: "Rostros",
    "People & face samples": "Personas y muestras faciales", "＋ Register person": "＋ Registrar persona", "Select a collection": "Seleccionar colección",
    "Search people": "Buscar personas", "New person": "Nueva persona", "Person ID": "ID de persona", optional: "opcional", "External ID": "ID externo",
    "Enrollment review": "Revisión de registro", "Off · largest face": "Desactivada · rostro mayor", "Standard quality": "Calidad estándar", "Strict identity": "Identidad estricta",
    "Embedding source": "Origen del embedding", "Extract from image": "Extraer de la imagen", "Trusted external feature": "Vector externo confiable",
    "Drop registration photos here": "Suelta aquí las fotos de registro", "Register person": "Registrar persona", "No person selected": "Ninguna persona seleccionada",
    "Detect faces": "Detectar rostros", "Drop an image": "Suelta una imagen", "Maximum faces": "Máximo de rostros", "Minimum score": "Puntuación mínima",
    "Compare two faces": "Comparar dos rostros", "Source image": "Imagen de origen", "Target image": "Imagen de destino", "Choose source": "Elegir origen",
    "Choose target": "Elegir destino", "Match threshold": "Umbral de coincidencia", "Compare faces": "Comparar rostros", "Waiting for images": "Esperando imágenes",
    "Search a collection": "Buscar en una colección", "Drop a query image": "Suelta una imagen de consulta", "Top results": "Mejores resultados", Default: "Predeterminado",
    "Camera is off": "La cámara está apagada", Mode: "Modo", "Detect only": "Solo detectar", "Frame rate": "Fotogramas por segundo",
    "Start camera": "Iniciar cámara", "Stop camera": "Detener cámara", ADMINISTRATION: "ADMINISTRACIÓN", "Refresh system": "Actualizar sistema",
    Server: "Servidor", Provider: "Proveedor", Accelerator: "Acelerador", "Not configured": "Sin configurar", RUNTIME: "EJECUCIÓN",
    "Hardware & software": "Hardware y software", STORAGE: "ALMACENAMIENTO", "Database & mounts": "Base de datos y montajes", "Loaded models": "Modelos cargados",
    AUTHENTICATION: "AUTENTICACIÓN", "API key": "Clave API", "Paste API key": "Pegar clave API", "Clear key": "Borrar clave",
    "Use for this tab": "Usar en esta pestaña", "Edit collection": "Editar colección", Cancel: "Cancelar", "Save changes": "Guardar cambios",
    README: "README", "User guide": "Guía de usuario", Documentation: "Documentación", "Open document": "Abrir documento",
    "Could not load documentation.": "No se pudo cargar la documentación.", "Similarity is not a probability.": "La similitud no es una probabilidad.",
    "No files selected": "No hay archivos seleccionados", "JPEG, PNG, WebP, or BMP": "JPEG, PNG, WebP o BMP",
    "Detector input sizes": "Tamaños de entrada del detector", "Detection threshold": "Umbral de detección", "NMS threshold": "Umbral NMS", "Single-face selection": "Selección de un rostro",
    "Largest face": "Rostro mayor", "Center-weighted largest": "Mayor ponderado por centro", "Detection profile": "Perfil de detección", "Off · collection selection": "Desactivada · selección de Collection",
  },
  fr: {
    "Skip to content": "Aller au contenu", "Primary navigation": "Navigation principale", "Server console": "Console serveur", Connecting: "Connexion",
    Dashboard: "Tableau de bord", Collections: "Collections", People: "Personnes", Detect: "Détecter", Compare: "Comparer", Search: "Rechercher",
    "Live camera": "Caméra en direct", System: "Système", Help: "Aide", "API docs": "Documentation API", "Configure API key": "Configurer la clé API",
    "No API key": "Aucune clé API", "Refresh current page": "Actualiser la page", Refresh: "Actualiser", "Execution provider": "Provider d’exécution",
    "Face samples": "Échantillons faciaux", Model: "Modèle", "Runtime status": "État d’exécution", "Recent errors": "Erreurs récentes", Clear: "Effacer",
    "No recent errors": "Aucune erreur récente", "＋ New collection": "＋ Nouvelle collection", "New collection": "Nouvelle collection", "Collection ID": "ID de collection",
    Name: "Nom", Description: "Description", "Default cosine threshold": "Seuil cosinus par défaut", "Search profile": "Profil de recherche",
    "Capacity (faces)": "Capacité (visages)", "Maximum faces per person": "Nombre maximal de visages par personne", "Load policy": "Politique de chargement",
    "Server default": "Valeur serveur", Lazy: "Différé", Eager: "Au démarrage", "Metadata (JSON)": "Métadonnées (JSON)",
    "Create collection": "Créer la collection", Collection: "Collection", Threshold: "Seuil", Profile: "Profil", Faces: "Visages",
    "People & face samples": "Personnes et échantillons faciaux", "＋ Register person": "＋ Inscrire une personne", "Select a collection": "Choisir une collection",
    "Search people": "Rechercher des personnes", "New person": "Nouvelle personne", "Person ID": "ID de personne", optional: "facultatif", "External ID": "ID externe",
    "Enrollment review": "Contrôle d’inscription", "Off · largest face": "Désactivé · plus grand visage", "Standard quality": "Qualité standard", "Strict identity": "Identité stricte",
    "Embedding source": "Source de l’embedding", "Extract from image": "Extraire de l’image", "Trusted external feature": "Vecteur externe approuvé",
    "Drop registration photos here": "Déposez les photos d’inscription ici", "Register person": "Inscrire la personne", "No person selected": "Aucune personne sélectionnée",
    "Detect faces": "Détecter les visages", "Drop an image": "Déposez une image", "Maximum faces": "Nombre maximal de visages", "Minimum score": "Score minimal",
    "Compare two faces": "Comparer deux visages", "Source image": "Image source", "Target image": "Image cible", "Choose source": "Choisir la source",
    "Choose target": "Choisir la cible", "Match threshold": "Seuil de correspondance", "Compare faces": "Comparer les visages", "Waiting for images": "En attente d’images",
    "Search a collection": "Rechercher dans une collection", "Drop a query image": "Déposez une image de recherche", "Top results": "Meilleurs résultats", Default: "Par défaut",
    "Camera is off": "La caméra est arrêtée", Mode: "Mode", "Detect only": "Détection uniquement", "Frame rate": "Fréquence d’images",
    "Start camera": "Démarrer la caméra", "Stop camera": "Arrêter la caméra", ADMINISTRATION: "ADMINISTRATION", "Refresh system": "Actualiser le système",
    Server: "Serveur", Provider: "Provider", Accelerator: "Accélérateur", "Not configured": "Non configuré", RUNTIME: "EXÉCUTION",
    "Hardware & software": "Matériel et logiciel", STORAGE: "STOCKAGE", "Database & mounts": "Base de données et montages", "Loaded models": "Modèles chargés",
    AUTHENTICATION: "AUTHENTIFICATION", "API key": "Clé API", "Paste API key": "Coller la clé API", "Clear key": "Effacer la clé",
    "Use for this tab": "Utiliser dans cet onglet", "Edit collection": "Modifier la collection", Cancel: "Annuler", "Save changes": "Enregistrer",
    README: "README", "User guide": "Guide utilisateur", Documentation: "Documentation", "Open document": "Ouvrir le document",
    "Could not load documentation.": "Impossible de charger la documentation.", "Similarity is not a probability.": "La similarité n’est pas une probabilité.",
    "No files selected": "Aucun fichier sélectionné", "JPEG, PNG, WebP, or BMP": "JPEG, PNG, WebP ou BMP",
    "Detector input sizes": "Tailles d’entrée du détecteur", "Detection threshold": "Seuil de détection", "NMS threshold": "Seuil NMS", "Single-face selection": "Sélection mono-visage",
    "Largest face": "Plus grand visage", "Center-weighted largest": "Plus grand pondéré par le centre", "Detection profile": "Profil de détection", "Off · collection selection": "Désactivé · sélection Collection",
  },
  ru: {
    "Skip to content": "Перейти к содержимому", "Primary navigation": "Основная навигация", "Server console": "Консоль сервера", Connecting: "Подключение",
    Dashboard: "Панель", Collections: "Коллекции", People: "Люди", Detect: "Детекция", Compare: "Сравнение", Search: "Поиск",
    "Live camera": "Камера", System: "Система", Help: "Помощь", "API docs": "Документация API", "Configure API key": "Настроить API-ключ",
    "No API key": "Нет API-ключа", "Refresh current page": "Обновить страницу", Refresh: "Обновить", "Execution provider": "Provider выполнения",
    "Face samples": "Образцы лиц", Model: "Модель", "Runtime status": "Состояние", "Recent errors": "Последние ошибки", Clear: "Очистить",
    "No recent errors": "Недавних ошибок нет", "＋ New collection": "＋ Новая коллекция", "New collection": "Новая коллекция", "Collection ID": "ID коллекции",
    Name: "Имя", Description: "Описание", "Default cosine threshold": "Порог cosine по умолчанию", "Search profile": "Профиль поиска",
    "Capacity (faces)": "Ёмкость (лица)", "Maximum faces per person": "Максимум лиц на человека", "Load policy": "Политика загрузки",
    "Server default": "Настройка сервера", Lazy: "Ленивая", Eager: "При запуске", "Metadata (JSON)": "Метаданные (JSON)",
    "Create collection": "Создать коллекцию", Collection: "Коллекция", Threshold: "Порог", Profile: "Профиль", Faces: "Лица",
    "People & face samples": "Люди и образцы лиц", "＋ Register person": "＋ Зарегистрировать человека", "Select a collection": "Выберите коллекцию",
    "Search people": "Поиск людей", "New person": "Новый человек", "Person ID": "ID человека", optional: "необязательно", "External ID": "Внешний ID",
    "Enrollment review": "Проверка регистрации", "Off · largest face": "Выкл. · крупнейшее лицо", "Standard quality": "Стандартное качество", "Strict identity": "Строгая идентичность",
    "Embedding source": "Источник embedding", "Extract from image": "Извлечь из изображения", "Trusted external feature": "Доверенный внешний вектор",
    "Drop registration photos here": "Перетащите сюда фотографии", "Register person": "Зарегистрировать", "No person selected": "Человек не выбран",
    "Detect faces": "Обнаружить лица", "Drop an image": "Перетащите изображение", "Maximum faces": "Максимум лиц", "Minimum score": "Минимальный score",
    "Compare two faces": "Сравнить два лица", "Source image": "Исходное изображение", "Target image": "Целевое изображение", "Choose source": "Выбрать исходное",
    "Choose target": "Выбрать целевое", "Match threshold": "Порог совпадения", "Compare faces": "Сравнить лица", "Waiting for images": "Ожидание изображений",
    "Search a collection": "Поиск по коллекции", "Drop a query image": "Перетащите изображение запроса", "Top results": "Лучшие результаты", Default: "По умолчанию",
    "Camera is off": "Камера выключена", Mode: "Режим", "Detect only": "Только детекция", "Frame rate": "Частота кадров",
    "Start camera": "Включить камеру", "Stop camera": "Остановить камеру", ADMINISTRATION: "АДМИНИСТРИРОВАНИЕ", "Refresh system": "Обновить систему",
    Server: "Сервер", Provider: "Provider", Accelerator: "Ускоритель", "Not configured": "Не настроено", RUNTIME: "СРЕДА",
    "Hardware & software": "Аппаратное и программное обеспечение", STORAGE: "ХРАНИЛИЩЕ", "Database & mounts": "База данных и точки монтирования",
    "Loaded models": "Загруженные модели", AUTHENTICATION: "АУТЕНТИФИКАЦИЯ", "API key": "API-ключ", "Paste API key": "Вставьте API-ключ",
    "Clear key": "Очистить ключ", "Use for this tab": "Использовать в этой вкладке", "Edit collection": "Изменить коллекцию", Cancel: "Отмена",
    "Save changes": "Сохранить", README: "README", "User guide": "Руководство", Documentation: "Документация", "Open document": "Открыть документ",
    "Could not load documentation.": "Не удалось загрузить документацию.", "Similarity is not a probability.": "Сходство не является вероятностью.",
    "No files selected": "Файлы не выбраны", "JPEG, PNG, WebP, or BMP": "JPEG, PNG, WebP или BMP",
    "Detector input sizes": "Входные размеры детектора", "Detection threshold": "Порог детекции", "NMS threshold": "Порог NMS", "Single-face selection": "Выбор одного лица",
    "Largest face": "Крупнейшее лицо", "Center-weighted largest": "Наибольшее с весом центра", "Detection profile": "Профиль детекции", "Off · collection selection": "Выкл. · выбор Collection",
  },
  pt: {
    "Skip to content": "Ir para o conteúdo", "Primary navigation": "Navegação principal", "Server console": "Consola do servidor", Connecting: "A ligar",
    Dashboard: "Painel", Collections: "Coleções", People: "Pessoas", Detect: "Detetar", Compare: "Comparar", Search: "Pesquisar",
    "Live camera": "Câmara ao vivo", System: "Sistema", Help: "Ajuda", "API docs": "Documentação API", "Configure API key": "Configurar chave API",
    "No API key": "Sem chave API", "Refresh current page": "Atualizar página", Refresh: "Atualizar", "Execution provider": "Provider de execução",
    "Face samples": "Amostras faciais", Model: "Modelo", "Runtime status": "Estado de execução", "Recent errors": "Erros recentes", Clear: "Limpar",
    "No recent errors": "Sem erros recentes", "＋ New collection": "＋ Nova coleção", "New collection": "Nova coleção", "Collection ID": "ID da coleção",
    Name: "Nome", Description: "Descrição", "Default cosine threshold": "Limiar cosine predefinido", "Search profile": "Perfil de pesquisa",
    "Capacity (faces)": "Capacidade (rostos)", "Maximum faces per person": "Máximo de rostos por pessoa", "Load policy": "Política de carregamento",
    "Server default": "Predefinição do servidor", Lazy: "Adiado", Eager: "No arranque", "Metadata (JSON)": "Metadados (JSON)",
    "Create collection": "Criar coleção", Collection: "Coleção", Threshold: "Limiar", Profile: "Perfil", Faces: "Rostos",
    "People & face samples": "Pessoas e amostras faciais", "＋ Register person": "＋ Registar pessoa", "Select a collection": "Selecionar coleção",
    "Search people": "Pesquisar pessoas", "New person": "Nova pessoa", "Person ID": "ID da pessoa", optional: "opcional", "External ID": "ID externo",
    "Enrollment review": "Revisão de registo", "Off · largest face": "Desativada · maior rosto", "Standard quality": "Qualidade padrão", "Strict identity": "Identidade rigorosa",
    "Embedding source": "Origem do embedding", "Extract from image": "Extrair da imagem", "Trusted external feature": "Vetor externo confiável",
    "Drop registration photos here": "Largue aqui as fotos de registo", "Register person": "Registar pessoa", "No person selected": "Nenhuma pessoa selecionada",
    "Detect faces": "Detetar rostos", "Drop an image": "Largue uma imagem", "Maximum faces": "Máximo de rostos", "Minimum score": "Pontuação mínima",
    "Compare two faces": "Comparar dois rostos", "Source image": "Imagem de origem", "Target image": "Imagem de destino", "Choose source": "Escolher origem",
    "Choose target": "Escolher destino", "Match threshold": "Limiar de correspondência", "Compare faces": "Comparar rostos", "Waiting for images": "À espera de imagens",
    "Search a collection": "Pesquisar numa coleção", "Drop a query image": "Largue uma imagem de consulta", "Top results": "Melhores resultados", Default: "Predefinido",
    "Camera is off": "A câmara está desligada", Mode: "Modo", "Detect only": "Apenas detetar", "Frame rate": "Taxa de fotogramas",
    "Start camera": "Iniciar câmara", "Stop camera": "Parar câmara", ADMINISTRATION: "ADMINISTRAÇÃO", "Refresh system": "Atualizar sistema",
    Server: "Servidor", Provider: "Provider", Accelerator: "Acelerador", "Not configured": "Não configurado", RUNTIME: "EXECUÇÃO",
    "Hardware & software": "Hardware e software", STORAGE: "ARMAZENAMENTO", "Database & mounts": "Base de dados e montagens", "Loaded models": "Modelos carregados",
    AUTHENTICATION: "AUTENTICAÇÃO", "API key": "Chave API", "Paste API key": "Colar chave API", "Clear key": "Limpar chave",
    "Use for this tab": "Usar neste separador", "Edit collection": "Editar coleção", Cancel: "Cancelar", "Save changes": "Guardar alterações",
    README: "README", "User guide": "Guia do utilizador", Documentation: "Documentação", "Open document": "Abrir documento",
    "Could not load documentation.": "Não foi possível carregar a documentação.", "Similarity is not a probability.": "A similaridade não é uma probabilidade.",
    "No files selected": "Nenhum ficheiro selecionado", "JPEG, PNG, WebP, or BMP": "JPEG, PNG, WebP ou BMP",
    "Detector input sizes": "Tamanhos de entrada do detetor", "Detection threshold": "Limiar de deteção", "NMS threshold": "Limiar NMS", "Single-face selection": "Seleção de um rosto",
    "Largest face": "Maior rosto", "Center-weighted largest": "Maior com peso central", "Detection profile": "Perfil de deteção", "Off · collection selection": "Desativado · seleção da Collection",
  },
  ko: {
    "Skip to content": "본문으로 이동", "Primary navigation": "주요 탐색", "Server console": "서버 콘솔", Connecting: "연결 중",
    Dashboard: "대시보드", Collections: "컬렉션", People: "사람", Detect: "검출", Compare: "비교", Search: "검색",
    "Live camera": "라이브 카메라", System: "시스템", Help: "도움말", "API docs": "API 문서", "Configure API key": "API 키 설정",
    "No API key": "API 키 없음", "Refresh current page": "현재 페이지 새로고침", Refresh: "새로고침", "Execution provider": "실행 Provider",
    "Face samples": "얼굴 샘플", Model: "모델", "Runtime status": "실행 상태", "Recent errors": "최근 오류", Clear: "지우기",
    "No recent errors": "최근 오류 없음", "＋ New collection": "＋ 새 컬렉션", "New collection": "새 컬렉션", "Collection ID": "컬렉션 ID",
    Name: "이름", Description: "설명", "Default cosine threshold": "기본 cosine 임계값", "Search profile": "검색 프로필",
    "Capacity (faces)": "용량(얼굴)", "Maximum faces per person": "사람별 최대 얼굴 수", "Load policy": "로드 정책",
    "Server default": "서버 기본값", Lazy: "지연", Eager: "시작 시", "Metadata (JSON)": "메타데이터(JSON)",
    "Create collection": "컬렉션 만들기", Collection: "컬렉션", Threshold: "임계값", Profile: "프로필", Faces: "얼굴",
    "People & face samples": "사람 및 얼굴 샘플", "＋ Register person": "＋ 사람 등록", "Select a collection": "컬렉션 선택",
    "Search people": "사람 검색", "New person": "새 사람", "Person ID": "사람 ID", optional: "선택", "External ID": "외부 ID",
    "Enrollment review": "등록 검토", "Off · largest face": "끄기 · 가장 큰 얼굴", "Standard quality": "표준 품질", "Strict identity": "엄격한 신원",
    "Embedding source": "임베딩 소스", "Extract from image": "이미지에서 추출", "Trusted external feature": "신뢰된 외부 특징",
    "Drop registration photos here": "등록 사진을 여기에 놓으세요", "Register person": "사람 등록", "No person selected": "선택된 사람이 없습니다",
    "Detect faces": "얼굴 검출", "Drop an image": "이미지를 놓으세요", "Maximum faces": "최대 얼굴 수", "Minimum score": "최소 점수",
    "Compare two faces": "두 얼굴 비교", "Source image": "원본 이미지", "Target image": "대상 이미지", "Choose source": "원본 선택",
    "Choose target": "대상 선택", "Match threshold": "일치 임계값", "Compare faces": "얼굴 비교", "Waiting for images": "이미지 대기 중",
    "Search a collection": "컬렉션 검색", "Drop a query image": "검색 이미지를 놓으세요", "Top results": "상위 결과", Default: "기본값",
    "Camera is off": "카메라가 꺼져 있습니다", Mode: "모드", "Detect only": "검출만", "Frame rate": "프레임 속도",
    "Start camera": "카메라 시작", "Stop camera": "카메라 중지", ADMINISTRATION: "관리", "Refresh system": "시스템 새로고침",
    Server: "서버", Provider: "Provider", Accelerator: "가속기", "Not configured": "설정되지 않음", RUNTIME: "런타임",
    "Hardware & software": "하드웨어 및 소프트웨어", STORAGE: "스토리지", "Database & mounts": "데이터베이스 및 마운트", "Loaded models": "로드된 모델",
    AUTHENTICATION: "인증", "API key": "API 키", "Paste API key": "API 키 붙여넣기", "Clear key": "키 지우기",
    "Use for this tab": "이 탭에서 사용", "Edit collection": "컬렉션 편집", Cancel: "취소", "Save changes": "변경 저장",
    README: "README", "User guide": "사용자 가이드", Documentation: "문서", "Open document": "문서 열기",
    "Could not load documentation.": "문서를 불러올 수 없습니다.", "Similarity is not a probability.": "유사도는 확률이 아닙니다.",
    "No files selected": "선택된 파일 없음", "JPEG, PNG, WebP, or BMP": "JPEG, PNG, WebP 또는 BMP",
    "Detector input sizes": "검출기 입력 크기", "Detection threshold": "검출 임계값", "NMS threshold": "NMS 임계값", "Single-face selection": "단일 얼굴 선택",
    "Largest face": "가장 큰 얼굴", "Center-weighted largest": "중심 가중 최대 얼굴", "Detection profile": "검출 프로필", "Off · collection selection": "끄기 · Collection 선택",
  },
};

const supplementalCatalogs = {
  zh: {
    Language: "语言", "Version-matched documentation is bundled with this server and works without internet access.": "与当前版本匹配的文档已内置，无需联网即可阅读。",
    "InsightFace Server dashboard": "InsightFace Server 仪表盘", "Detect, check liveness, compare, enroll, and search without sending images outside your server.": "无需将图片发送到服务器之外，即可完成人脸检测、活体检测、比对、注册和搜索。",
    "separate identity spaces": "独立身份空间", "enrolled identities": "已注册身份", "stored embeddings": "已存特征", "API readiness": "API 就绪状态", Database: "数据库",
    "API failures from this browser tab will appear here.": "此浏览器标签页中的 API 错误会显示在这里。", "tab memory only": "仅保留在标签页内存",
    "The displayed language follows the console language selector.": "文档语言跟随控制台语言选择。", "Loading documentation…": "正在加载文档…",
    "API key configured": "API Key 已配置", Configured: "已配置", "Ready to compare": "可以开始比对", Close: "关闭",
    "Open console": "打开控制台", "OpenAPI JSON": "OpenAPI JSON", "API Reference": "API 参考", "Simple face recognition APIs.": "简单的人脸识别 API。",
    "Base path": "基础路径", "API operations": "API 操作", "Filter operations": "筛选操作", "Loading API schema": "正在加载 API Schema",
    COMPONENTS: "组件", "Data schemas": "数据 Schema", Parameters: "参数", required: "必填", "Request body": "请求体", Responses: "响应",
    "Open-source model license": "开源模型许可", "InsightFace-provided open-source pretrained models, including buffalo_l, are for non-commercial research use only.": "InsightFace 提供的开源预训练模型（包括 buffalo_l）仅限非商业研究使用。", "Commercial use requires a separate license.": "商业使用需要单独许可。", "Commercial licensing": "商业许可信息",
  },
  ja: {
    Language: "言語", "Version-matched documentation is bundled with this server and works without internet access.": "このサーバーと同じバージョンのドキュメントが内蔵され、オフラインで利用できます。",
    "InsightFace Server dashboard": "InsightFace Server ダッシュボード", "The key stays in this tab's memory and is never saved by the UI.": "Key はこのタブのメモリだけに保持され、UI には保存されません。",
    "Detect, check liveness, compare, enroll, and search without sending images outside your server.": "画像をサーバー外へ送信せずに、検出、生体検知、比較、登録、検索を実行できます。", "separate identity spaces": "独立した ID 空間", "enrolled identities": "登録済み ID",
    "stored embeddings": "保存済み特徴量", "API readiness": "API 準備状態", Database: "データベース", "API failures from this browser tab will appear here.": "このタブの API エラーがここに表示されます。", "tab memory only": "タブのメモリのみ",
    "The displayed language follows the console language selector.": "文書の言語はコンソールの言語設定に従います。", "Loading documentation…": "ドキュメントを読み込み中…",
    "API key configured": "API キー設定済み", Configured: "設定済み", "Ready to compare": "比較できます", Close: "閉じる",
    "Open console": "コンソールを開く", "API Reference": "API リファレンス", "Simple face recognition APIs.": "シンプルな顔認識 API。", "Base path": "ベースパス",
    "API operations": "API 操作", "Filter operations": "操作を絞り込み", "Loading API schema": "API スキーマを読み込み中", COMPONENTS: "コンポーネント",
    "Data schemas": "データスキーマ", Parameters: "パラメーター", required: "必須", "Request body": "リクエスト本文", Responses: "レスポンス",
    "Open-source model license": "オープンソースモデルのライセンス", "InsightFace-provided open-source pretrained models, including buffalo_l, are for non-commercial research use only.": "buffalo_l を含む InsightFace 提供のオープンソース事前学習モデルは、非商用の研究目的に限り利用できます。", "Commercial use requires a separate license.": "商用利用には別途ライセンスが必要です。", "Commercial licensing": "商用ライセンス",
  },
  de: {
    Language: "Sprache", "Version-matched documentation is bundled with this server and works without internet access.": "Zur Serverversion passende Dokumentation ist enthalten und offline verfügbar.",
    "InsightFace Server dashboard": "InsightFace Server Übersicht", "The key stays in this tab's memory and is never saved by the UI.": "Der Key bleibt nur im Speicher dieses Tabs und wird von der UI nicht gespeichert.",
    "Detect, check liveness, compare, enroll, and search without sending images outside your server.": "Erkennen, Lebenderkennung durchführen, vergleichen, registrieren und suchen, ohne Bilder vom Server zu übertragen.", "separate identity spaces": "getrennte Identitätsräume", "enrolled identities": "registrierte Identitäten",
    "stored embeddings": "gespeicherte Embeddings", "API readiness": "API-Bereitschaft", Database: "Datenbank", "API failures from this browser tab will appear here.": "API-Fehler dieses Browser-Tabs erscheinen hier.", "tab memory only": "nur Tab-Speicher",
    "The displayed language follows the console language selector.": "Die Dokumentsprache folgt der Sprachauswahl der Konsole.", "Loading documentation…": "Dokumentation wird geladen…",
    "API key configured": "API-Schlüssel konfiguriert", Configured: "Konfiguriert", "Ready to compare": "Bereit zum Vergleichen", Close: "Schließen",
    "Open console": "Konsole öffnen", "API Reference": "API-Referenz", "Simple face recognition APIs.": "Einfache Gesichtserkennungs-APIs.", "Base path": "Basispfad",
    "API operations": "API-Operationen", "Filter operations": "Operationen filtern", "Loading API schema": "API-Schema wird geladen", COMPONENTS: "KOMPONENTEN",
    "Data schemas": "Datenschemas", Parameters: "Parameter", required: "erforderlich", "Request body": "Request-Body", Responses: "Antworten",
    "Open-source model license": "Lizenz für Open-Source-Modelle", "InsightFace-provided open-source pretrained models, including buffalo_l, are for non-commercial research use only.": "Von InsightFace bereitgestellte Open-Source-Modelle, einschließlich buffalo_l, dürfen nur für nichtkommerzielle Forschung verwendet werden.", "Commercial use requires a separate license.": "Die kommerzielle Nutzung erfordert eine separate Lizenz.", "Commercial licensing": "Kommerzielle Lizenzierung",
  },
  es: {
    Language: "Idioma", "Version-matched documentation is bundled with this server and works without internet access.": "La documentación correspondiente a esta versión está integrada y funciona sin internet.",
    "InsightFace Server dashboard": "Panel de InsightFace Server", "The key stays in this tab's memory and is never saved by the UI.": "La clave solo permanece en la memoria de esta pestaña y la UI no la guarda.",
    "Detect, check liveness, compare, enroll, and search without sending images outside your server.": "Detecte, realice pruebas de vida, compare, registre y busque sin enviar imágenes fuera de su servidor.", "separate identity spaces": "espacios de identidad separados", "enrolled identities": "identidades registradas",
    "stored embeddings": "vectores guardados", "API readiness": "disponibilidad API", Database: "Base de datos", "API failures from this browser tab will appear here.": "Los errores API de esta pestaña aparecerán aquí.", "tab memory only": "solo memoria de pestaña",
    "The displayed language follows the console language selector.": "El idioma del documento sigue el selector de la consola.", "Loading documentation…": "Cargando documentación…",
    "API key configured": "Clave API configurada", Configured: "Configurada", "Ready to compare": "Listo para comparar", Close: "Cerrar",
    "Open console": "Abrir consola", "API Reference": "Referencia API", "Simple face recognition APIs.": "API sencillas de reconocimiento facial.", "Base path": "Ruta base",
    "API operations": "Operaciones API", "Filter operations": "Filtrar operaciones", "Loading API schema": "Cargando esquema API", COMPONENTS: "COMPONENTES",
    "Data schemas": "Esquemas de datos", Parameters: "Parámetros", required: "obligatorio", "Request body": "Cuerpo de solicitud", Responses: "Respuestas",
    "Open-source model license": "Licencia del modelo de código abierto", "InsightFace-provided open-source pretrained models, including buffalo_l, are for non-commercial research use only.": "Los modelos preentrenados de código abierto proporcionados por InsightFace, incluido buffalo_l, son solo para investigación no comercial.", "Commercial use requires a separate license.": "El uso comercial requiere una licencia independiente.", "Commercial licensing": "Licencia comercial",
  },
  fr: {
    Language: "Langue", "Version-matched documentation is bundled with this server and works without internet access.": "La documentation correspondant à cette version est intégrée et disponible hors ligne.",
    "InsightFace Server dashboard": "Tableau de bord InsightFace Server", "The key stays in this tab's memory and is never saved by the UI.": "La clé reste uniquement en mémoire dans cet onglet et n'est pas enregistrée par l'interface.",
    "Detect, check liveness, compare, enroll, and search without sending images outside your server.": "Détectez, contrôlez le caractère vivant, comparez, inscrivez et recherchez sans envoyer les images hors de votre serveur.", "separate identity spaces": "espaces d'identité séparés", "enrolled identities": "identités inscrites",
    "stored embeddings": "embeddings enregistrés", "API readiness": "disponibilité API", Database: "Base de données", "API failures from this browser tab will appear here.": "Les erreurs API de cet onglet apparaîtront ici.", "tab memory only": "mémoire de l'onglet uniquement",
    "The displayed language follows the console language selector.": "La langue du document suit le sélecteur de la console.", "Loading documentation…": "Chargement de la documentation…",
    "API key configured": "Clé API configurée", Configured: "Configurée", "Ready to compare": "Prêt à comparer", Close: "Fermer",
    "Open console": "Ouvrir la console", "API Reference": "Référence API", "Simple face recognition APIs.": "API simples de reconnaissance faciale.", "Base path": "Chemin de base",
    "API operations": "Opérations API", "Filter operations": "Filtrer les opérations", "Loading API schema": "Chargement du schéma API", COMPONENTS: "COMPOSANTS",
    "Data schemas": "Schémas de données", Parameters: "Paramètres", required: "obligatoire", "Request body": "Corps de requête", Responses: "Réponses",
    "Open-source model license": "Licence du modèle open source", "InsightFace-provided open-source pretrained models, including buffalo_l, are for non-commercial research use only.": "Les modèles préentraînés open source fournis par InsightFace, dont buffalo_l, sont réservés à la recherche non commerciale.", "Commercial use requires a separate license.": "L’utilisation commerciale nécessite une licence distincte.", "Commercial licensing": "Licence commerciale",
  },
  ru: {
    Language: "Язык", "Version-matched documentation is bundled with this server and works without internet access.": "Документация этой версии встроена в сервер и доступна без интернета.",
    "InsightFace Server dashboard": "Панель InsightFace Server", "The key stays in this tab's memory and is never saved by the UI.": "Ключ хранится только в памяти вкладки и не сохраняется интерфейсом.",
    "Detect, check liveness, compare, enroll, and search without sending images outside your server.": "Выполняйте детекцию, проверку живого лица, сравнение, регистрацию и поиск, не отправляя изображения за пределы сервера.", "separate identity spaces": "отдельные пространства идентичностей", "enrolled identities": "зарегистрированные личности",
    "stored embeddings": "сохранённые векторы", "API readiness": "готовность API", Database: "База данных", "API failures from this browser tab will appear here.": "Ошибки API этой вкладки появятся здесь.", "tab memory only": "только память вкладки",
    "The displayed language follows the console language selector.": "Язык документа соответствует выбору в консоли.", "Loading documentation…": "Загрузка документации…",
    "API key configured": "API-ключ настроен", Configured: "Настроено", "Ready to compare": "Готово к сравнению", Close: "Закрыть",
    "Open console": "Открыть консоль", "API Reference": "Справочник API", "Simple face recognition APIs.": "Простые API распознавания лиц.", "Base path": "Базовый путь",
    "API operations": "Операции API", "Filter operations": "Фильтр операций", "Loading API schema": "Загрузка схемы API", COMPONENTS: "КОМПОНЕНТЫ",
    "Data schemas": "Схемы данных", Parameters: "Параметры", required: "обязательно", "Request body": "Тело запроса", Responses: "Ответы",
    "Open-source model license": "Лицензия модели с открытым исходным кодом", "InsightFace-provided open-source pretrained models, including buffalo_l, are for non-commercial research use only.": "Предоставленные InsightFace открытые предобученные модели, включая buffalo_l, предназначены только для некоммерческих исследований.", "Commercial use requires a separate license.": "Для коммерческого использования требуется отдельная лицензия.", "Commercial licensing": "Коммерческое лицензирование",
  },
  pt: {
    Language: "Idioma", "Version-matched documentation is bundled with this server and works without internet access.": "A documentação desta versão está incluída e funciona sem internet.",
    "InsightFace Server dashboard": "Painel do InsightFace Server", "The key stays in this tab's memory and is never saved by the UI.": "A chave fica apenas na memória deste separador e não é guardada pela interface.",
    "Detect, check liveness, compare, enroll, and search without sending images outside your server.": "Detete, verifique a vivacidade, compare, registe e pesquise sem enviar imagens para fora do seu servidor.", "separate identity spaces": "espaços de identidade separados", "enrolled identities": "identidades registadas",
    "stored embeddings": "vetores guardados", "API readiness": "disponibilidade da API", Database: "Base de dados", "API failures from this browser tab will appear here.": "Os erros API deste separador aparecerão aqui.", "tab memory only": "apenas memória do separador",
    "The displayed language follows the console language selector.": "O idioma do documento segue o seletor da consola.", "Loading documentation…": "A carregar documentação…",
    "API key configured": "Chave API configurada", Configured: "Configurada", "Ready to compare": "Pronto para comparar", Close: "Fechar",
    "Open console": "Abrir consola", "API Reference": "Referência API", "Simple face recognition APIs.": "APIs simples de reconhecimento facial.", "Base path": "Caminho base",
    "API operations": "Operações API", "Filter operations": "Filtrar operações", "Loading API schema": "A carregar esquema API", COMPONENTS: "COMPONENTES",
    "Data schemas": "Esquemas de dados", Parameters: "Parâmetros", required: "obrigatório", "Request body": "Corpo do pedido", Responses: "Respostas",
    "Open-source model license": "Licença do modelo open source", "InsightFace-provided open-source pretrained models, including buffalo_l, are for non-commercial research use only.": "Os modelos pré-treinados open source fornecidos pela InsightFace, incluindo buffalo_l, destinam-se apenas a investigação não comercial.", "Commercial use requires a separate license.": "A utilização comercial requer uma licença separada.", "Commercial licensing": "Licenciamento comercial",
  },
  ko: {
    Language: "언어", "Version-matched documentation is bundled with this server and works without internet access.": "현재 서버 버전에 맞는 문서가 내장되어 오프라인에서도 작동합니다.",
    "InsightFace Server dashboard": "InsightFace Server 대시보드", "The key stays in this tab's memory and is never saved by the UI.": "Key는 현재 탭의 메모리에만 있으며 UI에 저장되지 않습니다.",
    "Detect, check liveness, compare, enroll, and search without sending images outside your server.": "이미지를 서버 밖으로 보내지 않고 검출, 라이브니스 검사, 비교, 등록, 검색을 수행할 수 있습니다.", "separate identity spaces": "분리된 신원 공간", "enrolled identities": "등록된 신원",
    "stored embeddings": "저장된 임베딩", "API readiness": "API 준비 상태", Database: "데이터베이스", "API failures from this browser tab will appear here.": "이 브라우저 탭의 API 오류가 여기에 표시됩니다.", "tab memory only": "탭 메모리에만 저장",
    "The displayed language follows the console language selector.": "문서 언어는 콘솔 언어 선택을 따릅니다.", "Loading documentation…": "문서를 불러오는 중…",
    "API key configured": "API 키 설정됨", Configured: "설정됨", "Ready to compare": "비교 준비 완료", Close: "닫기",
    "Open console": "콘솔 열기", "API Reference": "API 참조", "Simple face recognition APIs.": "간단한 얼굴 인식 API.", "Base path": "기본 경로",
    "API operations": "API 작업", "Filter operations": "작업 필터", "Loading API schema": "API 스키마 불러오는 중", COMPONENTS: "컴포넌트",
    "Data schemas": "데이터 스키마", Parameters: "매개변수", required: "필수", "Request body": "요청 본문", Responses: "응답",
    "Open-source model license": "오픈 소스 모델 라이선스", "InsightFace-provided open-source pretrained models, including buffalo_l, are for non-commercial research use only.": "buffalo_l을 포함하여 InsightFace가 제공하는 오픈 소스 사전 학습 모델은 비상업적 연구 용도로만 사용할 수 있습니다.", "Commercial use requires a separate license.": "상업적 사용에는 별도의 라이선스가 필요합니다.", "Commercial licensing": "상업용 라이선스",
  },
};

const videoCatalogs = {
  zh: {
    "Video recognition": "视频识别", "LOCAL VIDEO": "本地视频", "Play a local video and identify every detected face. The video stays in your browser; only sampled JPEG frames are sent to this server.": "播放本地视频并识别每张检测到的人脸。视频保留在浏览器中，仅向服务器发送采样的 JPEG 帧。",
    "Video recognition face overlays": "视频识别人脸框", "No video selected": "未选择视频", "Choose a local video. It stays in this browser; only sampled JPEG frames are sent for recognition.": "选择本地视频。视频保留在浏览器中，仅发送采样的 JPEG 帧进行识别。",
    "Choose a video": "选择视频", "MP4, WebM, MOV, or MKV supported by this browser": "浏览器支持的 MP4、WebM、MOV 或 MKV", "No file selected": "未选择文件", "blank = collection default": "留空 = 人员库默认值", "Sampling rate": "采样频率", "Maximum faces per frame": "每帧最多人脸数",
    "Start recognition": "开始识别", "Stop recognition": "停止识别", "Select a video, then start recognition.": "选择视频，然后开始识别。", Unrecognized: "未识别", "Below threshold": "低于阈值",
    "Recognition running": "正在识别", "Video is playing. Sampled frames are processed without saving the video.": "视频正在播放。系统处理采样帧，但不会保存视频。", "Recognition stopped. The video remains in this browser.": "识别已停止。视频仍仅保留在浏览器中。",
    "Video finished. The last recognition result remains visible.": "视频播放结束，最后一次识别结果仍会保留。", "Video selected. Press Start recognition to begin.": "视频已选择，点击“开始识别”即可运行。", "ID {id}": "ID {id}", "ID {id} · {similarity}": "ID {id} · {similarity}", "{faces} faces · {recognized} recognized · {duration}": "检测 {faces} 张人脸 · 识别 {recognized} 张 · {duration}",
  },
  ja: {
    "Video recognition": "動画認識", "LOCAL VIDEO": "ローカル動画", "No video selected": "動画が選択されていません", "Choose a video": "動画を選択", "Sampling rate": "サンプリング頻度", "Maximum faces per frame": "フレームごとの最大顔数", "Start recognition": "認識を開始", "Stop recognition": "認識を停止", "Select a video, then start recognition.": "動画を選択して認識を開始してください。", Unrecognized: "未認識", "Below threshold": "しきい値未満", "Recognition running": "認識中",
  },
  de: {
    "Video recognition": "Videoerkennung", "LOCAL VIDEO": "LOKALES VIDEO", "No video selected": "Kein Video ausgewählt", "Choose a video": "Video auswählen", "Sampling rate": "Abtastrate", "Maximum faces per frame": "Maximale Gesichter pro Bild", "Start recognition": "Erkennung starten", "Stop recognition": "Erkennung stoppen", "Select a video, then start recognition.": "Video auswählen und dann die Erkennung starten.", Unrecognized: "Nicht erkannt", "Below threshold": "Unter dem Schwellenwert", "Recognition running": "Erkennung läuft",
  },
  es: {
    "Video recognition": "Reconocimiento de vídeo", "LOCAL VIDEO": "VÍDEO LOCAL", "No video selected": "Ningún vídeo seleccionado", "Choose a video": "Elegir vídeo", "Sampling rate": "Frecuencia de muestreo", "Maximum faces per frame": "Máximo de rostros por fotograma", "Start recognition": "Iniciar reconocimiento", "Stop recognition": "Detener reconocimiento", "Select a video, then start recognition.": "Seleccione un vídeo y luego inicie el reconocimiento.", Unrecognized: "Sin reconocer", "Below threshold": "Por debajo del umbral", "Recognition running": "Reconocimiento en curso",
  },
  fr: {
    "Video recognition": "Reconnaissance vidéo", "LOCAL VIDEO": "VIDÉO LOCALE", "No video selected": "Aucune vidéo sélectionnée", "Choose a video": "Choisir une vidéo", "Sampling rate": "Fréquence d’échantillonnage", "Maximum faces per frame": "Visages maximum par image", "Start recognition": "Démarrer la reconnaissance", "Stop recognition": "Arrêter la reconnaissance", "Select a video, then start recognition.": "Choisissez une vidéo, puis démarrez la reconnaissance.", Unrecognized: "Non reconnu", "Below threshold": "Sous le seuil", "Recognition running": "Reconnaissance en cours",
  },
  ru: {
    "Video recognition": "Распознавание видео", "LOCAL VIDEO": "ЛОКАЛЬНОЕ ВИДЕО", "No video selected": "Видео не выбрано", "Choose a video": "Выбрать видео", "Sampling rate": "Частота выборки", "Maximum faces per frame": "Максимум лиц в кадре", "Start recognition": "Начать распознавание", "Stop recognition": "Остановить распознавание", "Select a video, then start recognition.": "Выберите видео и запустите распознавание.", Unrecognized: "Не распознано", "Below threshold": "Ниже порога", "Recognition running": "Идёт распознавание",
  },
  pt: {
    "Video recognition": "Reconhecimento de vídeo", "LOCAL VIDEO": "VÍDEO LOCAL", "No video selected": "Nenhum vídeo selecionado", "Choose a video": "Escolher vídeo", "Sampling rate": "Taxa de amostragem", "Maximum faces per frame": "Máximo de rostos por fotograma", "Start recognition": "Iniciar reconhecimento", "Stop recognition": "Parar reconhecimento", "Select a video, then start recognition.": "Selecione um vídeo e inicie o reconhecimento.", Unrecognized: "Não reconhecido", "Below threshold": "Abaixo do limiar", "Recognition running": "Reconhecimento em curso",
  },
  ko: {
    "Video recognition": "동영상 인식", "LOCAL VIDEO": "로컬 동영상", "No video selected": "선택된 동영상 없음", "Choose a video": "동영상 선택", "Sampling rate": "샘플링 속도", "Maximum faces per frame": "프레임당 최대 얼굴 수", "Start recognition": "인식 시작", "Stop recognition": "인식 중지", "Select a video, then start recognition.": "동영상을 선택한 다음 인식을 시작하세요.", Unrecognized: "인식되지 않음", "Below threshold": "임계값 미만", "Recognition running": "인식 중",
  },
};

let activeLocale = "en";
const originalText = new WeakMap();
const originalAttributes = new WeakMap();
const catalogLayers = [UI_CATALOGS, videoCatalogs, supplementalCatalogs, catalogs];

function catalogTemplate(language, source) {
  for (const catalog of catalogLayers) {
    if (Object.prototype.hasOwnProperty.call(catalog[language] ?? {}, source)) return catalog[language][source];
  }
  return undefined;
}

function escapePattern(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function compileLocalizedTemplate(source, localized) {
  const variables = [];
  let cursor = 0;
  let pattern = "";
  for (const match of localized.matchAll(/\{([A-Za-z0-9_]+)\}/g)) {
    pattern += escapePattern(localized.slice(cursor, match.index));
    pattern += "(.+?)";
    variables.push(match[1]);
    cursor = match.index + match[0].length;
  }
  if (!variables.length) return null;
  pattern += escapePattern(localized.slice(cursor));
  return { source, variables, pattern: new RegExp(`^${pattern}$`) };
}

const reverseExactTranslations = new Map();
const reverseTemplateTranslations = [];
for (const language of LANGUAGES.filter(({ code }) => code !== "en")) {
  for (const catalog of catalogLayers) {
    for (const [source, localized] of Object.entries(catalog[language.code] ?? {})) {
      if (localized === source) continue;
      if (!reverseExactTranslations.has(localized)) reverseExactTranslations.set(localized, source);
      const compiled = compileLocalizedTemplate(source, localized);
      if (compiled) reverseTemplateTranslations.push(compiled);
    }
  }
}

function canonicalMessage(value) {
  const source = String(value);
  const exact = reverseExactTranslations.get(source);
  if (exact) return { source: exact, variables: {} };
  for (const candidate of reverseTemplateTranslations) {
    const match = candidate.pattern.exec(source);
    if (!match) continue;
    return {
      source: candidate.source,
      variables: Object.fromEntries(candidate.variables.map((name, index) => [name, match[index + 1]])),
    };
  }
  return { source, variables: {} };
}

export function normalizeLocale(value) {
  const candidate = String(value || "").trim().toLowerCase().replace("_", "-");
  const base = candidate.split("-")[0];
  return LANGUAGES.some((language) => language.code === base) ? base : "en";
}

export function detectLocale({ stored, languages = [] } = {}) {
  if (stored && LANGUAGES.some((language) => language.code === stored)) return stored;
  for (const language of languages) {
    const normalized = normalizeLocale(language);
    if (normalized !== "en" || String(language).toLowerCase().startsWith("en")) return normalized;
  }
  return "en";
}

export function locale() {
  return activeLocale;
}

export function t(source, variables = {}, language = activeLocale) {
  const direct = catalogTemplate(language, source);
  const canonical = direct === undefined ? canonicalMessage(source) : { source, variables: {} };
  const template = direct ?? catalogTemplate(language, canonical.source) ?? source;
  const values = { ...canonical.variables, ...variables };
  return String(template).replace(/\{([A-Za-z0-9_]+)\}/g, (_, name) => String(values[name] ?? `{${name}}`));
}

export function hasTranslation(source, language) {
  if (language === "en" || SHARED_UI_MESSAGES.has(source)) return true;
  return catalogLayers
    .some((catalog) => Object.prototype.hasOwnProperty.call(catalog[language] ?? {}, source));
}

function translateTextNode(node) {
  if (!originalText.has(node)) originalText.set(node, node.nodeValue || "");
  const source = originalText.get(node);
  const trimmed = source.trim();
  if (!trimmed) return;
  const translated = t(trimmed);
  node.nodeValue = source.replace(trimmed, translated);
}

function translateAttributes(element) {
  const names = ["aria-label", "placeholder", "title"];
  if (!originalAttributes.has(element)) originalAttributes.set(element, {});
  const saved = originalAttributes.get(element);
  for (const name of names) {
    if (!element.hasAttribute(name)) continue;
    if (!(name in saved)) saved[name] = element.getAttribute(name);
    element.setAttribute(name, t(saved[name]));
  }
}

function translationExcluded(node) {
  const element = node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
  return Boolean(element?.closest?.('[translate="no"], script, style, code, pre, textarea'));
}

export function translateTree(root = document) {
  if (translationExcluded(root)) return;
  if (root.nodeType === Node.TEXT_NODE) {
    translateTextNode(root);
    return;
  }
  if (!(root instanceof Element || root instanceof Document)) return;
  if (root instanceof Element) translateAttributes(root);
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
  let node = walker.nextNode();
  while (node) {
    if (translationExcluded(node)) {
      node = walker.nextNode();
      continue;
    }
    if (node.nodeType === Node.TEXT_NODE) {
      const parent = node.parentElement;
      if (parent) translateTextNode(node);
    } else {
      translateAttributes(node);
    }
    node = walker.nextNode();
  }
}

export function setLocale(language, { persist = true, announce = true } = {}) {
  activeLocale = normalizeLocale(language);
  const metadata = LANGUAGES.find((item) => item.code === activeLocale) ?? LANGUAGES[0];
  document.documentElement.lang = metadata.htmlLang;
  if (persist) {
    try { window.localStorage.setItem("insightface.locale", activeLocale); } catch { /* The language still applies in this tab. */ }
  }
  translateTree(document);
  const selector = document.querySelector("#language-select");
  if (selector) selector.value = activeLocale;
  if (announce) window.dispatchEvent(new CustomEvent("insightface:localechange", { detail: { locale: activeLocale } }));
  return activeLocale;
}

export function initializeI18n() {
  let stored = null;
  try {
    stored = window.localStorage.getItem("insightface.locale");
  } catch {
    stored = null;
  }
  const selected = detectLocale({ stored, languages: navigator.languages || [navigator.language] });
  setLocale(selected, { persist: false, announce: false });
  const selector = document.querySelector("#language-select");
  selector?.addEventListener("change", () => setLocale(selector.value));
  return selected;
}
