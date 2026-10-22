"""Single-page enterprise 1:1 and 1:N evaluation workflow."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..core.evaluation import (
    MULTI_FACE_REQUIRE_ONE,
    MULTI_FACE_SKIP,
    MULTI_FACE_USE_CENTERED_LARGEST,
    MULTI_FACE_USE_LARGEST,
    run_identity_identification_evaluation,
    run_identity_verification_evaluation,
    validate_enterprise_dataset,
)
from ..core.i18n import apply_translations, effective_language, tr
from ..core.licensing import current_model_license_display
from ..core.reporting import write_reports
from ..widgets.drop_input import DropInput
from .base import BasePage


DATASET_RULES_TEXT = """Enterprise Evaluation Dataset Rules

The Enterprise Evaluation page supports local 1:1 verification and 1:N identification evaluation from identity folders. No images, embeddings, or reports are uploaded automatically.

1:1 Verification

Auto Split enabled:

dataset_1v1/
  0001__Alice/
    gallery.jpg
    img002.jpg
    img003.jpg
  0002__Bob/
    img001.jpg
    img002.jpg

Each subfolder is one identity. A file whose name contains "gallery" is selected as that identity's gallery image. If no such file exists, the first sorted image is selected as gallery. All other images are probes. Evaluation compares every probe against every identity gallery, so each trial is probe vs gallery.

Auto Split disabled:

dataset_1v1/
  0001__Alice/
    img001.jpg
    img002.jpg
  0002__Bob/
    img001.jpg
    img002.jpg

Every image is treated as a probe. Evaluation runs full pairwise probe-vs-probe comparisons across all identity folders.

1:N Identification

Auto Split enabled:

dataset/
  identities/
    0001__Alice/
      gallery.jpg
      img002.jpg
      img003.jpg
    0002__Bob/
      img001.jpg
      img002.jpg

The identities/ folder is preferred when present. If it is not present, the selected dataset root may contain identity folders directly. Gallery selection follows the same Auto Split rule as 1:1.

Auto Split disabled:

dataset_1n/
  gallery/
    0001__Alice/
      enroll_001.jpg
      enroll_002.jpg
    0002__Bob/
      enroll_001.jpg
  probe/
    0001__Alice/
      test_001.jpg
      test_002.jpg
    0002__Bob/
      test_001.jpg
  unknown/
    unknown_001.jpg
    unknown_002.jpg

gallery/ and probe/ are required. unknown/ is optional. 1:N evaluation always requires gallery images.

Validation Checks

- The selected root exists and matches the selected mode.
- Required gallery and probe folders/images exist.
- Auto Split can create valid gallery/probe sets.
- 1:1 can generate both positive and negative pairs.
- 1:N probe identities have valid gallery coverage.
- Each image can be read.
- Detected face count follows the selected multi-face handling policy.

Multi-face Handling

- Require exactly one face: validation fails on multi-face images.
- Use largest face: evaluation uses the largest detected face.
- Use largest centered face: evaluation uses area - center_distance^2 * 2.0.
- Mark as skip: multi-face images are skipped; skipped gallery images can make validation fail.

Report Outputs

1:1 reports include best cosine threshold accuracy, FAR/FRR, TAR@FAR, threshold recommendations, latency, and raw result examples.

1:N reports include Top1, TAR@FAR, threshold recommendations, latency, and raw result examples.
"""

DATASET_RULES_LOCAL_TEXT = {
    "zh": """Enterprise Evaluation 页面支持从身份文件夹进行本地 1:1 验证和 1:N 识别评测。图片、特征和报告不会被自动上传。

1:1 验证

启用 Auto Split:

dataset_1v1/
  0001__Alice/
    gallery.jpg
    img002.jpg
    img003.jpg
  0002__Bob/
    img001.jpg
    img002.jpg

每个子文件夹代表一个身份。文件名中包含 "gallery" 的图片会被选为该身份的 gallery image。如果不存在这样的文件，则选择排序后的第一张图片作为 gallery。其余图片作为 probe。评测会将每个 probe 与每个身份的 gallery 做比对，因此每个 trial 都是 probe vs gallery。

关闭 Auto Split:

dataset_1v1/
  0001__Alice/
    img001.jpg
    img002.jpg
  0002__Bob/
    img001.jpg
    img002.jpg

每张图片都会被视为 probe。评测会在所有身份文件夹中的图片之间执行完整的 probe-vs-probe 两两比对。

1:N 识别

启用 Auto Split:

dataset/
  identities/
    0001__Alice/
      gallery.jpg
      img002.jpg
      img003.jpg
    0002__Bob/
      img001.jpg
      img002.jpg

如果存在 identities/ 文件夹，会优先使用它。如果不存在，所选数据集根目录也可以直接包含身份文件夹。Gallery 选择规则与 1:1 的 Auto Split 相同。

关闭 Auto Split:

dataset_1n/
  gallery/
    0001__Alice/
      enroll_001.jpg
      enroll_002.jpg
    0002__Bob/
      enroll_001.jpg
  probe/
    0001__Alice/
      test_001.jpg
      test_002.jpg
    0002__Bob/
      test_001.jpg
  unknown/
    unknown_001.jpg
    unknown_002.jpg

gallery/ 和 probe/ 是必需的。unknown/ 是可选的。1:N 评测始终需要 gallery 图片。

核验内容

- 所选根目录存在，并且与当前评测模式匹配。
- 必需的 gallery 和 probe 文件夹/图片存在。
- Auto Split 能生成有效的 gallery/probe 集合。
- 1:1 能生成正样本对和负样本对。
- 1:N 的 probe 身份有对应的有效 gallery。
- 每张需要核验的图片都可以读取。
- 检测到的人脸数量符合所选多人脸处理策略。

多人脸处理

- Require exactly one face: 多人脸图片会在核验阶段失败。
- Use largest face: 评测使用检测到的最大人脸。
- Use largest centered face: 评测使用 area - center_distance^2 * 2.0 得分最高的人脸。
- Mark as skip: 多人脸图片会被跳过；如果被跳过的是 gallery 图片，则可能导致核验失败。

报告输出

1:1 报告包含最佳 cosine 阈值下的 accuracy、FAR/FRR、TAR@FAR、阈值建议、延迟和原始结果样例。

1:N 报告包含 Top1、TAR@FAR、阈值建议、延迟和原始结果样例。""",
    "ja": """Enterprise Evaluation ページでは、ID フォルダからローカルの 1:1 検証と 1:N 識別評価を実行できます。画像、特徴量、レポートは自動アップロードされません。

1:1 検証

Auto Split 有効:

dataset_1v1/
  0001__Alice/
    gallery.jpg
    img002.jpg
    img003.jpg
  0002__Bob/
    img001.jpg
    img002.jpg

各サブフォルダは 1 つの ID を表します。ファイル名に "gallery" を含む画像がその ID の gallery image として選択されます。該当ファイルがない場合は、ソート順で最初の画像が gallery になります。それ以外の画像は probe です。評価では各 probe をすべての ID の gallery と比較するため、各 trial は probe vs gallery です。

Auto Split 無効:

dataset_1v1/
  0001__Alice/
    img001.jpg
    img002.jpg
  0002__Bob/
    img001.jpg
    img002.jpg

すべての画像が probe として扱われます。すべての ID フォルダ内の画像に対して、probe-vs-probe の総当たり比較を行います。

1:N 識別

Auto Split 有効:

dataset/
  identities/
    0001__Alice/
      gallery.jpg
      img002.jpg
      img003.jpg
    0002__Bob/
      img001.jpg
      img002.jpg

identities/ フォルダが存在する場合は優先されます。存在しない場合、選択したデータセットルートに ID フォルダを直接置くこともできます。Gallery の選択規則は 1:1 の Auto Split と同じです。

Auto Split 無効:

dataset_1n/
  gallery/
    0001__Alice/
      enroll_001.jpg
      enroll_002.jpg
    0002__Bob/
      enroll_001.jpg
  probe/
    0001__Alice/
      test_001.jpg
      test_002.jpg
    0002__Bob/
      test_001.jpg
  unknown/
    unknown_001.jpg
    unknown_002.jpg

gallery/ と probe/ は必須です。unknown/ は任意です。1:N 評価には常に gallery 画像が必要です。

検証項目

- 選択したルートが存在し、選択中の評価モードと一致していること。
- 必須の gallery/probe フォルダまたは画像が存在すること。
- Auto Split が有効な gallery/probe セットを作成できること。
- 1:1 で正例ペアと負例ペアを生成できること。
- 1:N の probe ID に有効な gallery があること。
- 検証対象の画像を読み込めること。
- 検出された顔数が選択した複数顔処理ポリシーに従うこと。

複数顔の処理

- Require exactly one face: 複数顔画像は検証で失敗します。
- Use largest face: 検出された最大の顔を使用します。
- Use largest centered face: area - center_distance^2 * 2.0 のスコアが最も高い顔を使用します。
- Mark as skip: 複数顔画像をスキップします。gallery 画像がスキップされると検証失敗になる場合があります。

レポート出力

1:1 レポートには、最適な cosine しきい値での accuracy、FAR/FRR、TAR@FAR、しきい値推奨、レイテンシ、原始結果例が含まれます。

1:N レポートには、Top1、TAR@FAR、しきい値推奨、レイテンシ、原始結果例が含まれます。""",
    "ko": """Enterprise Evaluation 페이지는 ID 폴더에서 로컬 1:1 검증 및 1:N 식별 평가를 수행합니다. 이미지, 임베딩, 보고서는 자동 업로드되지 않습니다.

1:1 검증

Auto Split 켜짐:

dataset_1v1/
  0001__Alice/
    gallery.jpg
    img002.jpg
    img003.jpg
  0002__Bob/
    img001.jpg
    img002.jpg

각 하위 폴더는 하나의 ID입니다. 파일명에 "gallery"가 포함된 이미지가 해당 ID의 gallery image로 선택됩니다. 그런 파일이 없으면 정렬된 첫 번째 이미지가 gallery가 됩니다. 나머지 이미지는 probe입니다. 평가는 각 probe를 모든 ID의 gallery와 비교하므로 각 trial은 probe vs gallery입니다.

Auto Split 꺼짐:

dataset_1v1/
  0001__Alice/
    img001.jpg
    img002.jpg
  0002__Bob/
    img001.jpg
    img002.jpg

모든 이미지는 probe로 처리됩니다. 모든 ID 폴더의 이미지에 대해 전체 probe-vs-probe 쌍 비교를 수행합니다.

1:N 식별

Auto Split 켜짐:

dataset/
  identities/
    0001__Alice/
      gallery.jpg
      img002.jpg
      img003.jpg
    0002__Bob/
      img001.jpg
      img002.jpg

identities/ 폴더가 있으면 우선 사용됩니다. 없으면 선택한 데이터셋 루트에 ID 폴더를 직접 둘 수 있습니다. Gallery 선택 규칙은 1:1 Auto Split과 동일합니다.

Auto Split 꺼짐:

dataset_1n/
  gallery/
    0001__Alice/
      enroll_001.jpg
      enroll_002.jpg
    0002__Bob/
      enroll_001.jpg
  probe/
    0001__Alice/
      test_001.jpg
      test_002.jpg
    0002__Bob/
      test_001.jpg
  unknown/
    unknown_001.jpg
    unknown_002.jpg

gallery/와 probe/는 필수입니다. unknown/은 선택 사항입니다. 1:N 평가는 항상 gallery 이미지가 필요합니다.

검증 항목

- 선택한 루트가 존재하고 선택한 평가 모드와 일치해야 합니다.
- 필요한 gallery/probe 폴더 또는 이미지가 있어야 합니다.
- Auto Split이 유효한 gallery/probe 세트를 만들 수 있어야 합니다.
- 1:1은 양성 쌍과 음성 쌍을 모두 생성할 수 있어야 합니다.
- 1:N probe ID에는 유효한 gallery가 있어야 합니다.
- 검증 대상 이미지를 읽을 수 있어야 합니다.
- 감지된 얼굴 수가 선택한 다중 얼굴 처리 정책을 따라야 합니다.

다중 얼굴 처리

- Require exactly one face: 다중 얼굴 이미지는 검증에서 실패합니다.
- Use largest face: 감지된 가장 큰 얼굴을 사용합니다.
- Use largest centered face: area - center_distance^2 * 2.0 점수가 가장 높은 얼굴을 사용합니다.
- Mark as skip: 다중 얼굴 이미지는 건너뜁니다. gallery 이미지가 건너뛰어지면 검증이 실패할 수 있습니다.

보고서 출력

1:1 보고서에는 최적 cosine 임계값의 accuracy, FAR/FRR, TAR@FAR, 임계값 권장, 지연 시간, 원시 결과 예시가 포함됩니다.

1:N 보고서에는 Top1, TAR@FAR, 임계값 권장, 지연 시간, 원시 결과 예시가 포함됩니다。""",
    "es": """La página Enterprise Evaluation admite evaluación local 1:1 y 1:N desde carpetas de identidad. Las imágenes, embeddings e informes no se suben automáticamente.

Verificación 1:1

Auto Split activado:

dataset_1v1/
  0001__Alice/
    gallery.jpg
    img002.jpg
    img003.jpg
  0002__Bob/
    img001.jpg
    img002.jpg

Cada subcarpeta representa una identidad. Una imagen cuyo nombre contiene "gallery" se selecciona como gallery image de esa identidad. Si no existe, se usa la primera imagen ordenada como gallery. Las demás imágenes son probes. La evaluación compara cada probe contra la gallery de cada identidad, por lo que cada trial es probe vs gallery.

Auto Split desactivado:

dataset_1v1/
  0001__Alice/
    img001.jpg
    img002.jpg
  0002__Bob/
    img001.jpg
    img002.jpg

Cada imagen se trata como probe. La evaluación ejecuta comparaciones completas probe-vs-probe entre todas las carpetas de identidad.

Identificación 1:N

Auto Split activado:

dataset/
  identities/
    0001__Alice/
      gallery.jpg
      img002.jpg
      img003.jpg
    0002__Bob/
      img001.jpg
      img002.jpg

La carpeta identities/ se prefiere cuando existe. Si no existe, la raíz seleccionada puede contener carpetas de identidad directamente. La selección de gallery sigue la misma regla de Auto Split que 1:1.

Auto Split desactivado:

dataset_1n/
  gallery/
    0001__Alice/
      enroll_001.jpg
      enroll_002.jpg
    0002__Bob/
      enroll_001.jpg
  probe/
    0001__Alice/
      test_001.jpg
      test_002.jpg
    0002__Bob/
      test_001.jpg
  unknown/
    unknown_001.jpg
    unknown_002.jpg

gallery/ y probe/ son obligatorias. unknown/ es opcional. La evaluación 1:N siempre requiere imágenes gallery.

Comprobaciones de validación

- La raíz seleccionada existe y coincide con el modo elegido.
- Las carpetas o imágenes gallery/probe requeridas existen.
- Auto Split puede crear conjuntos gallery/probe válidos.
- 1:1 puede generar pares positivos y negativos.
- Las identidades probe de 1:N tienen cobertura gallery válida.
- Cada imagen que debe validarse puede leerse.
- El número de caras detectadas sigue la política de manejo de múltiples caras.

Manejo de múltiples caras

- Require exactly one face: las imágenes con varias caras fallan en validación.
- Use largest face: se usa la cara detectada más grande.
- Use largest centered face: se usa la cara con mayor puntuación area - center_distance^2 * 2.0.
- Mark as skip: las imágenes con varias caras se omiten; omitir imágenes gallery puede hacer fallar la validación.

Salidas del informe

Los informes 1:1 incluyen accuracy con el mejor umbral cosine, FAR/FRR, TAR@FAR, recomendaciones de umbral, latencia y ejemplos de resultados sin procesar.

Los informes 1:N incluyen Top1, TAR@FAR, recomendaciones de umbral, latencia y ejemplos de resultados sin procesar.""",
    "fr": """La page Enterprise Evaluation prend en charge l’évaluation locale 1:1 et 1:N à partir de dossiers d’identité. Les images, embeddings et rapports ne sont pas téléversés automatiquement.

Vérification 1:1

Auto Split activé :

dataset_1v1/
  0001__Alice/
    gallery.jpg
    img002.jpg
    img003.jpg
  0002__Bob/
    img001.jpg
    img002.jpg

Chaque sous-dossier représente une identité. Une image dont le nom contient "gallery" est sélectionnée comme gallery image de cette identité. Si aucun fichier ne correspond, la première image triée est utilisée comme gallery. Les autres images sont des probes. L’évaluation compare chaque probe à la gallery de chaque identité ; chaque trial est donc probe vs gallery.

Auto Split désactivé :

dataset_1v1/
  0001__Alice/
    img001.jpg
    img002.jpg
  0002__Bob/
    img001.jpg
    img002.jpg

Chaque image est traitée comme probe. L’évaluation exécute des comparaisons probe-vs-probe complètes entre tous les dossiers d’identité.

Identification 1:N

Auto Split activé :

dataset/
  identities/
    0001__Alice/
      gallery.jpg
      img002.jpg
      img003.jpg
    0002__Bob/
      img001.jpg
      img002.jpg

Le dossier identities/ est préféré lorsqu’il existe. Sinon, la racine sélectionnée peut contenir directement les dossiers d’identité. La sélection de gallery suit la même règle Auto Split que 1:1.

Auto Split désactivé :

dataset_1n/
  gallery/
    0001__Alice/
      enroll_001.jpg
      enroll_002.jpg
    0002__Bob/
      enroll_001.jpg
  probe/
    0001__Alice/
      test_001.jpg
      test_002.jpg
    0002__Bob/
      test_001.jpg
  unknown/
    unknown_001.jpg
    unknown_002.jpg

gallery/ et probe/ sont obligatoires. unknown/ est optionnel. L’évaluation 1:N nécessite toujours des images gallery.

Contrôles de validation

- La racine sélectionnée existe et correspond au mode choisi.
- Les dossiers ou images gallery/probe requis existent.
- Auto Split peut créer des ensembles gallery/probe valides.
- 1:1 peut générer des paires positives et négatives.
- Les identités probe de 1:N disposent d’une gallery valide.
- Chaque image à valider peut être lue.
- Le nombre de visages détectés respecte la politique multi-visage sélectionnée.

Gestion multi-visage

- Require exactly one face: les images multi-visages échouent à la validation.
- Use largest face: l’évaluation utilise le plus grand visage détecté.
- Use largest centered face: l’évaluation utilise le visage au meilleur score area - center_distance^2 * 2.0.
- Mark as skip: les images multi-visages sont ignorées ; ignorer une image gallery peut faire échouer la validation.

Sorties de rapport

Les rapports 1:1 incluent l’accuracy au meilleur seuil cosine, FAR/FRR, TAR@FAR, recommandations de seuil, latence et exemples de résultats bruts.

Les rapports 1:N incluent Top1, TAR@FAR, recommandations de seuil, latence et exemples de résultats bruts.""",
    "de": """Die Seite Enterprise Evaluation unterstützt lokale 1:1-Verifikation und 1:N-Identifikation aus Identitätsordnern. Bilder, Embeddings und Berichte werden nicht automatisch hochgeladen.

1:1-Verifikation

Auto Split aktiviert:

dataset_1v1/
  0001__Alice/
    gallery.jpg
    img002.jpg
    img003.jpg
  0002__Bob/
    img001.jpg
    img002.jpg

Jeder Unterordner steht für eine Identität. Eine Datei, deren Name "gallery" enthält, wird als gallery image dieser Identität gewählt. Wenn keine solche Datei vorhanden ist, wird das erste sortierte Bild als gallery verwendet. Alle anderen Bilder sind probes. Die Evaluierung vergleicht jede probe mit der gallery jeder Identität; jeder trial ist also probe vs gallery.

Auto Split deaktiviert:

dataset_1v1/
  0001__Alice/
    img001.jpg
    img002.jpg
  0002__Bob/
    img001.jpg
    img002.jpg

Jedes Bild wird als probe behandelt. Die Evaluierung führt vollständige probe-vs-probe-Paarvergleiche über alle Identitätsordner aus.

1:N-Identifikation

Auto Split aktiviert:

dataset/
  identities/
    0001__Alice/
      gallery.jpg
      img002.jpg
      img003.jpg
    0002__Bob/
      img001.jpg
      img002.jpg

Der Ordner identities/ wird bevorzugt, wenn er vorhanden ist. Falls nicht, kann der ausgewählte Dataset-Root Identitätsordner direkt enthalten. Die Gallery-Auswahl folgt derselben Auto-Split-Regel wie bei 1:1.

Auto Split deaktiviert:

dataset_1n/
  gallery/
    0001__Alice/
      enroll_001.jpg
      enroll_002.jpg
    0002__Bob/
      enroll_001.jpg
  probe/
    0001__Alice/
      test_001.jpg
      test_002.jpg
    0002__Bob/
      test_001.jpg
  unknown/
    unknown_001.jpg
    unknown_002.jpg

gallery/ und probe/ sind erforderlich. unknown/ ist optional. Eine 1:N-Evaluierung benötigt immer gallery-Bilder.

Validierungsprüfungen

- Der ausgewählte Root existiert und passt zum gewählten Modus.
- Erforderliche gallery/probe-Ordner oder Bilder existieren.
- Auto Split kann gültige gallery/probe-Sets erzeugen.
- 1:1 kann positive und negative Paare erzeugen.
- 1:N-probe-Identitäten haben gültige gallery-Abdeckung.
- Jedes zu validierende Bild kann gelesen werden.
- Die erkannte Gesichtszahl folgt der ausgewählten Mehrgesichter-Strategie.

Mehrgesichter-Strategie

- Require exactly one face: Bilder mit mehreren Gesichtern schlagen in der Validierung fehl.
- Use largest face: Das größte erkannte Gesicht wird verwendet.
- Use largest centered face: Das Gesicht mit dem besten Score area - center_distance^2 * 2.0 wird verwendet.
- Mark as skip: Bilder mit mehreren Gesichtern werden übersprungen; übersprungene gallery-Bilder können die Validierung fehlschlagen lassen.

Berichtsausgaben

1:1-Berichte enthalten accuracy beim besten cosine-Schwellenwert, FAR/FRR, TAR@FAR, Schwellenwertempfehlungen, Latenz und Beispiele roher Ergebnisse.

1:N-Berichte enthalten Top1, TAR@FAR, Schwellenwertempfehlungen, Latenz und Beispiele roher Ergebnisse.""",
    "pt": """A página Enterprise Evaluation suporta avaliação local 1:1 e identificação 1:N a partir de pastas de identidade. Imagens, embeddings e relatórios não são carregados automaticamente.

Verificação 1:1

Auto Split ativado:

dataset_1v1/
  0001__Alice/
    gallery.jpg
    img002.jpg
    img003.jpg
  0002__Bob/
    img001.jpg
    img002.jpg

Cada subpasta representa uma identidade. Uma imagem cujo nome contém "gallery" é escolhida como gallery image dessa identidade. Se não existir, a primeira imagem ordenada é usada como gallery. As restantes imagens são probes. A avaliação compara cada probe com a gallery de cada identidade; cada trial é probe vs gallery.

Auto Split desativado:

dataset_1v1/
  0001__Alice/
    img001.jpg
    img002.jpg
  0002__Bob/
    img001.jpg
    img002.jpg

Cada imagem é tratada como probe. A avaliação executa comparações completas probe-vs-probe entre todas as pastas de identidade.

Identificação 1:N

Auto Split ativado:

dataset/
  identities/
    0001__Alice/
      gallery.jpg
      img002.jpg
      img003.jpg
    0002__Bob/
      img001.jpg
      img002.jpg

A pasta identities/ é preferida quando existe. Se não existir, a raiz selecionada pode conter diretamente pastas de identidade. A seleção de gallery segue a mesma regra de Auto Split do 1:1.

Auto Split desativado:

dataset_1n/
  gallery/
    0001__Alice/
      enroll_001.jpg
      enroll_002.jpg
    0002__Bob/
      enroll_001.jpg
  probe/
    0001__Alice/
      test_001.jpg
      test_002.jpg
    0002__Bob/
      test_001.jpg
  unknown/
    unknown_001.jpg
    unknown_002.jpg

gallery/ e probe/ são obrigatórios. unknown/ é opcional. A avaliação 1:N sempre requer imagens gallery.

Verificações de validação

- A raiz selecionada existe e corresponde ao modo selecionado.
- As pastas ou imagens gallery/probe necessárias existem.
- Auto Split consegue criar conjuntos gallery/probe válidos.
- 1:1 consegue gerar pares positivos e negativos.
- Identidades probe de 1:N têm cobertura gallery válida.
- Cada imagem a validar pode ser lida.
- A contagem de faces detectadas segue a política de múltiplas faces selecionada.

Tratamento de múltiplas faces

- Require exactly one face: imagens com múltiplas faces falham na validação.
- Use largest face: a avaliação usa a maior face detectada.
- Use largest centered face: a avaliação usa a face com maior pontuação area - center_distance^2 * 2.0.
- Mark as skip: imagens com múltiplas faces são ignoradas; imagens gallery ignoradas podem fazer a validação falhar.

Saídas do relatório

Relatórios 1:1 incluem accuracy no melhor limiar cosine, FAR/FRR, TAR@FAR, recomendações de limiar, latência e exemplos de resultados brutos.

Relatórios 1:N incluem Top1, TAR@FAR, recomendações de limiar, latência e exemplos de resultados brutos.""",
    "ru": """Страница Enterprise Evaluation поддерживает локальную оценку 1:1 и 1:N из папок идентичностей. Изображения, эмбеддинги и отчеты не загружаются автоматически.

Верификация 1:1

Auto Split включен:

dataset_1v1/
  0001__Alice/
    gallery.jpg
    img002.jpg
    img003.jpg
  0002__Bob/
    img001.jpg
    img002.jpg

Каждая подпапка представляет одну идентичность. Файл, имя которого содержит "gallery", выбирается как gallery image для этой идентичности. Если такого файла нет, первым gallery становится первое изображение в отсортированном списке. Все остальные изображения являются probes. Оценка сравнивает каждый probe с gallery каждой идентичности, поэтому каждый trial — это probe vs gallery.

Auto Split отключен:

dataset_1v1/
  0001__Alice/
    img001.jpg
    img002.jpg
  0002__Bob/
    img001.jpg
    img002.jpg

Каждое изображение считается probe. Оценка выполняет полные попарные сравнения probe-vs-probe по всем папкам идентичностей.

Идентификация 1:N

Auto Split включен:

dataset/
  identities/
    0001__Alice/
      gallery.jpg
      img002.jpg
      img003.jpg
    0002__Bob/
      img001.jpg
      img002.jpg

Папка identities/ используется с приоритетом, если она существует. Если ее нет, выбранный корень датасета может напрямую содержать папки идентичностей. Выбор gallery следует тому же правилу Auto Split, что и в 1:1.

Auto Split отключен:

dataset_1n/
  gallery/
    0001__Alice/
      enroll_001.jpg
      enroll_002.jpg
    0002__Bob/
      enroll_001.jpg
  probe/
    0001__Alice/
      test_001.jpg
      test_002.jpg
    0002__Bob/
      test_001.jpg
  unknown/
    unknown_001.jpg
    unknown_002.jpg

gallery/ и probe/ обязательны. unknown/ является опциональной. Оценка 1:N всегда требует gallery-изображения.

Проверки валидации

- Выбранный корень существует и соответствует выбранному режиму.
- Необходимые папки или изображения gallery/probe существуют.
- Auto Split может создать валидные наборы gallery/probe.
- 1:1 может создать положительные и отрицательные пары.
- Идентичности probe в 1:N имеют валидное покрытие gallery.
- Каждое проверяемое изображение можно прочитать.
- Число обнаруженных лиц соответствует выбранной политике обработки нескольких лиц.

Обработка нескольких лиц

- Require exactly one face: изображения с несколькими лицами не проходят валидацию.
- Use largest face: оценка использует самое крупное обнаруженное лицо.
- Use largest centered face: оценка использует лицо с лучшим счетом area - center_distance^2 * 2.0.
- Mark as skip: изображения с несколькими лицами пропускаются; пропуск gallery-изображения может привести к ошибке валидации.

Вывод отчетов

Отчеты 1:1 включают accuracy при лучшем cosine-пороге, FAR/FRR, TAR@FAR, рекомендации по порогу, задержку и примеры сырых результатов.

Отчеты 1:N включают Top1, TAR@FAR, рекомендации по порогу, задержку и примеры сырых результатов.""",
}


def dataset_rules_text(language: str | None) -> str:
    lang = effective_language(language)
    text = DATASET_RULES_LOCAL_TEXT.get(lang)
    if text:
        return f"{tr('Evaluation Dataset Rules', lang)}\n\n{text}"
    return DATASET_RULES_TEXT


class DatasetRulesDialog(QDialog):
    def __init__(self, language: str | None = None, parent=None):
        if parent is None and isinstance(language, QWidget):
            parent = language
            language = getattr(getattr(parent, "context", None), "config", None)
            language = getattr(language, "ui_language", None)
        super().__init__(parent)
        self.setWindowTitle("Evaluation Dataset Rules")
        self.resize(820, 680)
        layout = QVBoxLayout(self)
        intro = QLabel("Prepare local identity folders using one of the supported layouts below.")
        intro.setWordWrap(True)
        intro.setProperty("role", "muted")
        layout.addWidget(intro)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(dataset_rules_text(language))
        layout.addWidget(text, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        apply_translations(self, language)


class EnterpriseEvalPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(
            context,
            "Enterprise Evaluation",
            "Run local 1:1 verification or 1:N identification evaluation from identity folders and export procurement-ready reports.",
            parent,
        )
        self.eval_mode = QComboBox()
        self.eval_mode.setProperty("i18nItems", True)
        self.eval_mode.addItem("1:1 Verification", "1:1 Verification")
        self.eval_mode.addItem("1:N Identification", "1:N Identification")
        self.eval_mode.currentIndexChanged.connect(self._update_instructions)
        self.auto_split = QCheckBox("Auto Split")
        self.auto_split.setChecked(True)
        self.auto_split.setToolTip(
            "Automatically select each identity's gallery image from a file containing 'gallery', or the first sorted image."
        )
        self.auto_split.stateChanged.connect(self._update_instructions)
        self.multi_face_policy = QComboBox()
        self.multi_face_policy.setProperty("i18nItems", True)
        self.multi_face_policy.addItem("Require exactly one face", MULTI_FACE_REQUIRE_ONE)
        self.multi_face_policy.addItem("Use largest face", MULTI_FACE_USE_LARGEST)
        self.multi_face_policy.addItem("Use largest centered face", MULTI_FACE_USE_CENTERED_LARGEST)
        self.multi_face_policy.addItem("Mark as skip", MULTI_FACE_SKIP)
        self.multi_face_policy.setToolTip(
            "Choose how evaluation handles images where more than one face is detected."
        )
        self.multi_face_policy.currentIndexChanged.connect(self._update_instructions)
        self.dataset_root = DropInput("Identity Folders / Dataset Root", mode="folder")
        self.dataset_root.pathsChanged.connect(self._on_dataset_paths_changed)
        self.mode_summary = QLabel()
        self.mode_summary.setWordWrap(True)
        self.mode_summary.setProperty("role", "muted")
        self.validation_status = QLabel("Validate the dataset before running an evaluation.")
        self.validation_status.setWordWrap(True)
        self.validation_status.setProperty("role", "muted")

        workspace = QWidget()
        workspace_layout = QGridLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setHorizontalSpacing(14)
        workspace_layout.setVerticalSpacing(12)

        setup_card, setup_layout = self._card("Evaluation Setup")
        setup_layout.addWidget(self._field_row("Evaluation mode", self.eval_mode))
        setup_layout.addWidget(self._field_row("Data split", self.auto_split))
        setup_layout.addWidget(self._field_row("Multi-face handling", self.multi_face_policy))
        self.help_button = self.button("Dataset Rules", self.show_dataset_rules)
        self.help_button.setToolTip("Open the detailed identity folder rules for 1:1, 1:N, and Auto Split.")
        setup_layout.addWidget(self.help_button, alignment=Qt.AlignLeft)
        setup_layout.addStretch(1)

        data_card, data_layout = self._card("Dataset")
        data_card.setAcceptDrops(True)
        data_card.installEventFilter(self)
        self.dataset_drop_card = data_card
        data_layout.addWidget(self.dataset_root)
        data_layout.addWidget(self.mode_summary)
        data_layout.addStretch(1)

        actions_card, actions_layout = self._card("Run")
        actions_hint = QLabel("Validate the selected dataset and configuration before running evaluation.")
        actions_hint.setWordWrap(True)
        actions_hint.setProperty("role", "muted")
        actions_layout.addWidget(actions_hint)
        actions_layout.addWidget(self.validation_status)
        button_row = QHBoxLayout()
        self.validate_button = self.button("Validate Dataset", self.validate_dataset)
        self.run_button = self.button("Run Evaluation", self.run, enabled=False)
        self.open_reports_button = self.button("Open Report Folder", self.open_report_folder)
        button_row.addWidget(self.validate_button)
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.open_reports_button)
        button_row.addStretch(1)
        actions_layout.addLayout(button_row)
        actions_layout.addStretch(1)

        workspace_layout.addWidget(setup_card, 0, 0)
        workspace_layout.addWidget(data_card, 0, 1)
        workspace_layout.addWidget(actions_card, 1, 0, 1, 2)
        workspace_layout.setColumnStretch(0, 0)
        workspace_layout.setColumnStretch(1, 1)
        self.content.addWidget(workspace)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(260)
        self.content.addWidget(self.output, 1)
        self.last_result = None
        self.last_report_paths: dict[str, str] = {}
        self.validation_result = None
        self._validated_signature: tuple[str, bool, str, str] | None = None
        self._update_instructions()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        dataset_targets = {
            getattr(self, "dataset_drop_card", None),
        }
        if watched in dataset_targets:
            if event.type() in (QEvent.DragEnter, QEvent.DragMove):
                if self._dataset_drop_path(event):
                    self._set_dataset_drag_active(True)
                    event.acceptProposedAction()
                    return True
                event.ignore()
                return True
            if event.type() == QEvent.DragLeave:
                self._set_dataset_drag_active(False)
                return False
            if event.type() == QEvent.Drop:
                path = self._dataset_drop_path(event)
                self._set_dataset_drag_active(False)
                if path:
                    self.dataset_root.set_path(path)
                    self.set_status(f"Dataset root selected: {path}")
                    event.acceptProposedAction()
                    return True
                event.ignore()
                return True
        return super().eventFilter(watched, event)

    def _dataset_drop_path(self, event) -> str:
        mime = event.mimeData()
        if not mime.hasUrls():
            return ""
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile()).expanduser()
            if path.is_dir():
                return str(path)
        return ""

    def _set_dataset_drag_active(self, active: bool) -> None:
        if hasattr(self.dataset_root, "_set_property"):
            self.dataset_root._set_property("dragActive", bool(active))

    def _on_dataset_paths_changed(self, paths=None) -> None:
        del paths
        self._invalidate_validation()

    def _update_instructions(self) -> None:
        language = self.context.config.ui_language
        mode = self._eval_mode()
        auto_split = self.auto_split.isChecked()
        policy = tr(self.multi_face_policy.currentText(), language)
        if mode.startswith("1:1"):
            split_text = (
                "Auto Split selects one gallery image per identity and compares every probe against every identity gallery."
                if auto_split
                else "Auto Split is off, so every image is treated as a probe and full pairwise probe-vs-probe comparisons are generated."
            )
            text = f"1:1 verification expects one identity per subfolder. {split_text}"
        elif auto_split:
            text = (
                "1:N Auto Split can use identities/<identity folders> or identity folders directly. "
                "Each identity must provide a gallery image and at least one probe."
            )
        else:
            text = (
                "1:N structured evaluation requires gallery/<identity> and probe/<identity>. "
                "unknown/ is optional and contributes impostor scores for FAR thresholds."
            )
        self.mode_summary.setText(
            f"{tr(text, language)} {tr('Multi-face handling', language)}: {policy}. "
            f"{tr('Open Dataset Rules for detailed folder examples.', language)}"
        )
        self._invalidate_validation()

    def _card(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("enterpriseCard")
        frame.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)
        layout.setSpacing(10)
        heading = QLabel(title)
        heading.setObjectName("enterpriseCardTitle")
        heading.setStyleSheet("font-size: 15px; font-weight: 700;")
        layout.addWidget(heading)
        return frame, layout

    def _field_row(self, label: str, widget) -> QWidget:
        row = QWidget()
        layout = QGridLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(10)
        caption = QLabel(label)
        caption.setMinimumWidth(150)
        caption.setProperty("role", "secondary")
        layout.addWidget(caption, 0, 0, Qt.AlignTop)
        layout.addWidget(widget, 0, 1)
        layout.setColumnStretch(1, 1)
        return row

    def show_dataset_rules(self) -> None:
        dialog = DatasetRulesDialog(self.context.config.ui_language, self)
        dialog.exec()

    def _current_signature(self) -> tuple[str, bool, str, str]:
        return (
            self._eval_mode(),
            self.auto_split.isChecked(),
            self._multi_face_policy(),
            self.dataset_root.path().strip(),
        )

    def _eval_mode(self) -> str:
        return str(self.eval_mode.currentData() or "1:1 Verification")

    def _multi_face_policy(self) -> str:
        return str(self.multi_face_policy.currentData() or MULTI_FACE_REQUIRE_ONE)

    def _invalidate_validation(self, *args) -> None:
        del args
        self._validated_signature = None
        if hasattr(self, "run_button"):
            self.run_button.setEnabled(False)
        if hasattr(self, "validation_status"):
            self.validation_status.setText(
                tr("Dataset validation is required before running an evaluation.", self.context.config.ui_language)
            )

    def validate_dataset(self) -> None:
        if not self.context.engine.is_loaded():
            self.show_error("Model is not loaded. Please open Models.")
            return
        root = self.dataset_root.path().strip()
        if not root:
            self.show_error("Please select identity folders or a dataset root.")
            return
        signature = self._current_signature()
        mode = self._eval_mode()
        auto_split = self.auto_split.isChecked()
        multi_face_policy = self._multi_face_policy()

        def task(progress=None, is_cancelled=None):
            return validate_enterprise_dataset(
                root,
                mode,
                auto_split,
                self.context.engine,
                multi_face_policy=multi_face_policy,
                progress_callback=progress,
                cancel_callback=is_cancelled,
            )

        def done(result):
            self.validation_result = result
            self.output.setPlainText(self._validation_text(result))
            if result.ok:
                self._validated_signature = signature
                self.run_button.setEnabled(True)
                self.validation_status.setText(
                    tr("Dataset validation passed. Evaluation is ready to run.", self.context.config.ui_language)
                )
                self.set_status("Dataset validation passed.")
            else:
                self._validated_signature = None
                self.run_button.setEnabled(False)
                self.validation_status.setText(
                    tr(
                        "Dataset validation failed. Fix the listed issues before running evaluation.",
                        self.context.config.ui_language,
                    )
                )
                self.set_status("Dataset validation failed.")

        self.run_task("Dataset validation", task, done)

    def run(self) -> None:
        if not self.context.engine.is_loaded():
            self.show_error("Model is not loaded. Please open Models.")
            return
        root = self.dataset_root.path().strip()
        if not root:
            self.show_error("Please select identity folders or a dataset root.")
            return
        if self._validated_signature != self._current_signature():
            self.show_error("Please validate this dataset and configuration before running evaluation.")
            return
        mode = self._eval_mode()
        auto_split = self.auto_split.isChecked()
        multi_face_policy = self._multi_face_policy()
        license_status = current_model_license_display(self.context).status_text

        def task(progress=None, is_cancelled=None):
            if mode.startswith("1:1"):
                return run_identity_verification_evaluation(
                    root,
                    self.context.engine,
                    auto_split=auto_split,
                    multi_face_policy=multi_face_policy,
                    license_status=license_status,
                    progress_callback=progress,
                    cancel_callback=is_cancelled,
                )
            return run_identity_identification_evaluation(
                root,
                self.context.engine,
                auto_split=auto_split,
                multi_face_policy=multi_face_policy,
                license_status=license_status,
                progress_callback=progress,
                cancel_callback=is_cancelled,
            )

        def done(result):
            language = self.context.config.ui_language
            self.last_result = result
            self.last_report_paths = write_reports(result, self.context.config.report_dir, language=language)
            report_path = self.last_report_paths.get("pdf") or self.last_report_paths.get("markdown", "")
            self.context.storage.save_evaluation_run(
                result.scenario,
                result.model_name,
                result.provider,
                result.threshold,
                result.dataset_summary,
                result.metrics,
                result.latency,
                report_path,
                created_at=result.created_at,
            )
            self.output.setPlainText(self._summary_text(result, self.last_report_paths))
            self.set_status(f"{tr('Evaluation complete. Report saved to:', language)} {report_path}")
            QMessageBox.information(
                self,
                tr("Evaluation report saved", language),
                f"{tr('Evaluation report saved to:', language)}\n{report_path}",
            )

        self.run_task("Enterprise evaluation", task, done)

    def _validation_text(self, result) -> str:
        lines = [
            "Dataset Validation",
            f"Status: {'PASSED' if result.ok else 'FAILED'}",
            f"Mode: {result.mode}",
            f"Auto Split: {result.auto_split}",
            f"Multi-face policy: {result.multi_face_policy}",
            f"Root: {result.root}",
            "",
            "Summary:",
        ]
        for key, value in result.summary.items():
            lines.append(f"{key}: {value}")
        if result.errors:
            lines.extend(["", f"Errors: {len(result.errors)}"])
            for row in result.errors[:50]:
                lines.append(str(row))
            if len(result.errors) > 50:
                lines.append(f"... {len(result.errors) - 50} more errors")
        if result.warnings:
            lines.extend(["", f"Warnings: {len(result.warnings)}"])
            for row in result.warnings[:50]:
                lines.append(str(row))
            if len(result.warnings) > 50:
                lines.append(f"... {len(result.warnings) - 50} more warnings")
        return "\n".join(lines)

    def _summary_text(self, result, paths: dict[str, str]) -> str:
        language = self.context.config.ui_language
        lines = [
            f"{tr('Scenario', language)}: {result.scenario}",
            f"{tr('Dataset', language)}: {result.dataset_summary}",
            f"{tr('Report PDF', language)}: {paths.get('pdf', tr('not exported', language))}",
            f"{tr('Report Markdown', language)}: {paths.get('markdown', tr('not exported', language))}",
            f"{tr('Report HTML', language)}: {paths.get('html', tr('not exported', language))}",
            "",
            f"{tr('Metrics', language)}:",
        ]
        for key, value in result.metrics.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.6f}")
            else:
                lines.append(f"{key}: {value}")
        if result.errors:
            lines.extend(["", f"{tr('Errors', language)}: {len(result.errors)}", f"{tr('First errors', language)}:"])
            for row in result.errors[:10]:
                lines.append(str(row))
        return "\n".join(lines)

    def open_report_folder(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.context.config.report_dir))))
