import os
from string import Formatter

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from insightface.gui.core.i18n import (
    LANGUAGE_OPTIONS,
    effective_language,
    normalize_language,
    tr,
)


def test_supported_languages_match_homepage_language_set():
    values = {option.value for option in LANGUAGE_OPTIONS}

    assert {
        "system",
        "en",
        "zh",
        "ja",
        "ko",
        "es",
        "fr",
        "de",
        "pt",
        "ru",
    }.issubset(values)


def test_language_normalization_and_fallback():
    assert normalize_language("zh_CN") == "zh"
    assert normalize_language("pt-BR") == "pt"
    assert normalize_language("auto") == "system"
    assert normalize_language("xx") == "en"
    assert effective_language("en") == "en"


def test_core_business_translations_have_professional_terms():
    assert tr("Settings", "zh") == "设置"
    assert tr("Enterprise Evaluation", "ja") == "エンタープライズ評価"
    assert tr("Contact Enterprise Support", "ko") == "기업 지원 문의"
    assert tr("Commercial production", "de") == "Kommerzieller Produktivbetrieb"
    assert tr("requires commercial model license", "fr") == "requiert une licence commerciale du modèle"
    assert tr("Face Recognition", "es") == "Reconocimiento facial"
    assert tr("Run Evaluation", "pt") == "Executar avaliação"
    assert tr("License Center", "ru") == "Центр лицензий"


def test_person_analysis_chinese_copy_preserves_format_fields():
    from insightface.gui.core.i18n import _PERSON_ANALYSIS_ZH_TRANSLATIONS
    required = {"Person Analysis", "Start analysis", "Get commercial license", "Unmatched",
                "Automatically add body references", "Invalid person configuration",
                "Could not load person settings:\n{error}",
                "Advanced parameters", "Analyze at most (times/second)",
                "Could not save person settings:\n{error}",
                "Changes apply the next time you start analysis.",
                "Analyzed frames: {frame} · Time: {time} · Detected: {count} · Matched: {known}"}
    assert required.issubset(_PERSON_ANALYSIS_ZH_TRANSLATIONS)
    assert tr("Unmatched", "zh") == "未匹配"
    for source, translated in _PERSON_ANALYSIS_ZH_TRANSLATIONS.items():
        assert translated and translated != source
        assert tr(source, "zh") == translated
        fields = lambda text: {(name, spec, conversion) for _part, name, spec, conversion in Formatter().parse(text) if name}
        assert fields(source) == fields(translated), source


def test_person_analysis_page_can_switch_chinese_and_english(tmp_path):
    pytest.importorskip("PySide6")
    from types import SimpleNamespace
    from PySide6.QtWidgets import QApplication
    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.i18n import apply_translations
    from insightface.gui.pages.person_analysis_page import PersonAnalysisPage

    configure_qt_plugin_paths()
    app = QApplication.instance() or QApplication([])
    cfg = AppConfig(workspace_path=str(tmp_path), model_root=str(tmp_path / "models"), ui_language="zh")
    page = PersonAnalysisPage(SimpleNamespace(config=cfg))
    apply_translations(page, "zh")
    assert page.start_button.text() == "开始分析"
    assert page.commercial_button.text() == "获取商业授权"
    assert page.auto_update.text() == "自动补充人体参考样本"
    assert page.advanced_button.text() == "高级参数"
    assert page.analysis_max_fps.specialValueText() == "自动"
    assert "适用于视频和摄像头" in page.analysis_rate_hint.text()
    assert page.input_kind.itemText(0) == "本地视频"
    assert "按顺序" in page.input_hint.text()
    assert "按视频时间采样" in page.input_hint.text()
    assert "快于或慢于" in page.input_hint.text()
    page.input_kind.setCurrentIndex(page.input_kind.findData("rtsp"))
    assert "最新帧" in page.input_hint.text() and "跳过" in page.input_hint.text()
    cfg.ui_language = "en"
    apply_translations(page, "en")
    assert page.start_button.text() == "Start analysis"
    assert page.commercial_button.text() == "Get commercial license"
    assert page.auto_update.text() == "Automatically add body references"
    assert page.advanced_button.text() == "Advanced parameters"
    assert page.analysis_max_fps.specialValueText() == "Auto"
    assert page.input_kind.itemText(0) == "Local video"
    assert "latest frame" in page.input_hint.text() and "skips older" in page.input_hint.text()
    assert "Applies to videos and cameras" in page.analysis_rate_hint.text()
    page.input_kind.setCurrentIndex(page.input_kind.findData("video"))
    assert "every video frame" in page.input_hint.text()
    assert "samples by video time" in page.input_hint.text()
    assert "faster or slower than playback" in page.input_hint.text()
    app.processEvents()
    page.close()


def test_all_person_parameter_labels_and_help_switch_languages(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication
    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.i18n import apply_translations, _PERSON_ANALYSIS_ZH_TRANSLATIONS
    from insightface.gui.pages.person_analysis_page import PERSON_PARAMETER_COPY, PersonParametersDialog

    configure_qt_plugin_paths()
    app = QApplication.instance() or QApplication([])
    saved = []
    dialog = PersonParametersDialog({}, "zh", saved.append)
    assert dialog.windowTitle() == "高级参数"
    assert dialog.restore_button.text() == "恢复默认"
    assert dialog.save_button.text() == "保存" and dialog.cancel_button.text() == "取消"
    for name in ("face_det_size", "body_det_size"):
        assert dialog.controls[name].value() == 0
        assert dialog.controls[name].specialValueText() == ("默认（640）" if name == "face_det_size" else "模型默认")
    for name, (label, help_text) in PERSON_PARAMETER_COPY.items():
        assert label in _PERSON_ANALYSIS_ZH_TRANSLATIONS
        assert help_text in _PERSON_ANALYSIS_ZH_TRANSLATIONS
        assert dialog.field_labels[name].text() == tr(label, "zh") != label
        assert dialog.help_labels[name].text() == tr(help_text, "zh") != help_text
        assert dialog.controls[name].toolTip() == tr(help_text, "zh")
    apply_translations(dialog, "en")
    assert dialog.windowTitle() == "Advanced parameters"
    assert dialog.restore_button.text() == "Restore defaults"
    for name in ("face_det_size", "body_det_size"):
        assert dialog.controls[name].specialValueText() == ("Default (640)" if name == "face_det_size" else "Model default")
    for name, (label, help_text) in PERSON_PARAMETER_COPY.items():
        assert dialog.field_labels[name].text() == label
        assert dialog.help_labels[name].text() == help_text
        assert dialog.controls[name].toolTip() == help_text
    dialog.reject()
    assert not saved
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def test_privateframe_translation_catalog_is_complete_and_format_safe():
    from insightface.gui.core.i18n import _PRIVATEFRAME_UI_TRANSLATIONS

    languages = {"zh", "ja", "ko", "es", "fr", "de", "pt", "ru"}
    assert set(_PRIVATEFRAME_UI_TRANSLATIONS) == languages
    expected_keys = set(_PRIVATEFRAME_UI_TRANSLATIONS["zh"])
    assert len(expected_keys) >= 150

    for language in languages:
        translations = _PRIVATEFRAME_UI_TRANSLATIONS[language]
        assert set(translations) == expected_keys
        for source, translated in translations.items():
            source_fields = {
                name for _text, name, _spec, _conversion in Formatter().parse(source)
                if name
            }
            translated_fields = {
                name
                for _text, name, _spec, _conversion in Formatter().parse(translated)
                if name
            }
            assert translated_fields == source_fields, (language, source)


def test_model_download_action_translation_catalog_is_complete_and_format_safe():
    from insightface.gui.core.i18n import _MODEL_DOWNLOAD_ACTION_TRANSLATIONS

    languages = {"zh", "ja", "ko", "es", "fr", "de", "pt", "ru"}
    assert set(_MODEL_DOWNLOAD_ACTION_TRANSLATIONS) == languages
    expected_keys = set(_MODEL_DOWNLOAD_ACTION_TRANSLATIONS["zh"])
    assert len(expected_keys) >= 10

    for language in languages:
        translations = _MODEL_DOWNLOAD_ACTION_TRANSLATIONS[language]
        assert set(translations) == expected_keys
        for source, translated in translations.items():
            source_fields = {
                name
                for _text, name, _spec, _conversion in Formatter().parse(source)
                if name
            }
            translated_fields = {
                name
                for _text, name, _spec, _conversion in Formatter().parse(translated)
                if name
            }
            assert translated_fields == source_fields, (language, source)


def test_privateframe_photo_workflow_translations_are_used_and_have_no_fallbacks():
    from insightface.gui.core.i18n import (
        _PRIVATEFRAME_PHOTO_UI_LANGUAGES,
        _PRIVATEFRAME_PHOTO_UI_ROWS,
        _PRIVATEFRAME_PHOTO_UI_TRANSLATIONS,
        _PRIVATEFRAME_UI_TRANSLATIONS,
    )

    expected_languages = {
        option.value for option in LANGUAGE_OPTIONS
        if option.value not in {"system", "en"}
    }
    assert set(_PRIVATEFRAME_PHOTO_UI_LANGUAGES) == expected_languages
    assert set(_PRIVATEFRAME_PHOTO_UI_TRANSLATIONS) == expected_languages
    assert len(_PRIVATEFRAME_PHOTO_UI_ROWS) >= 62
    assert (
        "Checks up to 1, 3 or 5 selected frames per track unless overridden in the configuration. "
        "More checks take longer and provide more evidence; they do not guarantee a correct match."
    ) in _PRIVATEFRAME_PHOTO_UI_ROWS
    assert {"Processing details…", "PrivateFrame Processing Details"} <= set(
        _PRIVATEFRAME_PHOTO_UI_ROWS
    )
    for source, values in _PRIVATEFRAME_PHOTO_UI_ROWS.items():
        assert len(values) == len(expected_languages), source
        source_fields = sorted(
            (name, spec, conversion)
            for _text, name, spec, conversion in Formatter().parse(source)
            if name
        )
        for language in expected_languages:
            translated = _PRIVATEFRAME_PHOTO_UI_TRANSLATIONS[language][source]
            assert translated.strip() and translated != source, (language, source)
            assert tr(source, language) == translated
            assert _PRIVATEFRAME_UI_TRANSLATIONS[language][source] == translated
            assert sorted(
                (name, spec, conversion)
                for _text, name, spec, conversion in Formatter().parse(translated)
                if name
            ) == source_fields, (language, source)


def test_apply_translations_updates_basic_widgets():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.i18n import apply_translations

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    widget = QWidget()
    layout = QVBoxLayout(widget)
    label = QLabel("Settings")
    button = QPushButton("Run Evaluation")
    button.setToolTip("Choose how evaluation handles images where more than one face is detected.")
    layout.addWidget(label)
    layout.addWidget(button)

    apply_translations(widget, "zh")

    assert label.text() == "设置"
    assert button.text() == "运行评测"
    assert button.toolTip() == "选择评测中检测到多张人脸时的处理方式。"

    apply_translations(widget, "en")

    assert label.text() == "Settings"
    assert button.text() == "Run Evaluation"
    assert button.toolTip() == "Choose how evaluation handles images where more than one face is detected."


def test_apply_translations_localizes_generic_button_tooltip():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QPushButton

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.i18n import apply_translations

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    button = QPushButton("Run Face Swap")
    button.setToolTip("Run face swap with the configured local swap model.")

    apply_translations(button, "zh")

    assert button.text() == "运行换脸"
    assert button.toolTip() == "点击执行：运行换脸。"

    apply_translations(button, "en")

    assert button.text() == "Run Face Swap"
    assert button.toolTip() == "Run face swap with the configured local swap model."


def test_icon_only_button_keeps_symbol_when_tooltip_is_localized():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QPushButton

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.i18n import apply_translations

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    button = QPushButton("×")
    button.setToolTip("Remove the current file.")

    apply_translations(button, "zh")

    assert button.text() == "×"
    assert button.toolTip() == "点击执行：移除。"


def test_dataset_rules_dialog_has_localized_help_summary():
    from insightface.gui.pages.enterprise_eval_page import dataset_rules_text

    text = dataset_rules_text("zh")

    assert text.startswith("评测数据集规则")
    assert "支持从身份文件夹进行本地 1:1 验证和 1:N 识别评测" in text
    assert "dataset_1v1/" in text
    assert "dataset_1n/" in text
    assert "gallery/" in text
    assert "probe/" in text
    assert "多人脸处理" in text
    assert "报告输出" in text
    assert "Enterprise Evaluation Dataset Rules" not in text
    assert "Each subfolder is one identity" not in text


def test_album_and_recognition_page_copy_is_localized():
    assert (
        tr(
            "Import or refresh local album folders, cluster detected faces, and review the photos in each person group.",
            "zh",
        )
        == "导入或刷新本地相册文件夹，对检测到的人脸进行聚类，并查看每个人物组中的照片。"
    )
    assert "所有相册处理均在本地完成" in tr(
        "All album processing is local. Import / Refresh scans every selected folder, extracts features only for new images, then runs DBSCAN clustering over all indexed faces using the selected cosine threshold.",
        "zh",
    )
    assert "上传一张查询图片" in tr(
        "Upload one query image and a gallery image, image set, or folder. One gallery image runs 1:1 compare; multiple gallery images run 1:N gallery search.",
        "zh",
    )
    assert "源图 + 目标 = 结果" in tr(
        "Source + Target = Result. Target can be an image or a video; the workflow chooses image or video swap automatically.",
        "zh",
    )
    assert "商业授权" in tr(
        "Face swap may require separate commercial authorization depending on usage and model license. Use only with appropriate rights and consent.",
        "zh",
    )
    assert "采购决策" in tr(
        "Run local 1:1 verification or 1:N identification evaluation from identity folders and export procurement-ready reports.",
        "zh",
    )
    assert "运行后端" in tr(
        "Configure model packs, execution provider, face swap models, and runtime checks.",
        "zh",
    )
    assert "手动刷新 GitHub Release" in tr(
        "Manually refresh GitHub release model URLs and download selected model packages locally.",
        "zh",
    )
    assert "商业部署" in tr(
        "Code and model files may have different licenses. Commercial deployment requires appropriate model authorization.",
        "zh",
    )
    assert tr("Refresh source", "zh") == "刷新来源"
    assert tr("Local model root", "zh") == "本地模型根目录"
