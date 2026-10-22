from insightface.gui.core.theme import (
    THEME_OPTIONS,
    application_stylesheet,
    effective_theme,
    normalize_theme,
    theme_description,
    theme_label,
)


def test_theme_options_include_multiple_product_themes():
    values = [option.value for option in THEME_OPTIONS]

    assert values[0] == "system"
    assert len(values) == 7
    assert len(set(values)) == len(values)
    assert {
        "precision_light",
        "studio_dark",
        "graphite_pro",
        "azure_lab",
        "emerald_focus",
        "crimson_audit",
    }.issubset(values)


def test_theme_aliases_and_descriptions_are_backward_compatible():
    assert normalize_theme("light") == "precision_light"
    assert normalize_theme("dark") == "studio_dark"
    assert normalize_theme("unknown") == "system"
    assert theme_label("dark") == "Studio Dark"
    assert theme_description("azure_lab")


def test_application_stylesheet_contains_theme_specific_controls():
    for value in ["precision_light", "studio_dark", "graphite_pro", "azure_lab", "emerald_focus", "crimson_audit"]:
        stylesheet = application_stylesheet(value)
        assert "QFrame#uploadPreview" in stylesheet
        assert "QGraphicsView#imageViewer" in stylesheet
        assert "QWidget#imageViewerViewport" in stylesheet
        assert "QPushButton#removeUpload" in stylesheet
        assert "QWidget#dashboardCard" in stylesheet
        assert "QLabel[role=\"statusChip\"]" in stylesheet
        assert effective_theme(value) == value


def test_product_themes_have_distinct_accent_colors():
    stylesheets = {
        value: application_stylesheet(value)
        for value in ["precision_light", "studio_dark", "graphite_pro", "azure_lab", "emerald_focus", "crimson_audit"]
    }

    assert "#2f3a45" in stylesheets["precision_light"]
    assert "#7c5cff" in stylesheets["studio_dark"]
    assert "#f59e0b" in stylesheets["graphite_pro"]
    assert "#075eea" in stylesheets["azure_lab"]
    assert "#07844f" in stylesheets["emerald_focus"]
    assert "#ff3b6b" in stylesheets["crimson_audit"]
