import pytest


def test_insightface_links_add_gui_referrer():
    pytest.importorskip("PySide6")

    from insightface.gui.core.links import add_gui_referrer

    url = add_gui_referrer("https://www.insightface.ai/contact", content="license_enterprise_support")

    assert url.startswith("https://www.insightface.ai/contact?")
    assert "utm_source=insightface_gui" in url
    assert "utm_medium=" not in url
    assert "utm_campaign=" not in url
    assert "utm_content=" not in url
    assert "ref=" not in url


def test_non_insightface_links_are_not_modified():
    pytest.importorskip("PySide6")

    from insightface.gui.core.links import add_gui_referrer

    assert add_gui_referrer("https://github.com/deepinsight/insightface") == "https://github.com/deepinsight/insightface"
