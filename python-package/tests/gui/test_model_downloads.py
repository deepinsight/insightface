import json
import os
import urllib.error
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from insightface.gui.core import model_downloads
from insightface.gui.core.model_downloads import (
    GFPGAN_DOWNLOAD_URL,
    GITHUB_MODEL_ZOO_DOWNLOAD_URL,
    GITHUB_MODEL_ZOO_RELEASE_API,
    _content_range_total,
    fallback_model_assets,
    load_cached_assets,
    is_model_asset_installed,
    local_model_status,
    refresh_model_assets,
)
from insightface.gui.core.paths import default_workspace, workspace_paths


def test_default_gui_workspace_path():
    workspace = default_workspace()
    assert workspace.name == "gui"
    assert workspace.parent.name == ".insightface"
    assert workspace.is_absolute()


def test_workspace_paths_are_under_gui_workspace(tmp_path):
    workspace = (tmp_path / ".insightface" / "gui").resolve()
    paths = workspace_paths(workspace)
    assert paths["workspace"] == workspace
    for key in ("database", "crops", "exports", "reports", "logs", "cache"):
        assert paths[key].is_relative_to(paths["workspace"])


def test_fallback_model_assets_have_github_release_urls(tmp_path):
    assets = fallback_model_assets()
    names = {asset.name for asset in assets}
    assert {
        "buffalo_l.zip",
        "buffalo_s.zip",
        "antelopev2.zip",
        "raccoon_l.zip",
        "raccoon_s.zip",
        "cheetah_s.zip",
        "cheetah_l.zip",
        "GFPGANv1.4.onnx",
    }.issubset(names)
    for asset in assets:
        if asset.name == "GFPGANv1.4.onnx":
            assert asset.source == "third party"
            assert asset.kind == "third-party restore model"
            assert asset.browser_download_url == GFPGAN_DOWNLOAD_URL
            assert "harisreedhar/Face-Upscalers-ONNX/releases/download/Models" in asset.browser_download_url
        else:
            assert asset.tag_name == "model-zoo"
            assert asset.browser_download_url == (
                f"{GITHUB_MODEL_ZOO_DOWNLOAD_URL}{asset.name}"
            )
    assert local_model_status(assets[0], tmp_path) == "not installed"


def test_model_asset_install_state_distinguishes_complete_and_partial_packages(
    tmp_path,
):
    package = next(
        asset for asset in fallback_model_assets() if asset.name == "raccoon_s.zip"
    )
    package_dir = tmp_path / "models" / "raccoon_s"

    assert not is_model_asset_installed(package, tmp_path)
    package_dir.mkdir(parents=True)
    assert not is_model_asset_installed(package, tmp_path)
    assert local_model_status(package, tmp_path) == f"folder exists: {package_dir}"

    (package_dir / "detector.onnx").write_bytes(b"onnx")
    assert is_model_asset_installed(package, tmp_path)
    assert local_model_status(package, tmp_path) == f"installed: {package_dir}"


def test_zip_model_install_replaces_existing_package_only_after_full_extract(
    tmp_path,
    monkeypatch,
):
    source_archive = tmp_path / "source.zip"
    with model_downloads.zipfile.ZipFile(source_archive, "w") as archive:
        archive.writestr("detector.onnx", b"new detector")
        archive.writestr("recognizer.onnx", b"new recognizer")

    def stage_archive(_url, destination, *_args, **_kwargs):
        destination.write_bytes(source_archive.read_bytes())

    monkeypatch.setattr(model_downloads, "_download_with_retries", stage_archive)
    target = tmp_path / "root" / "models" / "raccoon_s"
    target.mkdir(parents=True)
    (target / "old.onnx").write_bytes(b"old")
    asset = model_downloads.ModelAsset(
        name="raccoon_s.zip",
        browser_download_url="https://example.invalid/raccoon_s.zip",
    )

    installed = model_downloads.download_model_asset(
        asset,
        model_root=tmp_path / "root",
        gui_cache_dir=tmp_path / "cache",
    )

    assert installed == target
    assert not (target / "old.onnx").exists()
    assert (target / "detector.onnx").read_bytes() == b"new detector"
    assert (target / "recognizer.onnx").read_bytes() == b"new recognizer"
    assert not list(target.parent.glob(".raccoon_s-install-*"))
    assert not list(target.parent.glob(".raccoon_s-previous-*"))


def test_failed_zip_extract_preserves_existing_package_and_cleans_staging(
    tmp_path,
    monkeypatch,
):
    source_archive = tmp_path / "source.zip"
    with model_downloads.zipfile.ZipFile(source_archive, "w") as archive:
        archive.writestr("detector.onnx", b"new detector")

    def stage_archive(_url, destination, *_args, **_kwargs):
        destination.write_bytes(source_archive.read_bytes())

    def fail_after_partial_extract(_archive, destination, *_args, **_kwargs):
        partial = Path(destination) / "detector.onnx"
        partial.write_bytes(b"partial")
        raise RuntimeError("simulated interrupted extraction")

    monkeypatch.setattr(model_downloads, "_download_with_retries", stage_archive)
    monkeypatch.setattr(
        model_downloads.zipfile.ZipFile,
        "extractall",
        fail_after_partial_extract,
    )
    target = tmp_path / "root" / "models" / "raccoon_s"
    target.mkdir(parents=True)
    (target / "old.onnx").write_bytes(b"old")
    asset = model_downloads.ModelAsset(
        name="raccoon_s.zip",
        browser_download_url="https://example.invalid/raccoon_s.zip",
    )

    with pytest.raises(RuntimeError, match="interrupted extraction"):
        model_downloads.download_model_asset(
            asset,
            model_root=tmp_path / "root",
            gui_cache_dir=tmp_path / "cache",
        )

    assert (target / "old.onnx").read_bytes() == b"old"
    assert not list(target.parent.glob(".raccoon_s-install-*"))
    assert not list(target.parent.glob(".raccoon_s-previous-*"))


def test_content_range_total_parser():
    assert _content_range_total("bytes 10-19/100") == 100
    assert _content_range_total("bytes 10-19/*") == 0
    assert _content_range_total(None) == 0
    assert _content_range_total("not-a-range") == 0


def test_cached_official_urls_are_migrated_without_changing_third_party(tmp_path):
    cache = tmp_path / "model_download_urls.json"
    cache.write_text(
        json.dumps(
            {
                "source": (
                    "https://api.github.com/repos/deepinsight/insightface/"
                    "releases/latest"
                ),
                "assets": [
                    {
                        "name": "buffalo_l.zip",
                        "browser_download_url": (
                            "https://github.com/deepinsight/insightface/"
                            "releases/download/v0.7/buffalo_l.zip"
                        ),
                        "tag_name": "v0.7",
                        "release_name": "insightface v0.7 model packages",
                    },
                    {
                        "name": "GFPGANv1.4.onnx",
                        "browser_download_url": GFPGAN_DOWNLOAD_URL,
                        "tag_name": "third-party",
                        "source": "third party",
                    },
                    {
                        "name": "external_restore.onnx",
                        "browser_download_url": (
                            "https://models.example/external_restore.onnx"
                        ),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assets = load_cached_assets(tmp_path)
    by_name = {asset.name: asset for asset in assets}

    assert by_name["buffalo_l.zip"].browser_download_url == (
        f"{GITHUB_MODEL_ZOO_DOWNLOAD_URL}buffalo_l.zip"
    )
    assert by_name["buffalo_l.zip"].tag_name == "model-zoo"
    assert by_name["raccoon_s.zip"].browser_download_url == (
        f"{GITHUB_MODEL_ZOO_DOWNLOAD_URL}raccoon_s.zip"
    )
    assert by_name["raccoon_l.zip"].browser_download_url == (
        f"{GITHUB_MODEL_ZOO_DOWNLOAD_URL}raccoon_l.zip"
    )
    assert by_name["GFPGANv1.4.onnx"].browser_download_url == GFPGAN_DOWNLOAD_URL
    assert by_name["external_restore.onnx"].browser_download_url == (
        "https://models.example/external_restore.onnx"
    )
    assert by_name["external_restore.onnx"].source == "third party"

    migrated = json.loads(cache.read_text(encoding="utf-8"))
    assert migrated["source"] == GITHUB_MODEL_ZOO_RELEASE_API
    assert all(
        "releases/download/v0.7/" not in item["browser_download_url"]
        for item in migrated["assets"]
    )


def test_refresh_uses_model_zoo_tag_api_and_canonical_download_urls(
    tmp_path,
    monkeypatch,
):
    requested = []
    payload = {
        "tag_name": "model-zoo",
        "name": "InsightFace model zoo",
        "assets": [
            {
                "name": "raccoon_s.zip",
                # The canonical URL is derived locally instead of trusting stale
                # API/cache data.
                "browser_download_url": (
                    "https://github.com/deepinsight/insightface/"
                    "releases/download/v0.7/raccoon_s.zip"
                ),
                "size": 123,
                "content_type": "application/zip",
                "updated_at": "2026-09-03T00:00:00Z",
            },
            {
                "name": "README.txt",
                "browser_download_url": "https://example.invalid/README.txt",
            },
        ],
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    def urlopen(request, timeout):
        requested.append((request.full_url, timeout))
        return Response()

    monkeypatch.setattr(model_downloads.urllib.request, "urlopen", urlopen)

    assets, message = refresh_model_assets(tmp_path, timeout=7)

    assert requested == [(GITHUB_MODEL_ZOO_RELEASE_API, 7)]
    assert "Refreshed" in message
    official = [asset for asset in assets if asset.source == "InsightFace"]
    # Keep the new choices even before the release publishes their assets.
    assert {"raccoon_s.zip", "raccoon_l.zip", "cheetah_s.zip", "cheetah_l.zip"}.issubset(
        {asset.name for asset in official}
    )
    assert all(asset.tag_name == "model-zoo" for asset in official)
    assert all(
        asset.browser_download_url
        == f"{GITHUB_MODEL_ZOO_DOWNLOAD_URL}{asset.name}"
        for asset in official
    )


def test_refresh_failure_falls_back_to_model_zoo_urls(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(model_downloads.urllib.request, "urlopen", fail)

    assets, message = refresh_model_assets(tmp_path)

    assert "using bundled model-zoo URLs" in message
    assert all(
        asset.browser_download_url.startswith(GITHUB_MODEL_ZOO_DOWNLOAD_URL)
        for asset in assets
        if asset.source == "InsightFace"
    )


def test_model_download_table_omits_release_column(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.model_downloads import ModelAsset
    from insightface.gui.pages import model_download_page

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    asset = ModelAsset(
        name="raccoon_s.zip",
        browser_download_url=(
            f"{GITHUB_MODEL_ZOO_DOWNLOAD_URL}raccoon_s.zip"
        ),
        tag_name="model-zoo-contract-remains",
        size=1024 * 1024,
        updated_at="2026-09-03T00:00:00Z",
    )
    monkeypatch.setattr(
        model_download_page,
        "load_cached_assets",
        lambda _cache_dir: [asset],
    )
    monkeypatch.setattr(
        model_download_page,
        "local_model_status",
        lambda _asset, _model_root: "not installed",
    )

    class Context:
        pass

    context = Context()
    context.config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_root=str(tmp_path / "model-root"),
        auto_load_model=False,
    )
    page = model_download_page.ModelDownloadPage(context)

    headers = [
        page.table.horizontalHeaderItem(column).text()
        for column in range(page.table.columnCount())
    ]
    values = [
        page.table.item(0, column).text()
        for column in range(page.table.columnCount())
    ]
    assert headers == [
        "asset",
        "source",
        "kind",
        "size",
        "updated_at",
        "local status",
        "download url",
    ]
    assert values == [
        "raccoon_s.zip",
        "InsightFace",
        "model package",
        "1.0 MB",
        "2026-09-03T00:00:00Z",
        "not installed",
        f"{GITHUB_MODEL_ZOO_DOWNLOAD_URL}raccoon_s.zip",
    ]
    assert asset.tag_name == "model-zoo-contract-remains"
    assert page.table._proportional_table_sizer.estimated_widths == [
        210,
        100,
        150,
        90,
        170,
        150,
        360,
    ]
    page.table.selectRow(0)
    assert page.selected_asset() is asset
    page.close()


def test_download_page_only_sets_catalog_package_zip_as_global_model(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.model_downloads import ModelAsset
    from insightface.gui.pages import model_download_page

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    assets = [
        ModelAsset(
            name="external.zip",
            browser_download_url="https://example.invalid/external.zip",
            source="third party",
        ),
        ModelAsset(
            name="raccoon_l.zip",
            browser_download_url=(
                f"{GITHUB_MODEL_ZOO_DOWNLOAD_URL}raccoon_l.zip"
            ),
            source="InsightFace",
        ),
    ]
    monkeypatch.setattr(
        model_download_page,
        "load_cached_assets",
        lambda _cache_dir: assets,
    )
    config = AppConfig(
        workspace_path=str(tmp_path / "workspace"),
        model_root=str(tmp_path / "model-root"),
        model_name="buffalo_l",
        custom_model_dir=str(tmp_path / "stale-custom"),
        auto_load_model=False,
    )
    context = type("Context", (), {"config": config})()
    page = model_download_page.ModelDownloadPage(context)
    errors = []
    monkeypatch.setattr(page, "show_error", errors.append)

    page.table.selectRow(0)
    assert page.use_selected_button.isEnabled() is False
    assert page.download_selected_button.isEnabled()
    page.use_selected_model()
    assert config.model_name == "buffalo_l"
    assert config.custom_model_dir == str(tmp_path / "stale-custom")
    assert errors == ["Download the selected model before using it."]

    page.table.selectRow(1)
    assert not page.use_selected_button.isEnabled()
    assert page.download_selected_button.isEnabled()
    page.use_selected_model()
    assert config.model_name == "buffalo_l"

    package_dir = Path(config.model_root) / "models" / "raccoon_l"
    package_dir.mkdir(parents=True)
    (package_dir / "detector.onnx").write_bytes(b"onnx")
    page.populate()
    page.table.selectRow(1)

    assert page.use_selected_button.isEnabled()
    assert not page.download_selected_button.isEnabled()
    page.use_selected_model()
    assert config.model_name == "raccoon_l"
    assert config.custom_model_dir == ""
    monkeypatch.setattr(
        page,
        "run_task",
        lambda *_args, **_kwargs: pytest.fail(
            "an installed model must not start another download"
        ),
    )
    page.download_selected()
    assert errors[-1] == (
        "The selected model asset is already downloaded and ready to use."
    )
    page.close()


def test_download_and_use_actions_are_mutually_exclusive_for_each_asset_type(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import configure_qt_plugin_paths
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.i18n import apply_translations, tr
    from insightface.gui.core.model_downloads import ModelAsset
    from insightface.gui.pages import model_download_page

    configure_qt_plugin_paths()
    QApplication.instance() or QApplication([])
    assets = [
        ModelAsset(
            name="raccoon_s.zip",
            browser_download_url=(
                f"{GITHUB_MODEL_ZOO_DOWNLOAD_URL}raccoon_s.zip"
            ),
        ),
        ModelAsset(
            name="inswapper_128.onnx",
            browser_download_url=(
                f"{GITHUB_MODEL_ZOO_DOWNLOAD_URL}inswapper_128.onnx"
            ),
        ),
    ]
    monkeypatch.setattr(
        model_download_page,
        "load_cached_assets",
        lambda _cache_dir: assets,
    )
    model_root = tmp_path / "model-root"
    context = type(
        "Context",
        (),
        {
            "config": AppConfig(
                workspace_path=str(tmp_path / "workspace"),
                model_root=str(model_root),
                auto_load_model=False,
            )
        },
    )()
    page = model_download_page.ModelDownloadPage(context)

    page.table.selectRow(0)
    assert page.download_selected_button.isEnabled()
    assert not page.use_selected_button.isEnabled()

    package_dir = model_root / "models" / "raccoon_s"
    package_dir.mkdir(parents=True)
    (package_dir / "detector.onnx").write_bytes(b"onnx")
    onnx_dir = model_root / "models" / "inswapper_128"
    onnx_dir.mkdir(parents=True)
    (onnx_dir / "inswapper_128.onnx").write_bytes(b"onnx")
    page.populate()

    page.table.selectRow(0)
    assert not page.download_selected_button.isEnabled()
    assert page.use_selected_button.isEnabled()

    page.table.selectRow(1)
    assert not page.download_selected_button.isEnabled()
    assert page.use_selected_button.isEnabled()
    page.use_selected_model()
    assert context.config.swap_model_path == str(
        onnx_dir / "inswapper_128.onnx"
    )

    context.config.ui_language = "zh"
    apply_translations(page, "zh")
    page.table.selectRow(0)
    assert page.use_selected_button.toolTip() == tr(
        "Use this downloaded model.",
        "zh",
    )
    assert page.download_selected_button.toolTip() == tr(
        "This model asset is already downloaded.",
        "zh",
    )
    page.close()
