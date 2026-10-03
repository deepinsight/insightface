"""Inspect release archives and smoke-test installed extras without model downloads.

The installed checks launch isolated Python processes in a temporary directory,
so running this helper from the checkout cannot hide missing wheel contents.
Install the base wheel, then [privateframe], then [gui], checking each profile
before installing the next extra. No pytest or source-tree fixtures are needed.
"""

from __future__ import annotations

import argparse
import base64
from email.parser import BytesParser
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
from zipfile import ZipFile


REQUIRED_FILES = (
    "insightface/app/person_analysis.py",
    "insightface/app/person/__init__.py",
    "insightface/app/person/config.py",
    "insightface/app/person/matrix.py",
    "insightface/app/person/types.py",
    "insightface/model_zoo/person_package.py",
    "insightface/model_zoo/person_detection.py",
    "insightface/model_zoo/person_reid.py",
    "insightface/gui/core/person_analysis.py",
    "insightface/gui/pages/person_analysis_page.py",
    "insightface/data/images/t1.jpg",
    "insightface/data/objects/meanshape_68.pkl",
    "insightface/gui/assets/app_icon.png",
    "insightface/gui/assets/app_icon.ico",
    "insightface/gui/assets/app_icon.icns",
    "insightface/app/privateframe/configs/base.yaml",
    "insightface/app/privateframe/docs/configuration.md",
    "insightface_privateframe_bootstrap/__init__.py",
    "insightface_privateframe_bootstrap/__main__.py",
    "insightface/model_zoo/trusted_keys/insightface-model-license-public-ed25519.pem",
)
NATIVE_SUFFIXES = (".so", ".pyd", ".dll", ".dylib")


def _check_public_key(data: bytes) -> None:
    lines = data.strip().splitlines()
    assert lines[0] == b"-----BEGIN PUBLIC KEY-----", "Expected a public PEM key"
    assert lines[-1] == b"-----END PUBLIC KEY-----", "Invalid public PEM key"
    der = base64.b64decode(b"".join(lines[1:-1]), validate=True)
    # RFC 8410 Ed25519 SubjectPublicKeyInfo: algorithm OID, then a 32-byte key.
    assert der.startswith(bytes.fromhex("302a300506032b6570032100"))
    assert len(der) == 44, "Expected an Ed25519 public key"


def _check_resources(names: set[str], read, *, archive: bool = False) -> None:
    missing = set(REQUIRED_FILES) - names
    assert not missing, f"Missing package resources: {sorted(missing)}"
    if archive:
        for name in names:
            parts = Path(name).parts
            assert not {".private", "__pycache__", ".pytest_cache"}.intersection(parts), name
            assert not name.endswith((".onnx", ".mp4", ".sqlite", ".db")), name
    for name in names:
        if name.startswith("insightface/model_zoo/trusted_keys/") and name.endswith(".pem"):
            _check_public_key(read(name))
    assert b"schema_version: 1" in read("insightface/app/privateframe/configs/base.yaml")
    assert b"tracking.kalman_optical_flow.roi_size" in read(
        "insightface/app/privateframe/docs/configuration.md"
    )


def _check_metadata(data: bytes):
    package = BytesParser().parsebytes(data)
    assert package["Name"] == "insightface"
    assert package["Requires-Python"] == ">=3.10", package["Requires-Python"]
    extras = set(package.get_all("Provides-Extra", []))
    assert {"gui", "privateframe", "face3d"} <= extras
    requirements = {
        re.match(r"[A-Za-z0-9_.-]+", value).group()
        for value in package.get_all("Requires-Dist", [])
    }
    for name in ("onnxruntime", "opencv-python", "PySide6-Essentials", "reportlab", "scikit-learn", "matplotlib", "av", "PyYAML"):
        assert name in requirements, name
    for name in ("pandas", "easydict", "prettytable", "PySide6"):
        assert name not in requirements, name
    assert package["License"] != "MIT"
    return package


def inspect_archives(wheel: Path, sdist: Path, *, allow_native: bool = False) -> None:
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        assert len(metadata_names) == 1, metadata_names
        package = _check_metadata(archive.read(metadata_names[0]))
        dist_info = metadata_names[0].rsplit("/", 1)[0]
        if not allow_native:
            wheel_metadata = archive.read(f"{dist_info}/WHEEL").decode("utf-8")
            assert "Root-Is-Purelib: true" in wheel_metadata
            assert "Tag: py3-none-any" in wheel_metadata
            assert not any(name.endswith(NATIVE_SUFFIXES) for name in names)
        assert not any(".data/data/insightface/data/" in name for name in names)
        _check_resources(names, archive.read, archive=True)
        entry_points = archive.read(f"{dist_info}/entry_points.txt").decode("utf-8")
        assert "insightface-privateframe = insightface_privateframe_bootstrap:main" in entry_points

    with tarfile.open(sdist, "r:gz") as archive:
        prefix = f"{package['Name']}-{package['Version']}/"
        names = {name.removeprefix(prefix) for name in archive.getnames() if name.startswith(prefix)}

        def read(name: str) -> bytes:
            member = archive.extractfile(prefix + name)
            assert member is not None, name
            return member.read()

        sdist_package = _check_metadata(read("PKG-INFO"))
        assert sdist_package["Version"] == package["Version"]
        _check_resources(names, read, archive=True)
    print(f"Release archive checks passed: {wheel}, {sdist}")


def _offline_audit(event: str, args) -> None:
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise AssertionError(f"Artifact smoke must not access the network: {event}")


def _probe(profile: str) -> None:
    # Fail on accidental network access during imports/default loading.
    sys.addaudithook(_offline_audit)
    import insightface
    import insightface_privateframe_bootstrap
    from insightface.app import FaceAnalysis, PersonAnalysis
    from insightface.app.person import PersonConfig
    from insightface import model_zoo

    distribution = metadata.distribution("insightface")
    assert distribution.metadata["Requires-Python"] == ">=3.10"
    assert insightface.__version__ == distribution.version
    assert Path(insightface.__file__).resolve() == Path(
        distribution.locate_file("insightface/__init__.py")
    ).resolve(), "Smoke checks must import the installed distribution"
    assert callable(FaceAnalysis) and callable(model_zoo.get_model)
    person = PersonAnalysis(config=PersonConfig())
    try:
        assert person.name == "cheetah_l"
        assert person.providers == ["CPUExecutionProvider"]
        for name in ("prepare", "register", "get", "get_reid", "match", "update",
                     "remove_person", "clear_references"):
            assert callable(getattr(person, name)), name
    finally:
        person.close()
    assert callable(insightface_privateframe_bootstrap.main)
    root = Path(distribution.locate_file(""))
    _check_resources({str(path).replace("\\", "/") for path in distribution.files}, lambda name: (root / name).read_bytes())

    if profile != "gui":
        from importlib.util import find_spec

        assert find_spec("PySide6") is None, "GUI dependencies leaked into a smaller extra"
        if profile == "base":
            assert find_spec("av") is None, "PyAV leaked into the base installation"
    if profile in {"privateframe", "gui"}:
        import av
        import yaml
        from insightface.app.privateframe.base_config import DEFAULT_CONFIG_PATH, read_default_config
        from insightface.app.privateframe.config import load_config
        from insightface.app.privateframe import analyze_streaming_pipeline

        assert av.__version__ and yaml.__version__
        assert callable(analyze_streaming_pipeline)
        assert read_default_config()["schema_version"] == 1
        assert load_config(DEFAULT_CONFIG_PATH, materialize_models=False)["schema_version"] == 1
    if profile == "gui":
        from insightface.gui.app import configure_qt_plugin_paths
        from insightface.model_zoo.model_license import _trusted_keys
        from PySide6.QtWidgets import QApplication

        assert _trusted_keys(), "The installed trusted keys must load successfully"
        configure_qt_plugin_paths()
        app = QApplication.instance() or QApplication([])
        assert app.platformName() == "offscreen"
        app.quit()
    print(f"Installed {profile} imports/resources passed: {insightface.__file__}")


def check_installed(profile: str) -> None:
    helper = Path(__file__).resolve()
    scripts = Path(sysconfig.get_path("scripts"))
    extension = ".exe" if sys.platform == "win32" else ""

    with tempfile.TemporaryDirectory(prefix="insightface-artifact-smoke-") as temporary:
        workdir = Path(temporary)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update(XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary,
                   MPLCONFIGDIR=temporary, PYTHONDONTWRITEBYTECODE="1",
                   ORT_DISABLE_TELEMETRY="1", QT_QPA_PLATFORM="offscreen")

        def run(command: list[str]) -> str:
            completed = subprocess.run(command, cwd=workdir, env=env, text=True,
                                       capture_output=True, timeout=90)
            assert completed.returncode == 0, (
                f"Command failed: {command}\n{completed.stdout}\n{completed.stderr}"
            )
            return completed.stdout

        def console(name: str, *args: str) -> str:
            return run([str(scripts / (name + extension)), *args])

        print(run([sys.executable, "-I", "-B", str(helper), "_probe", "--profile", profile]).strip())
        version = metadata.version("insightface")
        assert "insightface-cli" in console("insightface-cli", "--help")
        assert console("insightface-privateframe", "--version").strip() == f"insightface-privateframe {version}"
        if profile in {"privateframe", "gui"}:
            description = json.loads(console("insightface-privateframe", "describe"))
            assert description["ok"] is True and description["command"] == "describe"
            reference = Path(description["config"]["full_reference"]["path"])
            assert reference.resolve() == Path(metadata.distribution("insightface").locate_file(
                "insightface/app/privateframe/docs/configuration.md"
            )).resolve()
            assert reference.is_file()
            run([sys.executable, "-I", "-B", "-m", "insightface.app.privateframe.config_reference", "--check"])
        if profile == "gui":
            commands = [[sys.executable, "-I", "-B", "-m", "insightface.gui", "--version"]]
            commands += [[str(scripts / (name + extension)), "--version"] for name in (
                "insightface-gui", "insightface-eval-studio", "insightface-desktop"
            )]
            for command in commands:
                output = run(command)
                assert f"InsightFace Evaluation Studio {version}" in output
                assert f"insightface {version}" in output
        assert not list(workdir.rglob("*.onnx")), "Smoke checks must not materialize models"
    print(f"Installed {profile} CLI checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Check built wheel and source distribution")
    inspect.add_argument("--wheel", type=Path, required=True)
    inspect.add_argument("--sdist", type=Path, required=True)
    inspect.add_argument("--allow-native", action="store_true", help="Allow the optional face3d native build")
    for name in ("installed", "_probe"):
        command = commands.add_parser(name)
        command.add_argument("--profile", choices=("base", "privateframe", "gui"), required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        inspect_archives(args.wheel, args.sdist, allow_native=args.allow_native)
    elif args.command == "installed":
        check_installed(args.profile)
    else:
        _probe(args.profile)


if __name__ == "__main__":
    main()
