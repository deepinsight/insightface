"""Install an exact published release from PyPI, then test its native inference.

The workflow waits for PyPI propagation before invoking this script. Only the
network installation is retried; package provenance and inference failures fail
the job immediately. No locally built wheel is installed here.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import unquote, urlsplit


PLATFORMS = ("linux-x64", "linux-arm64", "darwin-x64", "darwin-arm64", "windows-x64")
INSTALL_ATTEMPTS = 5
INSTALL_RETRY_SECONDS = 60
PYPI_INDEX = "https://pypi.org/simple"


def clean_environment(environment):
    """Do not inherit alternate indexes, Python packages, or native overrides."""
    blocked = {
        "INSPIREFACE_LIBRARY_PATH", "INSPIREFACE_TEST_NATIVE_OVERRIDE",
        "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
        "LD_PRELOAD", "DYLD_INSERT_LIBRARIES", "VIRTUAL_ENV",
    }
    result = {
        key: value for key, value in environment.items()
        if not key.upper().startswith(("PIP_", "PYTHON")) and key.upper() not in blocked
    }
    # pip documents os.devnull as disabling every configuration file, including
    # machine-wide configuration. Clearing PIP_* also removes extra indexes.
    result["PIP_CONFIG_FILE"] = os.devnull
    return result


def load_manifest(path, version):
    if not re.fullmatch(r"[0-9][A-Za-z0-9.!+_-]*", version):
        raise ValueError("Invalid exact release version")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") != version:
        raise ValueError("Release manifest version does not match --version")
    wheels = manifest.get("wheels")
    if not isinstance(wheels, dict) or not wheels:
        raise ValueError("Release manifest must contain wheel SHA-256 hashes")
    for filename, digest in wheels.items():
        if (not isinstance(filename, str)
                or not filename.startswith("inspireface-" + version + "-")
                or not filename.endswith(".whl")
                or "/" in filename or "\\" in filename
                or not isinstance(digest, str)
                or not re.fullmatch(r"[a-fA-F0-9]{64}", digest)):
            raise ValueError("Invalid wheel filename or SHA-256 in release manifest")
    return manifest


def validate_install_report(report_path, manifest):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = [
        item for item in report.get("install", [])
        if re.sub(r"[-_.]+", "-", item.get("metadata", {}).get("name", "")).lower()
        == "inspireface"
    ]
    if len(selected) != 1:
        raise ValueError("pip report must contain exactly one installed InspireFace wheel")
    item = selected[0]
    if item.get("metadata", {}).get("version") != manifest["version"]:
        raise ValueError("Installed version does not match the release manifest")
    download = item.get("download_info", {})
    url = urlsplit(download.get("url", ""))
    if (url.scheme != "https" or url.hostname != "files.pythonhosted.org"
            or url.username or url.password or url.port not in (None, 443)
            or item.get("is_direct", False)):
        raise ValueError("InspireFace must be resolved from the public PyPI index")
    filename = unquote(url.path.rsplit("/", 1)[-1])
    expected_hash = manifest["wheels"].get(filename)
    if expected_hash is None:
        raise ValueError("Installed wheel filename is absent from the release manifest")
    actual_hash = download.get("archive_info", {}).get("hashes", {}).get("sha256")
    if not isinstance(actual_hash, str) or actual_hash.lower() != expected_hash.lower():
        raise ValueError("Installed wheel SHA-256 does not match the release manifest")
    return {
        "version": manifest["version"], "wheel": filename,
        "sha256": actual_hash.lower(), "url": download["url"],
    }


def run_command(command, *, environment, cwd, log_path):
    """Stream child output to both the Actions console and an artifact log."""
    with log_path.open("w", encoding="utf-8") as log:
        heading = "COMMAND " + json.dumps([str(part) for part in command]) + "\n"
        print(heading, end="", flush=True)
        log.write(heading)
        log.flush()
        with subprocess.Popen(
            [str(part) for part in command], cwd=str(cwd), env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ) as process:
            for raw_line in process.stdout:
                line = raw_line.decode("utf-8", errors="replace")
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()


def require_success(command, *, environment, cwd, log_path):
    result = run_command(command, environment=environment, cwd=cwd, log_path=log_path)
    if result:
        raise RuntimeError("Command exited with {}: see {}".format(result, log_path.name))


def install_from_pypi(python, version, output_dir, environment):
    for attempt in range(1, INSTALL_ATTEMPTS + 1):
        report = output_dir / "pip-install-{}.json".format(attempt)
        # Do not allow an earlier invocation's report to stand in for this one.
        report.unlink(missing_ok=True)
        command = [
            python, "-I", "-m", "pip", "--disable-pip-version-check", "--no-input",
            "--no-cache-dir", "install", "--index-url", PYPI_INDEX,
            "--only-binary=inspireface", "--force-reinstall", "--retries", "2",
            "--timeout", "60", "--report", report,
            "inspireface==" + version, "opencv-python-headless",
        ]
        result = run_command(
            command, environment=environment, cwd=output_dir,
            log_path=output_dir / "pip-install-{}.log".format(attempt),
        )
        if result == 0:
            return report
        if attempt < INSTALL_ATTEMPTS:
            print("PyPI install attempt {}/{} failed; retrying in {} seconds".format(
                attempt, INSTALL_ATTEMPTS, INSTALL_RETRY_SECONDS), flush=True)
            time.sleep(INSTALL_RETRY_SECONDS)
    raise RuntimeError("PyPI installation failed after {} attempts".format(INSTALL_ATTEMPTS))


def run_smoke(args):
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"status": "failed", "version": args.version, "platform": args.platform}
    validated_version = None
    try:
        summary["stage"] = "prepare"
        manifest = load_manifest(args.manifest.resolve(), args.version)
        validated_version = manifest["version"]
        (output_dir / "release-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        image = args.image.resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        venv = args.venv_dir.resolve()
        if venv.exists():
            raise ValueError("--venv-dir must not already exist; a fresh environment is required")
        environment = clean_environment(os.environ)
        require_success(
            [sys.executable, "-I", "-m", "venv", venv], environment=environment,
            cwd=output_dir, log_path=output_dir / "create-venv.log",
        )
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

        summary["stage"] = "install"
        report = install_from_pypi(python, args.version, output_dir, environment)
        summary["stage"] = "provenance"
        summary["installed"] = validate_install_report(report, manifest)
        for action in ("check", "freeze"):
            summary["stage"] = "pip-" + action
            require_success(
                [python, "-I", "-m", "pip", "--disable-pip-version-check", action],
                environment=environment, cwd=output_dir,
                log_path=output_dir / ("pip-" + action + ".log"),
            )

        summary["stage"] = "inference"
        verifier = Path(__file__).resolve().with_name("verify_inference.py")
        require_success(
            [python, "-I", verifier, "--expected-version", args.version,
             "--expected-platform", args.platform, "--image", image,
             "--output-dir", output_dir / "inference"],
            environment=environment, cwd=output_dir,
            log_path=output_dir / "inference.log",
        )
        summary.update(status="passed", stage="complete")
    except Exception as error:
        summary["error"] = str(error)
        raise
    finally:
        text = json.dumps(summary, indent=2) + "\n"
        (output_dir / "run-summary.json").write_text(text, encoding="utf-8")
        print(text, end="", flush=True)
        step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if step_summary:
            label = args.platform if args.platform in PLATFORMS else "unknown"
            markdown = (
                "### Published package verification\n\n"
                "| Platform | Version | Result | Stage |\n"
                "| --- | --- | --- | --- |\n"
                "| {} | `{}` | {} | {} |\n\n"
                "Download this job's artifact for installation logs, package provenance, "
                "and inference results.\n\n"
            ).format(label, validated_version or "unvalidated", summary["status"], summary["stage"])
            try:
                with Path(step_summary).open("a", encoding="utf-8") as destination:
                    destination.write(markdown)
            except OSError as error:
                print("Could not append the Actions job summary: {}".format(error), file=sys.stderr)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--platform", required=True, choices=PLATFORMS)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--venv-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    run_smoke(parser.parse_args())


if __name__ == "__main__":
    main()
