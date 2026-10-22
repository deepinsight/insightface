"""Manage explicitly installed model packages."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path

from .addon_config import (
    liveness_config_lock,
    require_editable_config,
    write_enabled_addons,
)
from .addons import install_addon, require_installed_addon
from .config import SUPPORTED_ADDONS, load_server_config
from .licensing import ModelLicense
from .models.packages import (
    PACKAGES,
    ModelPackage,
    ModelPackageError,
    install_package,
    installed_package_name,
    license_notice,
    package_for_name,
    verify_installed,
)


def _models_dir(value: str | None) -> Path:
    configured = value if value is not None else os.environ.get("INSIGHTFACE_MODELS_DIR", "/models")
    return Path(configured)


def _confirm_license(package: ModelPackage, accepted: bool) -> None:
    print(license_notice(package))
    if accepted:
        return
    if not sys.stdin.isatty():
        raise ModelPackageError(
            "License acceptance is required in non-interactive mode; add --accept-license."
        )
    answer = input("Accept this model license and continue? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        raise ModelPackageError("Model installation cancelled; license was not accepted.")


def _print_package(package: ModelPackage, installed: str | None) -> None:
    status = "installed" if installed == package.name else "not installed"
    print(f"{package.name}\t{status}")


def _print_verified_license(license_info: ModelLicense) -> None:
    summary = license_info.public_summary()
    print("LICENSE VERIFIED")
    print(f"Issuer: {summary['issuer']}")
    print(f"License ID: {summary['license_id']}")
    print(f"Model ID: {summary['model_id']}")
    print(f"Grant: {summary['grant']}")
    if summary["customer"]:
        print(f"Customer: {summary['customer']}")
    if summary["reference"]:
        print(f"Reference: {summary['reference']}")
    print(f"Valid from: {summary['valid_from']}")
    print(f"Valid until: {summary['valid_until'] or 'perpetual'}")
    print("Signature: VALID")
    print(
        "Commercial use: "
        + ("PERMITTED" if summary["commercial_use_permitted"] else "NOT PERMITTED")
    )
    if not summary["commercial_use_permitted"]:
        print("Commercial licensing: https://www.insightface.ai")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="models", description=__doc__)
    parser.add_argument("--models-dir", help="Override INSIGHTFACE_MODELS_DIR")
    parser.add_argument("--config-file", help="Override INSIGHTFACE_CONFIG_FILE")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="List supported packages and installation status")
    addons = commands.add_parser("addons", help="Install or verify flat addon models")
    addon_commands = addons.add_subparsers(dest="addon_command", required=True)
    for operation in ("install", "verify"):
        command = addon_commands.add_parser(operation)
        command.add_argument("addon", choices=SUPPORTED_ADDONS)
    install = commands.add_parser("install", help="Download and install a verified package")
    install.add_argument(
        "model",
        help=f"Package name: {', '.join(PACKAGES)}",
    )
    install.add_argument(
        "--accept-license",
        action="store_true",
        help="Accept the displayed model license in non-interactive environments",
    )
    install.add_argument(
        "--enable-liveness",
        action="store_true",
        help="Install and verify liveness, then enable it in server.toml for the next Server start",
    )
    verify = commands.add_parser(
        "verify", help="Verify model files, manifest, and the signed model license"
    )
    verify.add_argument("model", nargs="?", help="Optionally require this installed package")
    info = commands.add_parser("info", help="Show package source, hashes, and license")
    info.add_argument("model", help="Package name")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    models_dir = _models_dir(args.models_dir)
    try:
        config_value = args.config_file or os.environ.get("INSIGHTFACE_CONFIG_FILE")
        config_path = Path(config_value).expanduser() if config_value else None
        if args.command == "install" and args.enable_liveness:
            # Reject unsuitable mounts before downloading even the base package.
            config_path = require_editable_config(config_path)
        config = load_server_config(config_path)
        if args.command == "addons":
            operation = install_addon if args.addon_command == "install" else require_installed_addon
            path = operation(args.addon, models_dir)
            print(f"Addon {args.addon} is installed and verified: {path}")
            return 0
        if args.command == "list":
            print("NAME\tSTATUS")
            installed = installed_package_name(models_dir)
            for package in PACKAGES.values():
                _print_package(package, installed)
            return 0
        if args.command == "info":
            package = package_for_name(args.model)
            print(f"Name: {package.name}")
            print(f"Source: {package.url}")
            print(f"Archive SHA-256: {package.archive_sha256}")
            for item in package.files:
                print(f"{item.filename} SHA-256: {item.sha256}")
            print(license_notice(package))
            return 0
        if args.command == "install":
            package = package_for_name(args.model)
            lock = (
                liveness_config_lock(config_path)
                if args.enable_liveness and config_path is not None
                else nullcontext()
            )
            with lock:
                if args.enable_liveness:
                    config_path = require_editable_config(config_path)
                    config = load_server_config(config_path)
                if installed_package_name(models_dir) != package.name:
                    _confirm_license(package, args.accept_license)
                status = install_package(package, models_dir)
                requested_addons = tuple(dict.fromkeys(
                    (*config.auto_download_addons, *(("liveness",) if args.enable_liveness else ()))
                ))
                for addon in requested_addons:
                    try:
                        path = install_addon(addon, models_dir)
                        if args.enable_liveness:
                            require_installed_addon(addon, models_dir)
                    except (OSError, RuntimeError) as exc:
                        if args.enable_liveness:
                            raise ModelPackageError(
                                f"Base model package {package.name} is installed, but addon "
                                f"{addon} could not be installed or verified. Configuration was "
                                "not changed by this command. Check the model cache and network, "
                                "then rerun the same install command to reuse verified files."
                            ) from exc
                        raise ModelPackageError(
                            f"Base model package {package.name} is installed, but addon "
                            f"{addon} installation failed: {exc}. Rerun the same install "
                            "command to reuse the base package and complete the addons."
                        ) from exc
                    print(f"Addon {addon} is installed and verified: {path}")
                if args.enable_liveness:
                    assert config_path is not None
                    try:
                        write_enabled_addons(config_path, ("liveness",))
                    except (OSError, RuntimeError, ValueError) as exc:
                        raise ModelPackageError(
                            "Could not save the liveness configuration. The downloaded models "
                            "can be reused. Check server.toml, directory permissions, and "
                            "concurrent edits, then rerun the same command. Configuration "
                            "enablement has not been confirmed; the Server was not restarted."
                        ) from exc
                print(
                    f"Model package {package.name} is installed and verified in "
                    f"{models_dir.expanduser().resolve()}."
                    if status == "installed"
                    else f"Model package {package.name} is already installed and verified."
                )
                print(license_notice(package))
                if args.enable_liveness:
                    print("Liveness is installed and configured for the next Server start.")
                    print(
                        "Restart a running Server to apply the saved settings; "
                        "this command does not restart it."
                    )
                    print(
                        "Compose: docker compose -f server/deploy/compose.cpu.yml restart server "
                        "(use compose.cuda12.yml for CUDA and your deployment overrides)."
                    )
            return 0
        if args.command == "verify":
            installed, models, license_info = verify_installed(models_dir)
            if args.model and installed != args.model:
                raise ModelPackageError(
                    f"Expected {args.model!r}, but the installed package is {installed or 'custom/unknown'}."
                )
            print(f"Installed package: {installed or 'custom/unknown'}")
            for model in models:
                print(f"Model file: OK ({model['file']})")
            _print_verified_license(license_info)
            return 0
        parser.error("unknown command")
    except (ModelPackageError, RuntimeError, OSError, ValueError) as exc:
        print(f"models: error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
