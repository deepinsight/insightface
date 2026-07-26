"""Manage explicitly installed model packages."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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


def _print_package(package: ModelPackage, models_dir: Path) -> None:
    status = "installed" if installed_package_name(models_dir) == package.name else "not installed"
    print(f"{package.name}\t{package.release}\t{status}")


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
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="List supported packages and installation status")
    install = commands.add_parser("install", help="Download and install a verified package")
    install.add_argument(
        "model",
        help="Package name: buffalo_l, buffalo_m, buffalo_sc, or antelopev2",
    )
    install.add_argument(
        "--accept-license",
        action="store_true",
        help="Accept the displayed model license in non-interactive environments",
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
        if args.command == "list":
            print("NAME\tRELEASE\tSTATUS")
            for package in PACKAGES.values():
                _print_package(package, models_dir)
            return 0
        if args.command == "info":
            package = package_for_name(args.model)
            print(f"Name: {package.name}")
            print(f"Release: {package.release}")
            print(f"Source: {package.url}")
            print(f"Archive SHA-256: {package.archive_sha256}")
            for item in package.files:
                print(f"{item.filename} SHA-256: {item.sha256}")
            print(license_notice(package))
            return 0
        if args.command == "install":
            package = package_for_name(args.model)
            if installed_package_name(models_dir) != package.name:
                _confirm_license(package, args.accept_license)
            status = install_package(package, models_dir)
            print(
                f"Model package {package.name} is installed and verified in "
                f"{models_dir.expanduser().resolve()}."
                if status == "installed"
                else f"Model package {package.name} is already installed and verified."
            )
            print(license_notice(package))
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
    except (ModelPackageError, RuntimeError, OSError) as exc:
        print(f"models: error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
