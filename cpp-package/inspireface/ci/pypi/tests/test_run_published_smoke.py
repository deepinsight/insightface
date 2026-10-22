import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "run_published_smoke", Path(__file__).resolve().parents[1] / "run_published_smoke.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)

VERSION = "1.2.4.post3"
WHEEL = "inspireface-{}-py3-none-win_amd64.whl".format(VERSION)
DIGEST = "a" * 64


class PublishedSmokeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.manifest = {"version": VERSION, "wheels": {WHEEL: DIGEST}}
        self.report = {
            "install": [{
                "metadata": {"name": "inspireface", "version": VERSION},
                "is_direct": False,
                "download_info": {
                    "url": "https://files.pythonhosted.org/packages/aa/" + WHEEL,
                    "archive_info": {"hashes": {"sha256": DIGEST}},
                },
            }],
        }
        self.report_path = self.root / "report.json"

    def validate(self):
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")
        return smoke.validate_install_report(self.report_path, self.manifest)

    def test_accepts_exact_release_wheel_hash_from_pypi(self):
        result = self.validate()
        self.assertEqual(result["wheel"], WHEEL)
        self.assertEqual(result["sha256"], DIGEST)

    def test_rejects_wrong_version_even_if_manifest_wheel_hash_matches(self):
        self.report["install"][0]["metadata"]["version"] = "1.2.4.post2"
        with self.assertRaisesRegex(ValueError, "version"):
            self.validate()

    def test_rejects_hash_or_filename_outside_release(self):
        download = self.report["install"][0]["download_info"]
        download["archive_info"]["hashes"]["sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.validate()
        download["url"] = "https://files.pythonhosted.org/another.whl"
        with self.assertRaisesRegex(ValueError, "filename"):
            self.validate()

    def test_rejects_non_pypi_downloads_and_direct_urls(self):
        item = self.report["install"][0]
        for url in ("file:///tmp/" + WHEEL, "https://mirror.example/" + WHEEL,
                    "http://files.pythonhosted.org/" + WHEEL,
                    "https://files.pythonhosted.org.evil.example/" + WHEEL):
            with self.subTest(url=url):
                item["download_info"]["url"] = url
                with self.assertRaisesRegex(ValueError, "public PyPI"):
                    self.validate()
        item["download_info"]["url"] = "https://files.pythonhosted.org/" + WHEEL
        item["is_direct"] = True
        with self.assertRaisesRegex(ValueError, "public PyPI"):
            self.validate()

    def test_rejects_missing_or_duplicate_inspireface_install(self):
        item = self.report["install"][0]
        for items in ([], [item, item]):
            with self.subTest(items=len(items)):
                self.report["install"] = items
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    self.validate()

    def test_disables_inherited_python_pip_and_native_overrides(self):
        environment = smoke.clean_environment({
            "PATH": "/bin", "PIP_INDEX_URL": "https://mirror.example",
            "PIP_EXTRA_INDEX_URL": "https://other.example", "PIP_CONFIG_FILE": "/tmp/pip.conf",
            "PIP_FIND_LINKS": "/tmp/wheels", "PYTHONPATH": "/checkout/python",
            "PYTHONHOME": "/tmp/other", "INSPIREFACE_LIBRARY_PATH": "/tmp/native",
            "INSPIREFACE_TEST_NATIVE_OVERRIDE": "1", "DYLD_LIBRARY_PATH": "/tmp/native",
            "LD_PRELOAD": "/tmp/override.so", "VIRTUAL_ENV": "/tmp/old",
        })
        self.assertEqual(environment, {"PATH": "/bin", "PIP_CONFIG_FILE": smoke.os.devnull})

    @patch.object(smoke.time, "sleep")
    @patch.object(smoke, "run_command", side_effect=[1, 1, 0])
    def test_only_installs_exact_version_and_retries_with_bounded_delay(self, run, sleep):
        result = smoke.install_from_pypi(Path("python"), VERSION, self.root, {})
        self.assertEqual(result.name, "pip-install-3.json")
        self.assertEqual(run.call_count, 3)
        self.assertEqual([call.args for call in sleep.call_args_list], [(60,), (60,)])
        for call in run.call_args_list:
            command = call.args[0]
            self.assertIn("inspireface==" + VERSION, command)
            self.assertIn("--only-binary=inspireface", command)
            self.assertIn("--no-cache-dir", command)
            self.assertIn("--force-reinstall", command)
            self.assertEqual(command[command.index("--index-url") + 1], smoke.PYPI_INDEX)

    @patch.object(smoke.time, "sleep")
    @patch.object(smoke, "run_command", return_value=1)
    def test_stops_after_five_install_attempts(self, run, sleep):
        with self.assertRaisesRegex(RuntimeError, "after 5 attempts"):
            smoke.install_from_pypi(Path("python"), VERSION, self.root, {})
        self.assertEqual(run.call_count, 5)
        self.assertEqual(sleep.call_count, 4)

    def arguments(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(self.manifest), encoding="utf-8")
        image = self.root / "face.jpg"
        image.write_bytes(b"test fixture")
        return argparse.Namespace(
            version=VERSION, platform="windows-x64", image=image,
            output_dir=self.root / "result", venv_dir=self.root / "venv", manifest=manifest)

    def fake_command(self, command, **kwargs):
        if "--report" in command:
            Path(command[command.index("--report") + 1]).write_text(
                json.dumps(self.report), encoding="utf-8")
        return 0

    @patch.object(smoke.time, "sleep")
    def test_inference_failure_propagates_without_install_or_inference_retry(self, sleep):
        args = self.arguments()

        def run(command, **kwargs):
            if "--expected-version" in command:
                return 17
            return self.fake_command(command, **kwargs)

        with patch.object(smoke, "run_command", side_effect=run) as commands:
            with self.assertRaisesRegex(RuntimeError, "exited with 17"):
                smoke.run_smoke(args)
        self.assertEqual(sum("install" in call.args[0] for call in commands.call_args_list), 1)
        self.assertEqual(sum("--expected-version" in call.args[0] for call in commands.call_args_list), 1)
        sleep.assert_not_called()
        summary = json.loads((args.output_dir / "run-summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["stage"], "inference")

    @patch.object(smoke.time, "sleep")
    def test_provenance_failure_does_not_retry_or_run_inference(self, sleep):
        args = self.arguments()
        self.report["install"][0]["download_info"]["archive_info"]["hashes"]["sha256"] = "b" * 64
        with patch.object(smoke, "run_command", side_effect=self.fake_command) as commands:
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                smoke.run_smoke(args)
        self.assertEqual(commands.call_count, 2)
        sleep.assert_not_called()
        summary = json.loads((args.output_dir / "run-summary.json").read_text())
        self.assertEqual(summary["stage"], "provenance")

    def test_success_uses_isolated_venv_python_and_expected_platform(self):
        args = self.arguments()
        summary_path = self.root / "step-summary.md"
        with patch.dict(smoke.os.environ, {"GITHUB_STEP_SUMMARY": str(summary_path)}):
            with patch.object(smoke, "run_command", side_effect=self.fake_command) as commands:
                summary = smoke.run_smoke(args)
        self.assertEqual(summary["status"], "passed")
        command = commands.call_args_list[-1].args[0]
        self.assertEqual(command[1], "-I")
        self.assertEqual(command[command.index("--expected-version") + 1], VERSION)
        self.assertEqual(command[command.index("--expected-platform") + 1], "windows-x64")
        self.assertTrue(Path(command[0]).is_relative_to(args.venv_dir.resolve()))
        self.assertEqual([call.args[0][-1] for call in commands.call_args_list[2:4]], ["check", "freeze"])
        self.assertIn("windows-x64 | `{}` | passed | complete".format(VERSION),
                      summary_path.read_text(encoding="utf-8"))

    def test_refuses_existing_venv_and_wrong_manifest_version(self):
        args = self.arguments()
        args.venv_dir.mkdir()
        with patch.object(smoke, "run_command") as commands:
            with self.assertRaisesRegex(ValueError, "fresh environment"):
                smoke.run_smoke(args)
            args.version = "1.2.4.post2"
            with self.assertRaisesRegex(ValueError, "manifest version"):
                smoke.run_smoke(args)
        commands.assert_not_called()


if __name__ == "__main__":
    unittest.main()
