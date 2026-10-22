import base64
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "verify_inference.py"
SPEC = importlib.util.spec_from_file_location("verify_inference", SCRIPT)
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)
VERSION = "1.2.4.post2"


class Record(str):
    def __new__(cls, name, data):
        record = super().__new__(cls, name)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")
        record.hash = SimpleNamespace(mode="sha256", value=digest)
        record.size = len(data)
        return record


class InstallationVerificationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.environment = self.root / "venv"
        self.site = self.environment / "site-packages"

    def installation(self, target="windows-x64"):
        names = ["inspireface/__init__.py", "inspireface/modules/core/libs/" + verify.PLATFORM_LIBRARIES[target]]
        records = []
        for name in names:
            path = self.site / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = ("installed-wheel:" + name).encode("utf-8")
            path.write_bytes(data)
            records.append(Record(name, data))
        package = SimpleNamespace(version=VERSION, files=records, locate_file=lambda name: self.site / str(name))
        isf = SimpleNamespace(__version__=VERSION, __file__=str(self.site / names[0]))
        library = str(self.site / names[1])
        native = SimpleNamespace(_LIBRARY_FILENAME=library, _libs={
            library: SimpleNamespace(access={"cdecl": SimpleNamespace(_name=library)})})
        return package, isf, native, target

    def check(self, installation):
        package, isf, native, target = installation
        return verify.verify_installation(package, isf, native, VERSION, target, self.environment)

    def test_accepts_installed_native_library_for_each_release_platform(self):
        for target in verify.PLATFORM_LIBRARIES:
            with self.subTest(target=target):
                package_path, native_path = self.check(self.installation(target))
                self.assertEqual(package_path, (self.site / "inspireface/__init__.py").resolve())
                self.assertEqual(native_path, (self.site / "inspireface/modules/core/libs" /
                                              verify.PLATFORM_LIBRARIES[target]).resolve())

    def test_rejects_wrong_metadata_or_imported_package_version(self):
        for index, attribute in ((0, "version"), (1, "__version__")):
            with self.subTest(attribute=attribute):
                installed = self.installation()
                setattr(installed[index], attribute, "1.2.4.post1")
                with self.assertRaisesRegex(RuntimeError, "Expected InspireFace"):
                    self.check(installed)

    def test_rejects_source_tree_import(self):
        installed = self.installation()
        installed[1].__file__ = str(self.root / "checkout/python/inspireface/__init__.py")
        with self.assertRaisesRegex(RuntimeError, "installed distribution"):
            self.check(installed)

    def test_rejects_installation_outside_current_environment(self):
        installed = self.installation()
        self.environment = self.root / "other-venv"
        with self.assertRaisesRegex(RuntimeError, "current Python environment"):
            self.check(installed)

    def test_rejects_selected_native_override(self):
        installed = self.installation()
        installed[2]._LIBRARY_FILENAME = str(self.root / "checkout/libInspireFace.dll")
        with self.assertRaisesRegex(RuntimeError, "selected a native library outside"):
            self.check(installed)

    def test_rejects_loaded_native_override(self):
        installed = self.installation()
        native = installed[2]
        native._libs[native._LIBRARY_FILENAME].access["cdecl"]._name = str(self.root / "libInspireFace.dll")
        with self.assertRaisesRegex(RuntimeError, "loaded a native library outside"):
            self.check(installed)

    def test_rejects_missing_native_library(self):
        installed = self.installation()
        Path(installed[2]._LIBRARY_FILENAME).unlink()
        with self.assertRaisesRegex(RuntimeError, "no native library"):
            self.check(installed)

    def test_rejects_installed_native_modified_after_installation(self):
        installed = self.installation()
        library = Path(installed[2]._LIBRARY_FILENAME)
        library.write_bytes(library.read_bytes().replace(b"installed", b"modified!"))
        with self.assertRaisesRegex(RuntimeError, "does not match wheel RECORD"):
            self.check(installed)

    def test_rejects_missing_native_record_or_hash(self):
        for remove_record in (True, False):
            with self.subTest(remove_record=remove_record):
                installed = self.installation()
                if remove_record:
                    installed[0].files.pop()
                else:
                    installed[0].files[-1].hash = None
                with self.assertRaisesRegex(RuntimeError, "Missing wheel RECORD"):
                    self.check(installed)

    def test_normalizes_runner_architecture(self):
        for system, machine, expected in (
            ("Windows", "AMD64", "windows-x64"),
            ("Darwin", "x86_64", "darwin-x64"),
            ("Darwin", "arm64", "darwin-arm64"),
            ("Linux", "x86_64", "linux-x64"),
            ("Linux", "aarch64", "linux-arm64"),
        ):
            with self.subTest(system=system, machine=machine):
                self.assertEqual(verify.platform_key(system, machine), expected)
        with self.assertRaisesRegex(RuntimeError, "Unsupported test platform"):
            verify.platform_key("Windows", "x86")


if __name__ == "__main__":
    unittest.main()
