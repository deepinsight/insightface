import hashlib
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zipfile

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("windows_package_sdk", SCRIPTS / "package_sdk.py")
sdk_package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sdk_package)


def pe_dll(import_name="KERNEL32.dll", machine=0x8664):
    data = bytearray(1024)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 60, 64)
    data[64:68] = b"PE\0\0"
    struct.pack_into("<HH", data, 68, machine, 1)
    struct.pack_into("<HH", data, 84, 240, 0x2000)
    struct.pack_into("<H", data, 88, 0x20B)
    struct.pack_into("<II", data, 208, 0x1000, 40)
    struct.pack_into("<IIII", data, 336, 512, 0x1000, 512, 512)
    struct.pack_into("<IIIII", data, 512, 1, 0, 0, 0x1040, 0)
    encoded = import_name.encode("ascii") + b"\0"
    data[576:576 + len(encoded)] = encoded
    return bytes(data)


def import_library(dll="libInspireFace.dll", machine=0x8664):
    payload = b"HFQueryInspireFaceVersion\0" + dll.encode("ascii") + b"\0"
    member = struct.pack("<HHHHIIHH", 0, 0xffff, 0, machine, 0, len(payload), 0, 0) + payload
    header = b"InspireFace.dll/".ljust(16) + b"0".ljust(12) + b"0".ljust(6) + b"0".ljust(6)
    header += b"100644".ljust(8) + str(len(member)).encode("ascii").ljust(10) + b"`\n"
    return b"!<arch>\n" + header + member + (b"\n" if len(member) % 2 else b"")


class WindowsSdkPackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "CMakeLists.txt").write_text(
            "set(INSPIRE_FACE_VERSION_MAJOR 1)\nset(INSPIRE_FACE_VERSION_MINOR 2)\n"
            "set(INSPIRE_FACE_VERSION_PATCH 4)\n", encoding="utf-8")
        self.sdk = self.root / "SDK with spaces"
        for name in sdk_package.REQUIRED:
            self.write(name, b"header\n")
        self.write("version.txt", b"InspireFace Version: 1.2.4\n")
        self.write(sdk_package.CONFIG_VERSION, b'set(PACKAGE_VERSION "1.2.4")\n')
        self.write(sdk_package.CONFIG, (
            "add_library(InspireFace::InspireFace SHARED IMPORTED)\n"
            "libInspireFace.dll InspireFace.lib IMPORTED_IMPLIB INSPIRECV_API=__declspec(dllimport)\n"
        ).encode("ascii"))
        self.write(sdk_package.DLL, pe_dll())
        self.write(sdk_package.IMPLIB, import_library())

    def write(self, name, data):
        path = self.sdk / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def check(self, version=""):
        return sdk_package.check_sdk(self.repo, self.sdk, version)

    def test_release_tags_and_native_versions(self):
        for requested, expected in (("", "1.2.4"), ("1.2.4", "1.2.4"), ("v1.2.4", "1.2.4"),
                                    ("v1.2.4-rc.1", "1.2.4-rc.1"), ("1.2.4.post1", "1.2.4.post1")):
            with self.subTest(version=requested):
                self.assertEqual(self.check(requested)[0], expected)
        for requested in ("1.2.3", "v2.0.0", "../1.2.4", "1.2.4/extra", "1.2.4-", "v"):
            with self.subTest(version=requested), self.assertRaisesRegex(ValueError, "Release version"):
                self.check(requested)

    def test_stale_sdk_versions(self):
        for name, data in (("version.txt", b"InspireFace Version: 1.2.3\n"),
                           (sdk_package.CONFIG_VERSION, b'set(PACKAGE_VERSION "1.2.3")\n')):
            original = (self.sdk / name).read_bytes()
            self.write(name, data)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "version"):
                self.check()
            self.write(name, original)

    def test_missing_public_sdk_files(self):
        for name in sdk_package.REQUIRED:
            path = self.sdk / name
            data = path.read_bytes()
            path.unlink()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "missing required"):
                self.check()
            path.write_bytes(data)

    def test_no_internal_or_foreign_payload(self):
        for name in ("prompt_docs/notes.md", "build.log", "InspireFace/lib/libInspireFace.so",
                     "InspireFace/lib/MNN.dll", "InspireFace/lib/libInspireFace.dylib",
                     "InspireFace/include/private.md", ".git/config", "test/Test.exe"):
            self.write(name, b"unexpected")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unexpected"):
                self.check()
            (self.sdk / name).unlink()
            if name == ".git/config":
                (self.sdk / ".git").rmdir()

    def test_native_dll_architecture_and_dependencies(self):
        for data in (pe_dll(machine=0x14c), pe_dll(machine=0xaa64), pe_dll("MSVCP140D.dll"),
                     pe_dll("MNN.dll"), b"not a DLL"):
            with self.subTest(data=data[:8]), self.assertRaises(ValueError):
                self.write(sdk_package.DLL, data)
                self.check()

    def test_import_library_identity(self):
        for data in (import_library(machine=0x14c), import_library("Other.dll"), b"not an import library",
                     import_library()[:-3]):
            with self.subTest(data=data[:8]), self.assertRaises(ValueError):
                self.write(sdk_package.IMPLIB, data)
                self.check()

    def test_reject_static_cmake_package(self):
        path = self.sdk / sdk_package.CONFIG
        path.write_text(path.read_text().replace("SHARED", "STATIC"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "shared InspireFace target"):
            self.check()

    def test_archive_preserves_layout_and_checksum(self):
        self.write("InspireFace/include/inspirecv/task/types.h", b"transitive header\n")
        destination = sdk_package.package_sdk(self.repo, self.sdk, self.root / "output", "v1.2.4-rc.1")
        self.assertEqual(destination.name, "inspireface-windows-x64-1.2.4-rc.1.zip")
        with zipfile.ZipFile(destination) as archive:
            expected = {destination.stem + "/" + path.relative_to(self.sdk).as_posix(): path.read_bytes()
                        for path in self.sdk.rglob("*") if path.is_file()}
            self.assertEqual(set(archive.namelist()), set(expected))
            for name, data in expected.items():
                self.assertEqual(archive.read(name), data)
        checksum = hashlib.sha256(destination.read_bytes()).hexdigest() + "  " + destination.name + "\n"
        self.assertEqual(destination.with_suffix(".zip.sha256").read_text(), checksum)
        with self.assertRaises(FileExistsError):
            sdk_package.package_sdk(self.repo, self.sdk, self.root / "output", "v1.2.4-rc.1")

    def test_output_cannot_pollute_sdk(self):
        for output in (self.sdk, self.sdk / "output"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "outside"):
                sdk_package.package_sdk(self.repo, self.sdk, output)


if __name__ == "__main__":
    unittest.main()
