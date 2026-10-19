import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from package_sdk import TYPE_MANIFEST, check_c_exports, check_elf, check_version, write_archive


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.native = self.root / 'native'
        self.har = self.root / 'har'
        self.native.mkdir()
        self.har.mkdir()
        (self.native / 'version.txt').write_text('InspireFace Version: 1.2.4\n')
        for name in ('oh-package.json5', TYPE_MANIFEST):
            path = self.har / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({'version': '1.2.4'}))

    def test_build_and_release_versions(self):
        for requested in ('', '1.2.4', 'v1.2.4'):
            self.assertEqual(check_version(self.native, self.har, requested), '1.2.4')
        self.assertEqual(check_version(self.native, self.har, 'v1.2.4-rc1'), '1.2.4-rc1')

    def test_wrong_release_tag_cannot_label_an_sdk(self):
        for requested in ('v1.2.5', '../1.2.4', '1.2.4\n', 'development'):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                check_version(self.native, self.har, requested)

    def test_stale_har_versions_are_rejected(self):
        for name in ('oh-package.json5', TYPE_MANIFEST):
            path = self.har / name
            path.write_text(json.dumps({'version': '1.2.3'}))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'HAR version'):
                check_version(self.native, self.har, '')
            path.write_text(json.dumps({'version': '1.2.4'}))

    def test_native_dependency_and_architecture_checks(self):
        valid = 'Machine: AArch64\nShared library: [libc.so]\nGNU_STACK 0x0000 RW 0x0\n'
        check_elf(valid)
        for invalid in (valid.replace('AArch64', 'X86-64'),
                        valid + 'Shared library: [liblog.so]\n',
                        valid + 'Shared library: [libc++_shared.so]\n',
                        valid.replace('RW', 'RWE'), valid.replace('GNU_STACK', 'OTHER')):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                check_elf(invalid)

    def test_napi_requires_registration_and_runtime(self):
        valid = ('Machine: AArch64\nShared library: [libc.so]\n'
                 'Shared library: [libace_napi.z.so]\nGNU_STACK 0x0000 RW 0x0\n'
                 'napi_module_register\n')
        check_elf(valid, napi=True)
        for invalid in (valid.replace('napi_module_register', ''),
                        valid.replace('Shared library: [libace_napi.z.so]', '')):
            with self.assertRaises(ValueError):
                check_elf(invalid, napi=True)

    def test_missing_c_exports_are_rejected(self):
        # Undefined references must never be counted as exported implementations.
        header = '\n'.join(f'HYPER_CAPI_EXPORT extern HResult HFExample{i}();' for i in range(125))
        symbols = '\n'.join(f'{i}: 0 8 FUNC GLOBAL DEFAULT 1 HFExample{i}' for i in range(125))
        check_c_exports(header, symbols)
        symbols = symbols.replace('GLOBAL DEFAULT 1 HFExample124', 'GLOBAL DEFAULT UND HFExample124')
        with self.assertRaisesRegex(ValueError, 'HFExample124'):
            check_c_exports(header, symbols)

    def test_archive_has_one_root_and_keeps_har_readme(self):
        package = self.root / 'inspireface-harmonyos-arm64-v8a-1.2.4'
        readme = package / 'HarmonyOS/har/README.md'
        readme.parent.mkdir(parents=True)
        readme.write_text('HAR usage')
        (package / 'version.txt').write_text('1.2.4')
        output = self.root / 'sdk.zip'
        write_archive(package, output)
        with zipfile.ZipFile(output) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(set(archive.namelist()), {
                f'{package.name}/HarmonyOS/har/README.md', f'{package.name}/version.txt'})

    def test_internal_documents_and_symlinks_cannot_enter_archive(self):
        package = self.root / 'sdk'
        package.mkdir()
        for name in ('prompt_docs/plan.txt', 'build.log', 'notes.md', '.secret'):
            path = package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('internal')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Unexpected release content'):
                write_archive(package, self.root / 'sdk.zip')
            path.unlink()
            if path.parent != package:
                path.parent.rmdir()
        (package / 'outside').symlink_to(self.native)
        with self.assertRaisesRegex(ValueError, 'Unexpected release content'):
            write_archive(package, self.root / 'sdk.zip')


if __name__ == '__main__':
    unittest.main()
