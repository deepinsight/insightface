import copy
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_archive import EXPECTED, check_manifest, check_package, write_archive


class ReleasePolicyTests(unittest.TestCase):
    def setUp(self):
        self.manifest = [dict(sdk=sdk, arch=arch, backend='cpu') for sdk, arch in sorted(EXPECTED)]

    def test_complete_cpu_package(self):
        self.assertEqual(check_manifest(self.manifest), EXPECTED)

    def test_coreml_cannot_enter_release(self):
        for mixed in (True, False):
            manifest = copy.deepcopy(self.manifest)
            for entry in (manifest[:1] if mixed else manifest):
                entry['backend'] = 'coreml'
            with self.assertRaisesRegex(ValueError, 'CPU SDKs only'):
                check_manifest(manifest)

    def test_missing_simulator_and_duplicate_architecture_fail(self):
        for manifest in (self.manifest[:-1], self.manifest + [self.manifest[0]]):
            with self.assertRaisesRegex(ValueError, 'incomplete or duplicated'):
                check_manifest(manifest)

    def test_engineering_docs_cannot_enter_release(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / 'sdk-manifest.json').write_text(json.dumps(self.manifest))
            for name in ('InspireFace.xcframework', 'InspireFaceSwift.xcframework', 'Frameworks', 'SDKs'):
                (package / name).mkdir()
            (package / 'prompt_docs').mkdir()
            with self.assertRaisesRegex(ValueError, 'top-level'):
                check_package(package)
            (package / 'prompt_docs').rmdir()
            (package / 'SDKs/private-plan.md').write_text('local only')
            with self.assertRaisesRegex(ValueError, 'Internal documents'):
                check_package(package)

    def test_zip_preserves_framework_links_and_executable_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            package = directory / 'sdk'
            version = package / 'InspireFace.framework/Versions/A'
            version.mkdir(parents=True)
            binary = version / 'InspireFace'
            binary.write_bytes(b'framework fixture')
            binary.chmod(0o755)
            (version.parent / 'Current').symlink_to('A')
            (package / 'InspireFace.framework/InspireFace').symlink_to('Versions/Current/InspireFace')
            archive = directory / 'release.zip'
            write_archive(package, archive, 'inspireface-apple-test')
            with zipfile.ZipFile(archive) as stream:
                link = stream.getinfo('inspireface-apple-test/InspireFace.framework/InspireFace')
                self.assertTrue(stat.S_ISLNK(link.external_attr >> 16))
            restored = directory / 'restored'
            subprocess.run(['unzip', '-q', str(archive), '-d', str(restored)], check=True)
            binary = restored / 'inspireface-apple-test/InspireFace.framework/InspireFace'
            self.assertTrue(binary.is_symlink())
            self.assertEqual(binary.read_bytes(), b'framework fixture')
            self.assertEqual(binary.stat().st_mode & 0o777, 0o755)

    def test_finder_metadata_is_not_distributed(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            package = directory / 'sdk'
            framework = package / 'InspireFace.framework'
            framework.mkdir(parents=True)
            (package / '.DS_Store').write_bytes(b'finder metadata')
            (framework / '.DS_Store').write_bytes(b'finder metadata')
            (framework / 'InspireFace').write_bytes(b'binary fixture')
            archive = directory / 'release.zip'
            write_archive(package, archive, 'sdk')
            with zipfile.ZipFile(archive) as stream:
                self.assertEqual(stream.namelist(), ['sdk/InspireFace.framework/InspireFace'])


if __name__ == '__main__':
    unittest.main()
