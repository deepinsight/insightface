#!/usr/bin/env python3
"""Transfer only validated CPU slices between existing native Apple jobs."""
import argparse
import json
from pathlib import Path
import tarfile
from assemble import cpu_sdks

p = argparse.ArgumentParser()
p.add_argument('operation', choices=['pack', 'unpack'])
p.add_argument('--archive', type=Path, required=True)
p.add_argument('--directory', type=Path, default=Path('build'))
p.add_argument('--version', default='')
a = p.parse_args()
if a.operation == 'pack':
    sdks = {key: value for key, value in cpu_sdks(a.directory, a.version).items() if key[1] == 'x86_64'}
    if set(sdks) != {('macosx', 'x86_64'), ('iphonesimulator', 'x86_64')}:
        raise RuntimeError('Intel transfer requires both CPU macOS and simulator slices')
    a.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.archive, 'w:gz') as archive:
        for sdk in sdks.values():
            archive.add(sdk, arcname=sdk.name)
else:
    a.directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.archive) as archive:
        # Artifacts originate from the same workflow; still reject paths outside build/.
        for member in archive.getmembers():
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or member.isdev():
                raise RuntimeError(f'Unsafe transfer path: {member.name}')
            if member.issym() or member.islnk():
                target = (a.directory / path.parent / member.linkname).resolve()
                if a.directory.resolve() not in target.parents:
                    raise RuntimeError(f'Unsafe transfer link: {member.name}')
        roots = {Path(member.name).parts[0] for member in archive.getmembers()}
        archive.extractall(a.directory, filter='data')
    identities = []
    for name in roots:
        metadata = json.loads((a.directory / name / 'sdk-info.json').read_text())
        identities.append((metadata['sdk'], metadata['arch']))
        if metadata['backend'] != 'cpu':
            raise RuntimeError('Non-CPU SDK in transfer')
    if len(identities) != 2 or set(identities) != {('macosx', 'x86_64'), ('iphonesimulator', 'x86_64')}:
        raise RuntimeError('Transferred artifact must contain exactly the two CPU Intel slices')
