"""Fetch pinned original InceptionNeXt files and verify every manifest hash, without model loading."""

import hashlib
import json
import urllib.request
from pathlib import Path


def main():
    manifest = json.loads(Path('licenses/inception_next_tiny/manifest.json').read_text())
    for entry in manifest['files']:
        path = Path(entry['path'])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(entry['url'], timeout=120) as response:
                data = response.read()
            if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise ValueError(f'Download hash mismatch: {path}')
            path.write_bytes(data)
        data = path.read_bytes()
        if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f'File differs from original: {path}')
        print(f'VERIFIED {path}', flush=True)


if __name__ == '__main__':
    main()
