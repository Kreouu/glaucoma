"""Check original B3 files and compare torchvision/timm tensor fingerprints."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import torch


def main():
    expected = {
        'efficientnet_b3_rwightman-b3899882.pth': 'b3899882250c22946d0229d266049fcd133c169233530b36b9ffa7983988362f',
        'efficientnet_b3_ra2_original.bin': '4862731817e56b83f538e76767de09caf38ce8345dcf90132c4c59fa6a0048ec',
    }
    states = []
    for name, digest in expected.items():
        path = Path('pretrain_model') / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Original file hash mismatch: {name}')
        states.append(torch.load(path, map_location='cpu', weights_only=True))

    def fingerprints(state):
        return Counter((str(t.dtype), tuple(t.shape), hashlib.sha256(t.numpy().tobytes()).hexdigest())
                       for t in state.values())

    if fingerprints(states[0]) != fingerprints(states[1]):
        raise ValueError('The torchvision and timm tensor multisets differ')
    report = dict(torchvision_tensor_count=len(states[0]), timm_tensor_count=len(states[1]),
                  identical_tensor_multiset=True, original_file_hashes=expected,
                  note='Fingerprint multiset equality does not establish graph mapping by itself; see upstream torchvision port source.')
    Path('licenses/efficientnet_b3/verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
