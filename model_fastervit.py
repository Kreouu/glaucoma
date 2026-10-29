"""FasterViT-1 any-resolution, pinned source and audited local initialization."""

import torch
from vendor.fastervit.faster_vit_any_res import FasterViT


def build_model(pretrained_path=None, size=512):
    if size != 512:
        raise ValueError('This comparison supports 512 input only')
    # Preserve the pretrained stage-3 2x2 carrier-token layout at 512 resolution.
    model = FasterViT(dim=80, in_dim=32, depths=[1, 3, 8, 5],
                      num_heads=[2, 4, 8, 16], window_size=[7, 7, 16, 16],
                      ct_size=2, mlp_ratio=4, resolution=[size, size],
                      drop_path_rate=0.0, hat=[False, False, True, False])
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=True)
        state = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        target = model.state_dict()
        # Coordinate/index buffers and computed bias caches depend on input/window size.
        rebuilt = [k for k in target if k.endswith(
            ('relative_coords_table', 'relative_position_index', 'relative_bias'))]
        for key in rebuilt:
            state[key] = target[key]
        model.load_state_dict(state, strict=True)
        parameters = dict(model.named_parameters())
        assert all(torch.equal(value.detach(), state[name]) for name, value in parameters.items())
        print(f'FasterViT pretrained parameters {len(parameters)}/{len(parameters)} exact; '
              f'rebuilt {len(rebuilt)} geometry/cache buffers', flush=True)
    model.head = torch.nn.Linear(model.head.in_features, 1)
    model.num_classes = 1
    return model
