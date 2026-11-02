"""Six ConvNeXt stage groups; all parameters stay trainable."""


def stage_lr_groups(model, lr, decay):
    groups = [dict(params=[], lr=lr * decay ** (5 - level),
                   lr_scale=decay ** (5 - level)) for level in range(6)]
    for name, parameter in model.named_parameters():
        if name.startswith('stem.'):
            level = 0
        elif name.startswith('stages.'):
            level = int(name.split('.')[1]) + 1
        elif name.startswith('head.'):
            level = 5
        else:
            raise ValueError(f'Unexpected ConvNeXt parameter: {name}')
        assert parameter.requires_grad
        groups[level]['params'].append(parameter)
    assert all(group['params'] for group in groups)
    return groups
