"""Keep the entire backbone trainable while adapting the new classifier faster."""


def classifier_lr_groups(model, lr, multiplier):
    head=list(model.get_classifier().parameters())
    ids={id(p) for p in head}
    backbone=[p for p in model.parameters() if id(p) not in ids]
    assert head and backbone and all(p.requires_grad for p in model.parameters())
    assert len(head)+len(backbone)==len(list(model.parameters()))
    return [dict(params=backbone,lr=lr,lr_scale=1.),dict(params=head,lr=lr*multiplier,lr_scale=multiplier)]
