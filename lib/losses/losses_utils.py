_LOSSES = {}

def register_loss(cls):
    name = cls.__name__
    if name in _LOSSES:
        raise ValueError(f'{name} is already registered!')
    _LOSSES[name] = cls
    return cls

def get_loss(cfg):
    return _LOSSES[cfg.loss.name](cfg)