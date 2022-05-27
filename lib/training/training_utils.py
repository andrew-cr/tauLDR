_TRAINSTEPS = {}

def register_train_step(cls):
    name = cls.__name__
    if name in _TRAINSTEPS:
        raise ValueError(f'{name} is already registered!')
    _TRAINSTEPS[name] = cls
    return cls

def get_train_step(cfg):
    return _TRAINSTEPS[cfg.training.train_step_name](cfg)
