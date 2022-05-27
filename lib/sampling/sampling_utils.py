_SAMPLERS = {}

def register_sampler(cls):
    name = cls.__name__
    if name in _SAMPLERS:
        raise ValueError(f'{name} is already registered!')
    _SAMPLERS[name] = cls
    return cls

def get_sampler(cfg):
    return _SAMPLERS[cfg.sampler.name](cfg)