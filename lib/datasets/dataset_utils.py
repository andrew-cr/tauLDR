_DATASETS = {}

def register_dataset(cls):
    name = cls.__name__
    if name in _DATASETS:
        raise ValueError(f'{name} is already registered!')
    _DATASETS[name] = cls
    return cls

def get_dataset(cfg, device):
    return _DATASETS[cfg.data.name](cfg, device)