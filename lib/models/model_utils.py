_MODELS = {}

def register_model(cls):
    name = cls.__name__
    if name in _MODELS:
        raise ValueError(f'{name} is already registered!')
    _MODELS[name] = cls
    return cls


def get_model(name):
    return _MODELS[name]


def create_model(cfg, device, rank=None):
    model = get_model(cfg.model.name)(cfg, device, rank)
    model = model.to(device)

    return model