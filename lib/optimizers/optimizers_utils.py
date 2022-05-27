_OPTIMIZERS = {}

def register_optimizer(function):
    name = function.__name__
    if name in _OPTIMIZERS:
        raise ValueError(f'{name} is already registered!')
    _OPTIMIZERS[name] = function
    return function


def get_optimizer(model_parameters, cfg):
    return _OPTIMIZERS[cfg.optimizer.name](model_parameters, cfg)

