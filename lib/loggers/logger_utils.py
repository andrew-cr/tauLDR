_LOGGERS = {}

def register_logger(func):
    name = func.__name__
    if name in _LOGGERS:
        raise ValueError(f'{name} is already registered!')
    _LOGGERS[name] = func
    return func

def get_logger(name):
    return _LOGGERS[name]
