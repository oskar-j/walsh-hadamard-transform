import collections
import inspect

globalCache = {}


def ensure_hashable(args):
    def replace_unhashable(x):
        if isinstance(x, collections.Hashable):
            return x
        else:
            return str(x)
    return map(replace_unhashable, args)


def make_key(function, passed_args, kwargs):
    argspec = inspect.getargspec(function)

    args = [None] * len(argspec.args)

    if argspec.defaults:
        n = len(argspec.defaults)
        args[-n:] = argspec.defaults

    n = len(passed_args)
    args[:n] = list(passed_args)

    keymap = {}
    for i in xrange(0, len(argspec.args)):
        keymap[argspec.args[i]] = i
    for (key, value) in kwargs.items():
        args[keymap[key]] = value

    return tuple([function] + ensure_hashable(args))


def cached(function):
    def wrapper(*args, **kwargs):
        key = make_key(function, args, kwargs)
        if key in globalCache.keys():
            return globalCache[key]
        else:
            result = function(*args, **kwargs)
            globalCache[key] = result
            return result
    return wrapper
