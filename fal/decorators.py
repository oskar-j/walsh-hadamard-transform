import collections
import inspect

globalCache = {}

def ensureHashable(args):
    def replaceUnhashable(x):
        if isinstance(x, collections.Hashable):
            return x
        else:
            return str(x)
    return map(replaceUnhashable, args)

def makeKey(function, passedArgs, kwargs):
    argspec = inspect.getargspec(function)

    args = [None] * len(argspec.args)

    if argspec.defaults:
        n = len(argspec.defaults)
        args[-n:] = argspec.defaults

    n = len(passedArgs)
    args[:n] = list(passedArgs)

    dict = {}
    for i in xrange(0, len(argspec.args)):
        dict[argspec.args[i]] = i
    for (key, value) in kwargs.items():
        args[dict[key]] = value

    return tuple([function] + ensureHashable(args))

def cached(function):
    def wrapper(*args, **kwargs):
        key = makeKey(function, args, kwargs)
        if key in globalCache.keys():
            return globalCache[key]
        else:
            result = function(*args, **kwargs)
            globalCache[key] = result
            return result
    return wrapper
