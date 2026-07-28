"""Small caching helper used to memoise expensive matrix construction."""

from __future__ import annotations

import functools
from collections.abc import Callable, Hashable
from typing import Any, ParamSpec, TypeVar

__all__ = ["cached"]

P = ParamSpec("P")
R = TypeVar("R")

#: Separates positional from keyword arguments in a cache key. Must be a module
#: level constant: a fresh marker per call would make every lookup miss.
_KWARGS_MARKER = object()


def _hashable(value: Any) -> Hashable:
    """Return ``value`` if it can be used as a dict key, else its ``repr``."""
    if isinstance(value, Hashable):
        return value
    return repr(value)


def _make_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Hashable:
    key: tuple[Hashable, ...] = tuple(_hashable(a) for a in args)
    if kwargs:
        key = (
            *key,
            _KWARGS_MARKER,
            *((name, _hashable(value)) for name, value in sorted(kwargs.items())),
        )
    return key


def cached(function: Callable[P, R]) -> Callable[P, R]:
    """Memoise ``function`` on its arguments, tolerating unhashable ones.

    Unlike :func:`functools.lru_cache` this accepts unhashable arguments by
    falling back to their ``repr``, which is what the numpy-heavy call sites
    here need. The cache is unbounded and lives for the life of the process.
    """
    cache: dict[Hashable, R] = {}

    @functools.wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        key = _make_key(args, kwargs)
        try:
            return cache[key]
        except KeyError:
            result = cache[key] = function(*args, **kwargs)
            return result

    wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
    wrapper.cache_size = lambda: len(cache)  # type: ignore[attr-defined]
    return wrapper
