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
    """Coerce one argument into something usable as a dict key.

    Args:
        value: The argument to coerce.

    Returns:
        ``value`` itself if it is hashable, otherwise its ``repr``. Two
        unequal-but-identically-represented objects therefore collide, which is
        acceptable here: the cached callables are pure.
    """
    if isinstance(value, Hashable):
        return value
    return repr(value)


def _make_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Hashable:
    """Build a cache key from a call's arguments.

    Args:
        args: Positional arguments, including ``self`` for methods.
        kwargs: Keyword arguments. Sorted by name, so call order does not
            produce a different key.

    Returns:
        A hashable key identifying this argument combination.
    """
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

    Args:
        function: The callable to memoise.

    Returns:
        A wrapper around ``function`` with the same signature, carrying
        ``cache_clear()`` and ``cache_size()`` attributes.
    """
    cache: dict[Hashable, R] = {}

    @functools.wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        """Return the memoised result, computing it on the first call.

        Args:
            *args: Positional arguments forwarded to the wrapped callable.
            **kwargs: Keyword arguments forwarded to the wrapped callable.

        Returns:
            Whatever the wrapped callable returned for these arguments, from
            the cache after the first call.
        """
        key = _make_key(args, kwargs)
        try:
            return cache[key]
        except KeyError:
            result = cache[key] = function(*args, **kwargs)
            return result

    wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
    wrapper.cache_size = lambda: len(cache)  # type: ignore[attr-defined]
    return wrapper
