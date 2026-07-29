"""Small caching helper used to memoise expensive matrix construction."""

from __future__ import annotations

import functools
from collections.abc import Callable, Hashable
from typing import Any, Generic, ParamSpec, TypeVar

__all__ = ["Memo", "cached"]

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


class Memo(Generic[P, R]):
    """A callable wrapping one function together with its memo.

    A class rather than a closure with attributes bolted on: ``cache_clear``
    and ``cache_size`` are then real, typed members, so callers get them
    checked instead of the type checker being told to look away.
    """

    def __init__(self, function: Callable[P, R]) -> None:
        """Wrap ``function``, starting with an empty cache.

        Args:
            function: The callable to memoise.
        """
        self._function = function
        self._cache: dict[Hashable, R] = {}
        functools.update_wrapper(self, function)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
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
            return self._cache[key]
        except KeyError:
            result = self._cache[key] = self._function(*args, **kwargs)
            return result

    def cache_clear(self) -> None:
        """Discard every memoised result."""
        self._cache.clear()

    def cache_size(self) -> int:
        """Report how many results are currently memoised.

        Returns:
            The number of distinct argument combinations held.
        """
        return len(self._cache)


def cached(function: Callable[P, R]) -> Memo[P, R]:
    """Memoise ``function`` on its arguments, tolerating unhashable ones.

    Unlike :func:`functools.lru_cache` this accepts unhashable arguments by
    falling back to their ``repr``, which is what the numpy-heavy call sites
    here need. The cache is unbounded and lives for the life of the process.

    .. warning::
       Do not apply this to a method. ``self`` becomes part of the key and is
       held by a strong reference, so every instance ever used is kept alive
       and the cache grows without bound. Make the function module level and
       key it on the values it actually depends on, as
       :func:`~walsh.transforms.hadamard_matrix` does. :class:`Memo` is not a
       descriptor, so misusing it this way fails loudly rather than leaking
       quietly.

    Args:
        function: The callable to memoise.

    Returns:
        A :class:`Memo` with the same call signature as ``function``, plus
        :meth:`Memo.cache_clear` and :meth:`Memo.cache_size`.
    """
    return Memo(function)
