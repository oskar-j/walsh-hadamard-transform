from __future__ import annotations

import pytest

from walsh.decorators import Memo, cached


def test_results_are_memoised() -> None:
    calls = 0

    @cached
    def double(value: int) -> int:
        nonlocal calls
        calls += 1
        return value * 2

    assert double(21) == 42
    assert double(21) == 42
    assert calls == 1


def test_distinct_arguments_are_cached_separately() -> None:
    @cached
    def identity(value: int) -> int:
        return value

    identity(1)
    identity(2)
    assert identity.cache_size() == 2


def test_cache_clear_empties_it() -> None:
    @cached
    def identity(value: int) -> int:
        return value

    identity(1)
    assert identity.cache_size() == 1
    identity.cache_clear()
    assert identity.cache_size() == 0


def test_unhashable_arguments_fall_back_to_repr() -> None:
    """The reason this exists rather than functools.lru_cache."""
    calls = 0

    @cached
    def total(values: list[int]) -> int:
        nonlocal calls
        calls += 1
        return sum(values)

    assert total([1, 2, 3]) == 6
    assert total([1, 2, 3]) == 6
    assert calls == 1, "an unhashable argument should still hit the cache"


def test_keyword_arguments_are_part_of_the_key() -> None:
    @cached
    def offset(value: int, *, by: int = 0) -> int:
        return value + by

    assert offset(1, by=1) == 2
    assert offset(1, by=2) == 3
    assert offset(1) == 1
    assert offset.cache_size() == 3


def test_keyword_order_does_not_change_the_key() -> None:
    calls = 0

    @cached
    def combine(*, first: int, second: int) -> int:
        nonlocal calls
        calls += 1
        return first + second

    combine(first=1, second=2)
    combine(second=2, first=1)
    assert calls == 1


def test_wrapped_metadata_is_preserved() -> None:
    @cached
    def documented(value: int) -> int:
        """A docstring worth keeping."""
        return value

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "A docstring worth keeping."


def test_memo_is_not_a_descriptor() -> None:
    """Applying it to a method must fail loudly rather than leak instances."""

    class Holder:
        @cached
        def method(self, value: int) -> int:  # pragma: no cover - never called
            return value

    assert isinstance(Holder.__dict__["method"], Memo)
    with pytest.raises(TypeError):
        Holder().method(1)
