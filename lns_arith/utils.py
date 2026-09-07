"""Reproducible test-vector generators for the LNS error experiments.

Every generator takes an explicit ``seed`` and uses its own
:class:`random.Random` instance, so results are reproducible and independent of
global RNG state (and of any other library that touches it).

Categories
----------
``positive``
    Log-uniform magnitudes in ``[1e-3, 1e3]``, all positive.  The "typical
    activations and weights" case.
``negative``
    The same distribution, all negative -- exercises the sign path.
``zero``
    Exact zeros.  Paired against ordinary values by :func:`make_pairs` so that
    ``0 + x``, ``x + 0`` and ``0 + 0`` are all covered.
``small``
    Log-uniform magnitudes in ``[1e-38, 1e-3]``, mixed signs.  Inside ``LNS16``
    range (min ``2**-127.99 ~ 2.96e-39``) but far below ``LNS8`` range (min
    ``2**-7.875 ~ 4.3e-3``), so this category drives ``LNS8`` underflow
    clamping.
``large``
    Log-uniform magnitudes in ``[1e3, 1e38]``, mixed signs.  Inside ``LNS16``
    range (max ``2**127.99 ~ 3.4e38``) but far above ``LNS8`` range (max
    ``2**7.875 ~ 235``), so this category drives ``LNS8`` overflow clamping and
    also pushes ``FP16`` (max 65504) past its limit.
``random_log``
    Log-uniform over ``[1e-6, 1e6]``, mixed signs -- uniform coverage of the
    *exponent* axis, which is where LNS quantisation is uniform.
``random_linear``
    Uniform over ``[-4, 4]`` in the linear domain, mixed signs -- the
    distribution weight-initialisers and activations actually look like.
``random``
    A 50/50 blend of ``random_log`` and ``random_linear``.
``extreme``
    Magnitudes from ``1e-60`` to ``1e60``, mixed signs.  Deliberately outside
    *both* LNS16 and FP32 range; used only by the clamping report, where the
    question is "was the flag raised and was the value clamped?", not "what was
    the relative error against an infinite reference?".
"""

from __future__ import annotations

import math
import random
from typing import Callable, Final, Sequence

__all__ = [
    "CATEGORIES",
    "CATEGORY_NAMES",
    "generate",
    "make_pairs",
    "gen_positive",
    "gen_negative",
    "gen_zero",
    "gen_small",
    "gen_large",
    "gen_random_log",
    "gen_random_linear",
    "gen_random",
    "gen_extreme",
]


def _log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    """Draw a magnitude uniformly in ``log10`` space between ``lo`` and ``hi``."""
    return 10.0 ** rng.uniform(math.log10(lo), math.log10(hi))


def _signed(rng: random.Random, magnitude: float) -> float:
    """Attach a random sign to a magnitude."""
    return magnitude if rng.random() < 0.5 else -magnitude


def gen_positive(n: int, seed: int = 0) -> list[float]:
    """``n`` positive log-uniform values in ``[1e-3, 1e3]``."""
    rng = random.Random(seed)
    return [_log_uniform(rng, 1e-3, 1e3) for _ in range(n)]


def gen_negative(n: int, seed: int = 0) -> list[float]:
    """``n`` negative log-uniform values in ``[-1e3, -1e-3]``."""
    rng = random.Random(seed)
    return [-_log_uniform(rng, 1e-3, 1e3) for _ in range(n)]


def gen_zero(n: int, seed: int = 0) -> list[float]:
    """``n`` exact zeros."""
    return [0.0] * n


def gen_small(n: int, seed: int = 0) -> list[float]:
    """``n`` very small mixed-sign magnitudes in ``[1e-38, 1e-3]``."""
    rng = random.Random(seed)
    return [_signed(rng, _log_uniform(rng, 1e-38, 1e-3)) for _ in range(n)]


def gen_large(n: int, seed: int = 0) -> list[float]:
    """``n`` very large mixed-sign magnitudes in ``[1e3, 1e38]``."""
    rng = random.Random(seed)
    return [_signed(rng, _log_uniform(rng, 1e3, 1e38)) for _ in range(n)]


def gen_random_log(n: int, seed: int = 0) -> list[float]:
    """``n`` mixed-sign values, log-uniform in magnitude over ``[1e-6, 1e6]``."""
    rng = random.Random(seed)
    return [_signed(rng, _log_uniform(rng, 1e-6, 1e6)) for _ in range(n)]


def gen_random_linear(n: int, seed: int = 0) -> list[float]:
    """``n`` values uniform in the linear domain over ``[-4, 4]``.

    Zero is vanishingly unlikely here but tiny magnitudes are not, which is the
    point: linear-uniform sampling stresses the low end of the log axis.
    """
    rng = random.Random(seed)
    return [rng.uniform(-4.0, 4.0) for _ in range(n)]


def gen_random(n: int, seed: int = 0) -> list[float]:
    """``n`` values, half log-uniform and half linear-uniform, interleaved."""
    rng = random.Random(seed)
    out: list[float] = []
    for _ in range(n):
        if rng.random() < 0.5:
            out.append(_signed(rng, _log_uniform(rng, 1e-6, 1e6)))
        else:
            out.append(rng.uniform(-4.0, 4.0))
    return out


def gen_extreme(n: int, seed: int = 0) -> list[float]:
    """``n`` mixed-sign magnitudes spanning ``[1e-60, 1e60]``.

    Outside FP32 range at both ends, so these are used for clamping/flag checks
    rather than for error statistics.
    """
    rng = random.Random(seed)
    out: list[float] = []
    for i in range(n):
        magnitude = _log_uniform(rng, 1e-60, 1e-39) if i % 2 else _log_uniform(rng, 1e39, 1e60)
        out.append(_signed(rng, magnitude))
    return out


#: Every named category, mapping name -> ``(n, seed) -> values``.
CATEGORIES: Final[dict[str, Callable[[int, int], list[float]]]] = {
    "positive": gen_positive,
    "negative": gen_negative,
    "zero": gen_zero,
    "small": gen_small,
    "large": gen_large,
    "random_log": gen_random_log,
    "random_linear": gen_random_linear,
    "random": gen_random,
    "extreme": gen_extreme,
}

#: The categories the standard error report iterates over (``extreme`` is
#: handled separately by the clamping report).
CATEGORY_NAMES: Final[tuple[str, ...]] = (
    "positive",
    "negative",
    "zero",
    "small",
    "large",
    "random_log",
    "random_linear",
    "random",
)


def generate(category: str, n: int, seed: int = 0) -> list[float]:
    """Generate ``n`` values from a named category.

    Args:
        category: One of the keys of :data:`CATEGORIES`.
        n: How many values to produce.
        seed: RNG seed; identical arguments always give identical output.

    Returns:
        A list of ``n`` Python floats.

    Raises:
        KeyError: If the category name is unknown.
    """
    try:
        generator = CATEGORIES[category]
    except KeyError:
        raise KeyError(
            f"unknown category {category!r}; expected one of {sorted(CATEGORIES)}"
        ) from None
    return generator(n, seed)


def make_pairs(
    category: str, n: int, seed: int = 0
) -> list[tuple[float, float, float]]:
    """Build ``n`` operand triples ``(a, b, acc)`` for a category.

    ``a`` always comes from the requested category.  The choice of ``b`` is
    category-dependent so that each test stays informative:

    * ``small`` / ``large`` -- ``b`` is drawn from ``random_linear`` (a typical
      moderate value).  Pairing two extremes would push the FP32 *reference*
      itself to ``inf`` or ``0`` for most samples and the comparison would
      measure nothing.
    * ``zero`` -- ``a`` and ``b`` alternate between exact zero and an ordinary
      value, so the sample set covers ``0 + x``, ``x + 0`` and ``0 + 0``.
    * everything else -- ``b`` is a second independent draw from the same
      category.

    ``acc`` (the MAC accumulator) is always a moderate ``random_linear`` value.

    Args:
        category: One of the keys of :data:`CATEGORIES`.
        n: Number of triples.
        seed: RNG seed.

    Returns:
        A list of ``n`` ``(a, b, acc)`` tuples.
    """
    a_values = generate(category, n, seed)
    acc_values = gen_random_linear(n, seed + 977)

    if category in ("small", "large"):
        b_values = gen_random_linear(n, seed + 101)
    elif category == "zero":
        others = gen_random_linear(n, seed + 101)
        a_values = [0.0 if i % 3 != 2 else others[i] for i in range(n)]
        b_values = [0.0 if i % 3 != 1 else others[n - 1 - i] for i in range(n)]
    else:
        b_values = generate(category, n, seed + 101)

    return list(zip(a_values, b_values, acc_values))


def as_sequence(values: Sequence[float]) -> list[float]:
    """Return ``values`` as a plain list (convenience for the web front-ends)."""
    return list(values)
