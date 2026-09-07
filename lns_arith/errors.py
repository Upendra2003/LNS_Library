"""Warning classes and error metrics for :mod:`lns_arith`.

Range violations are *never* silent and *never* wrap.  Every encode or
arithmetic result that leaves the representable range is clamped to the nearest
representable code, and the event is surfaced two ways:

1. As a boolean flag on the ``*_with_flags`` variants (see
   :mod:`lns_arith.convert` and :mod:`lns_arith.ops`) -- the preferred path for
   batch experiments, because it costs nothing and is easy to aggregate.
2. As a catchable :class:`LNSOverflowWarning` / :class:`LNSUnderflowWarning`
   emitted through :mod:`warnings` -- the preferred path for interactive use.

Because these derive from :class:`UserWarning` they can be promoted to
exceptions with ``warnings.simplefilter("error", LNSWarning)``, or muted with
``warnings.simplefilter("ignore", LNSWarning)``.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

__all__ = [
    "LNSWarning",
    "LNSOverflowWarning",
    "LNSUnderflowWarning",
    "OverflowWarning",
    "UnderflowWarning",
    "warn_overflow",
    "warn_underflow",
    "abs_error",
    "relative_error",
    "error_metric",
]


class LNSWarning(UserWarning):
    """Base class for all range warnings raised by :mod:`lns_arith`."""


class LNSOverflowWarning(LNSWarning):
    """A magnitude was too large for the format and was clamped to ``max_code``."""


class LNSUnderflowWarning(LNSWarning):
    """A magnitude was too small for the format and was clamped to ``min_code``.

    Note that this is *not* flush-to-zero: the reserved zero code is only ever
    produced by an exact zero input or by exact cancellation in a subtraction.
    """


# Aliases matching the plain names used in the assignment specification.
OverflowWarning = LNSOverflowWarning
UnderflowWarning = LNSUnderflowWarning


def warn_overflow(context: str, fmt_name: str, stacklevel: int = 3) -> None:
    """Emit an :class:`LNSOverflowWarning` describing a clamped result."""
    warnings.warn(
        f"{context}: result exceeds {fmt_name} range, clamped to the maximum code",
        LNSOverflowWarning,
        stacklevel=stacklevel,
    )


def warn_underflow(context: str, fmt_name: str, stacklevel: int = 3) -> None:
    """Emit an :class:`LNSUnderflowWarning` describing a clamped result."""
    warnings.warn(
        f"{context}: result is below the {fmt_name} range, clamped to the minimum code",
        LNSUnderflowWarning,
        stacklevel=stacklevel,
    )


# ----------------------------------------------------------------------
# Error metrics.  These operate on ordinary Python floats and are only ever
# used *outside* the arithmetic core -- for testing and reporting.
# ----------------------------------------------------------------------


def abs_error(reference: float, test: float) -> float:
    """Absolute error ``|reference - test|``.

    Args:
        reference: The ground-truth value (typically the FP32 result).
        test: The value under evaluation (typically the decoded LNS result).

    Returns:
        The absolute difference, or ``inf`` if either input is non-finite and
        the two differ.
    """
    if reference == test:
        return 0.0
    if not (math.isfinite(reference) and math.isfinite(test)):
        return math.inf
    return abs(reference - test)


def relative_error(reference: float, test: float) -> float:
    """Relative error ``|reference - test| / |reference|``.

    Falls back to :func:`abs_error` when ``reference == 0``, as required by the
    experiment protocol (a relative error against zero is undefined).

    Args:
        reference: The ground-truth value.
        test: The value under evaluation.

    Returns:
        The relative error, or the absolute error when ``reference == 0``.
    """
    if reference == 0.0:
        return abs_error(reference, test)
    if reference == test:
        return 0.0
    if not (math.isfinite(reference) and math.isfinite(test)):
        return math.inf
    return abs(reference - test) / abs(reference)


def error_metric(reference: float, test: float) -> tuple[float, Literal["rel", "abs"]]:
    """Return ``(error, kind)`` where ``kind`` records which metric was used.

    This is the exact rule used by the benchmark suite: relative error where it
    is defined, absolute error when the reference is exactly zero.
    """
    if reference == 0.0:
        return abs_error(reference, test), "abs"
    return relative_error(reference, test), "rel"
