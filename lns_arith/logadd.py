"""The log-domain addition correction term -- the heart of LNS addition.

Multiplication is trivial in the log domain (add the logs).  Addition is the
hard part.  Given two magnitudes with logs ``Lh >= Ll`` and
``d = Ll - Lh <= 0``:

* **Same signs** (a true add)::

      log2(2**Lh + 2**Ll) = Lh + log2(1 + 2**d)

* **Opposite signs** (an effective subtract)::

      log2(2**Lh - 2**Ll) = Lh + log2(1 - 2**d)

So all of LNS addition reduces to evaluating one of two single-argument
correction functions, ``s_add(d) = log2(1 + 2**d)`` and
``s_sub(d) = log2(1 - 2**d)``, on ``d <= 0``.

Two evaluation modes are provided:

``"exact"``
    Evaluate with :mod:`math` in double precision.  This is a *software
    idealisation*: no real accelerator computes a logarithm per addition.

``"lut"``
    Read the nearest entry from a precomputed table whose *values are
    quantised to the target format's own fixed-point log grid*.  This is what
    hardware actually does, and it introduces a second, independent error
    source on top of format quantisation.

    The default table is sampled at the format's own log-step (7 fractional
    bits for ``LNS16`` -> 2049 entries over ``d in [-16, 0]``; 3 fractional
    bits for ``LNS8`` -> 65 entries over ``d in [-8, 0]``).  Beyond the table's
    domain the correction is below half a log-step and is taken as zero.

    A note on what this shows: at *full* table resolution the LUT is exact with
    respect to the representable grid, because ``d`` in LNS addition is always
    an exact multiple of the log-step (it is a difference of two integer
    codes), so the lookup lands on a sampled point and the stored value has
    already been rounded to the same grid the result will be rounded to.  Pass
    a smaller ``lut_size`` to model a decimated hardware table -- that is where
    LUT quantisation error becomes visible, and the benchmark suite sweeps
    exactly that.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator, Literal, Sequence

from .formats import LNSConfig, get_format, round_to_nearest

__all__ = [
    "Mode",
    "Kind",
    "exact_mode",
    "lut_mode",
    "correction",
    "default_lut_size",
    "get_lut",
    "CorrectionLUT",
    "set_mode",
    "get_mode",
    "use_mode",
]

Mode = Literal["exact", "lut"]
Kind = Literal["add", "sub"]

_LN2 = math.log(2.0)

# Global default, overridable per call and via :func:`set_mode` / :func:`use_mode`.
_GLOBAL_MODE: Mode = "exact"


# ----------------------------------------------------------------------
# Mode switch.
# ----------------------------------------------------------------------


def set_mode(mode: Mode) -> Mode:
    """Set the global default log-add mode.

    Args:
        mode: ``"exact"`` or ``"lut"``.

    Returns:
        The previous mode.

    Raises:
        ValueError: If ``mode`` is not a recognised mode name.
    """
    global _GLOBAL_MODE
    if mode not in ("exact", "lut"):
        raise ValueError(f"unknown logadd mode {mode!r}; expected 'exact' or 'lut'")
    previous, _GLOBAL_MODE = _GLOBAL_MODE, mode
    return previous


def get_mode() -> Mode:
    """Return the current global log-add mode."""
    return _GLOBAL_MODE


@contextmanager
def use_mode(mode: Mode) -> Iterator[Mode]:
    """Context manager that temporarily switches the global log-add mode."""
    previous = set_mode(mode)
    try:
        yield mode
    finally:
        set_mode(previous)


# ----------------------------------------------------------------------
# Mode 1: exact evaluation.
# ----------------------------------------------------------------------


def exact_mode(d: float, kind: Kind = "add") -> float:
    """Evaluate the correction term exactly, in double precision.

    Args:
        d: The log-domain gap ``Ll - Lh``; must be ``<= 0``.
        kind: ``"add"`` for ``log2(1 + 2**d)`` (same-sign operands) or
            ``"sub"`` for ``log2(1 - 2**d)`` (opposite-sign operands).

    Returns:
        The correction in log2 units.  For ``kind="sub"`` and ``d == 0`` this
        is ``-inf``, signalling exact cancellation.

    Raises:
        ValueError: If ``d > 0`` or ``kind`` is unknown.
    """
    if d > 0.0:
        raise ValueError(f"correction term is defined for d <= 0, got {d!r}")
    if kind == "add":
        # log1p keeps full precision for very negative d, where 2**d is tiny.
        return math.log1p(2.0**d) / _LN2
    if kind == "sub":
        if d == 0.0:
            return -math.inf
        # 1 - 2**d == -expm1(d * ln2); expm1 keeps precision as d -> 0-.
        return math.log2(-math.expm1(d * _LN2))
    raise ValueError(f"unknown correction kind {kind!r}; expected 'add' or 'sub'")


# ----------------------------------------------------------------------
# Mode 2: quantised lookup table.
# ----------------------------------------------------------------------


def default_lut_size(fmt: LNSConfig | str) -> int:
    """Number of table entries when sampling at the format's own log-step.

    ``LNS16`` -> ``16 * 128 + 1 = 2049``; ``LNS8`` -> ``8 * 8 + 1 = 65``.
    """
    fmt = get_format(fmt)
    return int(round(fmt.lut_domain * fmt.scale)) + 1


class CorrectionLUT:
    """A precomputed, value-quantised table of one correction function.

    The table samples ``d`` uniformly over ``[-domain, 0]`` with ``size``
    entries (index ``0`` is ``d = 0``, index ``size - 1`` is ``d = -domain``)
    and stores each value rounded to the target format's fixed-point log grid,
    exactly as a hardware ROM would hold it.

    Attributes:
        fmt: The format whose grid the entries are quantised to.
        kind: ``"add"`` or ``"sub"``.
        domain: The ``|d|`` span covered by the table.
        size: Number of entries.
        step: Spacing between samples, ``domain / (size - 1)``.
        entries: The quantised correction values, in log2 units.
    """

    __slots__ = ("fmt", "kind", "domain", "size", "step", "entries")

    def __init__(self, fmt: LNSConfig, kind: Kind, domain: float, size: int) -> None:
        if size < 2:
            raise ValueError(f"lut_size must be at least 2, got {size}")
        if domain <= 0:
            raise ValueError(f"lut domain must be positive, got {domain}")
        self.fmt = fmt
        self.kind = kind
        self.domain = float(domain)
        self.size = int(size)
        self.step = self.domain / (self.size - 1)
        scale = fmt.scale
        entries: list[float] = []
        for i in range(self.size):
            value = exact_mode(-i * self.step, kind)
            if math.isinf(value):
                entries.append(value)  # exact cancellation stays exact
            else:
                entries.append(round_to_nearest(value * scale) / scale)
        self.entries: Sequence[float] = tuple(entries)

    def lookup(self, d: float) -> float:
        """Return the nearest stored correction for gap ``d`` (``d <= 0``).

        Values of ``d`` beyond the table domain return ``0.0``: there the true
        correction is smaller than half a log-step and cannot change the code.

        For a ``"sub"`` table, address 0 holds ``-inf`` -- the exact
        cancellation case -- and is only reachable by ``d == 0`` exactly.  A
        decimated table whose first step spans more than one log-step would
        otherwise round a small *nonzero* gap onto it and wrongly annihilate
        the result, so a nonzero gap is floored to address 1.  Hardware makes
        the same distinction: exact cancellation is detected by comparing
        codes, not by reading the ROM.
        """
        if d > 0.0:
            raise ValueError(f"correction term is defined for d <= 0, got {d!r}")
        gap = -d
        if gap >= self.domain:
            return 0.0
        index = round_to_nearest(gap / self.step)
        if index >= self.size:
            index = self.size - 1
        elif index == 0 and self.kind == "sub" and gap > 0.0:
            index = 1
        return self.entries[index]

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CorrectionLUT({self.fmt.name}, kind={self.kind!r}, "
            f"domain={self.domain:g}, size={self.size}, step={self.step:g})"
        )


@lru_cache(maxsize=64)
def _build_lut(fmt_name: str, kind: Kind, domain: float, size: int) -> CorrectionLUT:
    return CorrectionLUT(get_format(fmt_name), kind, domain, size)


def get_lut(
    fmt: LNSConfig | str,
    kind: Kind = "add",
    lut_size: int | None = None,
    domain: float | None = None,
) -> CorrectionLUT:
    """Return a cached :class:`CorrectionLUT`, building it on first use.

    Args:
        fmt: Target format (config or name).
        kind: ``"add"`` or ``"sub"``.
        lut_size: Number of entries; defaults to :func:`default_lut_size`,
            i.e. one sample per log-step.  Smaller values model a decimated
            hardware table and introduce visible lookup error.
        domain: ``|d|`` span; defaults to ``fmt.lut_domain`` (16 for LNS16,
            8 for LNS8).

    Returns:
        The (cached) table.
    """
    fmt = get_format(fmt)
    if kind not in ("add", "sub"):
        raise ValueError(f"unknown correction kind {kind!r}; expected 'add' or 'sub'")
    size = default_lut_size(fmt) if lut_size is None else int(lut_size)
    span = fmt.lut_domain if domain is None else float(domain)
    return _build_lut(fmt.name, kind, span, size)


def lut_mode(
    d: float,
    kind: Kind,
    fmt: LNSConfig | str,
    lut_size: int | None = None,
    domain: float | None = None,
) -> float:
    """Evaluate the correction term by nearest-entry table lookup.

    Args:
        d: The log-domain gap ``Ll - Lh``; must be ``<= 0``.
        kind: ``"add"`` or ``"sub"``.
        fmt: Target format, which fixes the table's value quantisation.
        lut_size: Optional table resolution override.
        domain: Optional table domain override.

    Returns:
        The stored (format-quantised) correction in log2 units.
    """
    return get_lut(fmt, kind, lut_size, domain).lookup(d)


# ----------------------------------------------------------------------
# Unified entry point used by :mod:`lns_arith.ops`.
# ----------------------------------------------------------------------


def correction(
    d: float,
    kind: Kind,
    fmt: LNSConfig | str,
    mode: Mode | None = None,
    lut_size: int | None = None,
) -> float:
    """Evaluate the log-add correction term in the selected mode.

    Args:
        d: The log-domain gap ``Ll - Lh <= 0``.
        kind: ``"add"`` (same signs) or ``"sub"`` (opposite signs).
        fmt: Target format.
        mode: ``"exact"``, ``"lut"``, or ``None`` to use the global mode.
        lut_size: Table resolution when ``mode == "lut"``.

    Returns:
        The correction in log2 units.
    """
    resolved: Mode = _GLOBAL_MODE if mode is None else mode
    if resolved == "exact":
        return exact_mode(d, kind)
    if resolved == "lut":
        return lut_mode(d, kind, fmt, lut_size)
    raise ValueError(f"unknown logadd mode {resolved!r}; expected 'exact' or 'lut'")
