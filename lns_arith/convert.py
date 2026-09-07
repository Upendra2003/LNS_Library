"""Conversion between IEEE floating point and LNS codes.

The conversion boundary is the *only* place in the library where floats and
LNS codes meet.  :mod:`lns_arith.ops` is pure code-domain arithmetic and never
calls anything from this module.

Source-format quantisation
--------------------------
``fp32_to_lns`` first rounds the incoming Python float (a C ``double``) to
binary32, and ``fp16_to_lns`` rounds it to binary16, so that the encoder sees
exactly the bits a real FP32/FP16 tensor would have held.  This uses
:mod:`struct` (``'f'`` and ``'e'`` codes), so the package has **no third-party
dependencies**.

Encoding
--------
``code = round_to_nearest(log2(|x|) * 2**frac_bits + bias)``, then clamped into
``[min_code, max_code]``.  Exact zero maps to the reserved zero sentinel.
Non-finite inputs clamp to the extreme code and flag overflow.
"""

from __future__ import annotations

import math
import struct
from typing import Literal, NamedTuple

from .errors import warn_overflow, warn_underflow
from .formats import (
    LNS8,
    LNS16,
    LNSConfig,
    code_of,
    code_of_log,
    get_format,
    is_zero,
    log_of_code,
    pack,
    sign_of,
)

__all__ = [
    "EncodeResult",
    "to_fp32",
    "to_fp16",
    "quantize_source",
    "fp32_to_lns",
    "fp16_to_lns",
    "lns_to_float",
    "fp32_to_lns16",
    "fp32_to_lns8",
    "fp16_to_lns16",
    "fp16_to_lns8",
    "lns16_to_float",
    "lns8_to_float",
    "encode_with_flags",
]

SourceFormat = Literal["fp32", "fp16", "fp64"]


class EncodeResult(NamedTuple):
    """Everything the encoder learned, for flag-driven callers and tracing.

    Attributes:
        word: The packed LNS word.
        overflow: ``True`` if the magnitude was clamped to ``max_code``.
        underflow: ``True`` if the magnitude was clamped to ``min_code``.
        source_value: The input after rounding to the source float format.
        log2_magnitude: ``log2(|source_value|)`` before quantisation
            (``-inf`` for zero).
    """

    word: int
    overflow: bool
    underflow: bool
    source_value: float
    log2_magnitude: float


# ----------------------------------------------------------------------
# IEEE source-format rounding (dependency free).
# ----------------------------------------------------------------------


def to_fp32(x: float) -> float:
    """Round a Python float to the nearest binary32 value.

    Out-of-range magnitudes saturate to ``±inf``, exactly as an FP32 store
    would.
    """
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def to_fp16(x: float) -> float:
    """Round a Python float to the nearest binary16 (IEEE half) value.

    Out-of-range magnitudes saturate to ``±inf``, exactly as an FP16 store
    would.
    """
    try:
        return struct.unpack("<e", struct.pack("<e", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def quantize_source(x: float, source: SourceFormat = "fp32") -> float:
    """Round ``x`` to the given IEEE source format.

    Args:
        x: Input value.
        source: One of ``"fp32"``, ``"fp16"`` or ``"fp64"`` (no rounding).

    Returns:
        The rounded value.

    Raises:
        ValueError: If ``source`` is not a recognised format name.
    """
    if source == "fp32":
        return to_fp32(x)
    if source == "fp16":
        return to_fp16(x)
    if source == "fp64":
        return float(x)
    raise ValueError(f"unknown source format {source!r}; expected fp32, fp16 or fp64")


# ----------------------------------------------------------------------
# Float -> LNS.
# ----------------------------------------------------------------------


def encode_with_flags(
    x: float,
    fmt: LNSConfig | str,
    *,
    source: SourceFormat = "fp32",
) -> EncodeResult:
    """Encode a float into an LNS word, returning range flags instead of warning.

    This is the flag-based entry point used by the benchmark harness and by the
    step-by-step simulator trace.

    Args:
        x: Value to encode.
        fmt: Target :class:`~lns_arith.formats.LNSConfig` or format name.
        source: IEEE format the value is deemed to come from; the value is
            rounded to it before the log is taken.

    Returns:
        An :class:`EncodeResult`.

    Raises:
        ValueError: If ``x`` is NaN (NaN has no LNS representation).
    """
    fmt = get_format(fmt)
    if isinstance(x, bool):  # bool is an int subclass; be explicit about intent
        x = float(x)
    xf = quantize_source(float(x), source)

    if math.isnan(xf):
        raise ValueError("cannot encode NaN into an LNS format")

    if xf == 0.0:
        return EncodeResult(pack(0, fmt.zero_code, fmt), False, False, 0.0, -math.inf)

    sign = 1 if math.copysign(1.0, xf) < 0 else 0
    magnitude = abs(xf)

    if math.isinf(magnitude):
        log2_mag = math.inf
    else:
        log2_mag = math.log2(magnitude)

    code, overflow, underflow = code_of_log(log2_mag, fmt)
    return EncodeResult(pack(sign, code, fmt), overflow, underflow, xf, log2_mag)


def fp32_to_lns(x: float, fmt: LNSConfig | str, *, warn: bool = True) -> int:
    """Convert an FP32 value to an LNS word.

    Args:
        x: Value to convert; rounded to binary32 first.
        fmt: Target format (config object or name).
        warn: If ``True``, emit :class:`~lns_arith.errors.LNSOverflowWarning` /
            :class:`~lns_arith.errors.LNSUnderflowWarning` on range violations.
            The result is clamped either way.

    Returns:
        The packed LNS word.
    """
    fmt = get_format(fmt)
    result = encode_with_flags(x, fmt, source="fp32")
    if warn:
        if result.overflow:
            warn_overflow(f"fp32_to_lns({x!r})", fmt.name)
        elif result.underflow:
            warn_underflow(f"fp32_to_lns({x!r})", fmt.name)
    return result.word


def fp16_to_lns(x: float, fmt: LNSConfig | str, *, warn: bool = True) -> int:
    """Convert an FP16 value to an LNS word.

    Args:
        x: Value to convert; rounded to binary16 first, so anything above
            65504 in magnitude arrives at the encoder as ``inf`` and overflows.
        fmt: Target format (config object or name).
        warn: Emit range warnings as well as clamping.

    Returns:
        The packed LNS word.
    """
    fmt = get_format(fmt)
    result = encode_with_flags(x, fmt, source="fp16")
    if warn:
        if result.overflow:
            warn_overflow(f"fp16_to_lns({x!r})", fmt.name)
        elif result.underflow:
            warn_underflow(f"fp16_to_lns({x!r})", fmt.name)
    return result.word


# ----------------------------------------------------------------------
# LNS -> float.  Used only for reporting / display -- never inside ops.
# ----------------------------------------------------------------------


def lns_to_float(word: int, fmt: LNSConfig | str) -> float:
    """Decode an LNS word to a Python float.

    .. warning::
       This function exists for testing, reporting and display only.  No
       routine in :mod:`lns_arith.ops` calls it: LNS arithmetic stays in the
       code domain from end to end.  ``tests/test_ops.py`` enforces this by
       scanning the module source.

    Args:
        word: Packed LNS word.
        fmt: Its format.

    Returns:
        The represented value as a Python float (``0.0`` for the zero
        sentinel).
    """
    fmt = get_format(fmt)
    if is_zero(word, fmt):
        return 0.0
    magnitude = 2.0 ** log_of_code(code_of(word, fmt), fmt)
    return -magnitude if sign_of(word, fmt) else magnitude


# ----------------------------------------------------------------------
# Named convenience wrappers required by the specification.
# ----------------------------------------------------------------------


def fp32_to_lns16(x: float, *, warn: bool = True) -> int:
    """Convert an FP32 value to an :data:`~lns_arith.formats.LNS16` word."""
    return fp32_to_lns(x, LNS16, warn=warn)


def fp32_to_lns8(x: float, *, warn: bool = True) -> int:
    """Convert an FP32 value to an :data:`~lns_arith.formats.LNS8` word."""
    return fp32_to_lns(x, LNS8, warn=warn)


def fp16_to_lns16(x: float, *, warn: bool = True) -> int:
    """Convert an FP16 value to an :data:`~lns_arith.formats.LNS16` word."""
    return fp16_to_lns(x, LNS16, warn=warn)


def fp16_to_lns8(x: float, *, warn: bool = True) -> int:
    """Convert an FP16 value to an :data:`~lns_arith.formats.LNS8` word."""
    return fp16_to_lns(x, LNS8, warn=warn)


def lns16_to_float(word: int) -> float:
    """Decode an :data:`~lns_arith.formats.LNS16` word to a Python float."""
    return lns_to_float(word, LNS16)


def lns8_to_float(word: int) -> float:
    """Decode an :data:`~lns_arith.formats.LNS8` word to a Python float."""
    return lns_to_float(word, LNS8)
