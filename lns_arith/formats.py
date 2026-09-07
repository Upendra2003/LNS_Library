"""Number-format definitions for the Logarithmic Number System (LNS).

An LNS value stores the *base-2 logarithm* of a number's magnitude in a
fixed-point field, plus a separate sign bit::

    word = [ sign : 1 bit ][ code : code_bits ]

The stored fixed-point ``code`` is an unsigned integer that is decoded as::

    L = (code - bias) / 2**frac_bits          # log2 of the magnitude
    x = (-1)**sign * 2**L

Two formats are provided:

===========  =====  ========  =========  ======  =========  =====================
Format       Total  Sign      Int bits   Frac    bias       log-step
===========  =====  ========  =========  ======  =========  =====================
``LNS16``    16     1         8          7       16384      1/128 ≈ 0.0078125
``LNS8``     8      1         4          3       64         1/8   = 0.125
===========  =====  ========  =========  ======  =========  =====================

Zero handling
-------------
Logarithms cannot represent zero (``log2(0) = -inf``), so one code point must be
sacrificed as a sentinel.  We reserve **code 0** -- the most-negative log slot --
to mean *exact zero*.  The canonical zero word is ``sign=0, code=0``; the word
``sign=1, code=0`` decodes to zero as well and is canonicalised to the positive
form by :func:`canonicalize`.

*Trade-off*: reserving the bottom slot shrinks the smallest representable
nonzero magnitude by exactly one log-step.  For ``LNS16`` the smallest nonzero
magnitude is ``2**-127.9921875`` instead of ``2**-128``; for ``LNS8`` it is
``2**-7.875`` instead of ``2**-8``.  In exchange we gain an *exact* zero, which
matters a great deal for DNN workloads (ReLU outputs, sparsity, padding, masked
attention) where zeros are extremely common and must not drift.

Overflow / underflow
--------------------
Encoding never wraps.  A magnitude whose ``log2`` falls outside
``[min_log, max_log]`` is clamped to the nearest representable code and the
event is reported -- either as a returned flag (see :mod:`lns_arith.convert`)
or as a catchable :class:`lns_arith.errors.LNSOverflowWarning` /
:class:`lns_arith.errors.LNSUnderflowWarning`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

__all__ = [
    "LNSConfig",
    "LNS16",
    "LNS8",
    "LNS16Config",
    "LNS8Config",
    "FORMATS",
    "get_format",
    "round_to_nearest",
    "pack",
    "sign_of",
    "code_of",
    "is_zero",
    "zero",
    "canonicalize",
    "negate",
    "log_of_code",
    "code_of_log",
    "clamp_code",
    "describe",
]


def round_to_nearest(x: float) -> int:
    """Round-to-nearest with ties resolved away from ``-inf``.

    This is the rounding rule applied to every fixed-point log code produced by
    the library (encoding and the log-add correction term).  It matches the
    cheap ``floor(x + 0.5)`` incrementer that hardware would use, and it is
    deliberately *not* Python's banker's rounding.

    Args:
        x: Real-valued fixed-point code.

    Returns:
        The nearest integer to ``x``.
    """
    return math.floor(x + 0.5)


@dataclass(frozen=True)
class LNSConfig:
    """Static description of one LNS format.

    Attributes:
        name: Human-readable format name, e.g. ``"LNS16"``.
        int_bits: Number of integer bits in the fixed-point log field.
        frac_bits: Number of fractional bits in the fixed-point log field.
        code_bits: ``int_bits + frac_bits`` -- width of the unsigned log code.
        total_bits: ``code_bits + 1`` -- full word width including the sign bit.
        bias: Offset subtracted from the code, ``2**(code_bits - 1)``.
        scale: Fixed-point scale factor, ``2**frac_bits``.
        log_step: Resolution of the log field, ``1 / scale``.
        zero_code: The code reserved for exact zero (always ``0``).
        min_code: Smallest code that denotes a nonzero magnitude (always ``1``).
        max_code: Largest representable code.
        min_log / max_log: Log2 range of representable nonzero magnitudes.
        min_value / max_value: Linear-domain magnitude range.
        sign_shift: Bit position of the sign bit inside the packed word.
        word_mask: Mask covering all ``total_bits`` of the packed word.
        lut_domain: Default ``|d|`` span of the log-add correction table.
    """

    name: str
    int_bits: int
    frac_bits: int
    lut_domain: float = 16.0

    # Derived, filled in by __post_init__ so they are plain attributes (fast).
    code_bits: int = field(init=False)
    total_bits: int = field(init=False)
    bias: int = field(init=False)
    scale: int = field(init=False)
    log_step: float = field(init=False)
    zero_code: int = field(init=False, default=0)
    min_code: int = field(init=False, default=1)
    max_code: int = field(init=False)
    min_log: float = field(init=False)
    max_log: float = field(init=False)
    min_value: float = field(init=False)
    max_value: float = field(init=False)
    sign_shift: int = field(init=False)
    word_mask: int = field(init=False)
    code_mask: int = field(init=False)

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        code_bits = self.int_bits + self.frac_bits
        scale = 1 << self.frac_bits
        bias = 1 << (code_bits - 1)
        max_code = (1 << code_bits) - 1
        set_(self, "code_bits", code_bits)
        set_(self, "total_bits", code_bits + 1)
        set_(self, "bias", bias)
        set_(self, "scale", scale)
        set_(self, "log_step", 1.0 / scale)
        set_(self, "max_code", max_code)
        set_(self, "min_log", (1 - bias) / scale)
        set_(self, "max_log", (max_code - bias) / scale)
        set_(self, "min_value", 2.0 ** ((1 - bias) / scale))
        set_(self, "max_value", 2.0 ** ((max_code - bias) / scale))
        set_(self, "sign_shift", code_bits)
        set_(self, "word_mask", (1 << (code_bits + 1)) - 1)
        set_(self, "code_mask", max_code)

    # -- convenience ----------------------------------------------------
    @property
    def max_relative_step_error(self) -> float:
        """Worst-case relative error of a *conversion* into this format.

        A half-step error in the log domain becomes a relative error of
        ``2**(log_step/2) - 1`` in the linear domain, independent of magnitude.
        """
        return 2.0 ** (self.log_step / 2.0) - 1.0

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.name}(1+{self.int_bits}+{self.frac_bits} bits, bias={self.bias}, "
            f"step={self.log_step:g}, magnitude in [{self.min_value:.3e}, "
            f"{self.max_value:.3e}])"
        )


#: 16-bit format: 1 sign + 8 integer + 7 fractional bits, bias 16384.
LNS16: Final[LNSConfig] = LNSConfig(name="LNS16", int_bits=8, frac_bits=7, lut_domain=16.0)

#: 8-bit format: 1 sign + 4 integer + 3 fractional bits, bias 64.
LNS8: Final[LNSConfig] = LNSConfig(name="LNS8", int_bits=4, frac_bits=3, lut_domain=8.0)

# Aliases matching the names used in the assignment specification.
LNS16Config: Final[LNSConfig] = LNS16
LNS8Config: Final[LNSConfig] = LNS8

#: Lookup of every supported format by (case-insensitive) name.
FORMATS: Final[dict[str, LNSConfig]] = {"lns16": LNS16, "lns8": LNS8}


def get_format(fmt: LNSConfig | str) -> LNSConfig:
    """Resolve a format given either a config object or its name.

    Args:
        fmt: An :class:`LNSConfig`, or a name such as ``"LNS8"``.

    Returns:
        The matching :class:`LNSConfig`.

    Raises:
        KeyError: If the name is not a known format.
    """
    if isinstance(fmt, LNSConfig):
        return fmt
    try:
        return FORMATS[fmt.lower()]
    except KeyError:
        raise KeyError(f"unknown LNS format {fmt!r}; expected one of {sorted(FORMATS)}") from None


# ----------------------------------------------------------------------
# Word packing / unpacking.  A "word" is a plain Python ``int`` holding the
# full sign+code bit pattern.  Every arithmetic routine in :mod:`lns_arith.ops`
# consumes and produces these words -- never floats.
# ----------------------------------------------------------------------


def pack(sign: int, code: int, fmt: LNSConfig) -> int:
    """Pack a sign bit and a log code into an LNS word.

    Args:
        sign: ``0`` for positive, ``1`` for negative.
        code: Unsigned fixed-point log code in ``[0, fmt.max_code]``.
        fmt: Target format.

    Returns:
        The packed word.  ``code == 0`` always yields the canonical zero word.

    Raises:
        ValueError: If ``code`` is outside the representable code range.
    """
    if not 0 <= code <= fmt.max_code:
        raise ValueError(f"code {code} out of range for {fmt.name} (0..{fmt.max_code})")
    if code == fmt.zero_code:
        return 0  # canonical zero: sign bit forced low
    return ((sign & 1) << fmt.sign_shift) | code


def sign_of(word: int, fmt: LNSConfig) -> int:
    """Return the sign bit (``0`` positive, ``1`` negative) of an LNS word."""
    return (word >> fmt.sign_shift) & 1


def code_of(word: int, fmt: LNSConfig) -> int:
    """Return the unsigned fixed-point log code of an LNS word."""
    return word & fmt.code_mask


def is_zero(word: int, fmt: LNSConfig) -> bool:
    """Return ``True`` if the word is the reserved exact-zero sentinel."""
    return (word & fmt.code_mask) == fmt.zero_code


def zero(fmt: LNSConfig) -> int:
    """Return the canonical exact-zero word for ``fmt``."""
    return 0


def canonicalize(word: int, fmt: LNSConfig) -> int:
    """Mask a word to its format width and normalise negative zero to zero."""
    word &= fmt.word_mask
    return 0 if (word & fmt.code_mask) == 0 else word


def negate(word: int, fmt: LNSConfig) -> int:
    """Flip the sign of an LNS word.

    Zero is its own negation (there is no signed zero in this encoding), so
    ``negate(zero) == zero``.
    """
    if is_zero(word, fmt):
        return 0
    return word ^ (1 << fmt.sign_shift)


def log_of_code(code: int, fmt: LNSConfig) -> float:
    """Decode a fixed-point code to the real-valued ``log2`` it represents.

    Note this is a *format helper*, not a value decoder: it never leaves the
    log domain and is safe to use inside arithmetic for tracing/diagnostics.
    """
    return (code - fmt.bias) / fmt.scale


def clamp_code(code: int, fmt: LNSConfig) -> tuple[int, bool, bool]:
    """Clamp an integer log code into the representable range.

    Args:
        code: Candidate code, possibly out of range.
        fmt: Target format.

    Returns:
        ``(clamped_code, overflow, underflow)``.  ``overflow`` is set when the
        code was above ``fmt.max_code``; ``underflow`` when it was below
        ``fmt.min_code`` (which includes the reserved zero slot).
    """
    if code > fmt.max_code:
        return fmt.max_code, True, False
    if code < fmt.min_code:
        return fmt.min_code, False, True
    return code, False, False


def code_of_log(log2_magnitude: float, fmt: LNSConfig) -> tuple[int, bool, bool]:
    """Quantise a real ``log2`` magnitude to a fixed-point code.

    Applies round-to-nearest on the code, then clamps into range.

    Args:
        log2_magnitude: The value ``log2(|x|)`` to encode.
        fmt: Target format.

    Returns:
        ``(code, overflow, underflow)`` -- see :func:`clamp_code`.
    """
    if math.isnan(log2_magnitude):
        raise ValueError("cannot encode NaN into an LNS format")
    if log2_magnitude == math.inf:
        return fmt.max_code, True, False
    if log2_magnitude == -math.inf:
        return fmt.min_code, False, True
    return clamp_code(round_to_nearest(log2_magnitude * fmt.scale + fmt.bias), fmt)


def describe(word: int, fmt: LNSConfig) -> str:
    """Return a human-readable one-line description of a packed word.

    Used by the simulator's trace display, e.g.::

        LNS16 word=0xC001 sign=1 code=16385 (0x4001) L=+0.0078125 -> -1.005430
    """
    s = sign_of(word, fmt)
    c = code_of(word, fmt)
    if c == fmt.zero_code:
        return f"{fmt.name} word=0x{word:0{(fmt.total_bits + 3) // 4}X} ZERO (reserved code 0)"
    L = log_of_code(c, fmt)
    value = (-1.0 if s else 1.0) * (2.0**L)
    hexw = (fmt.total_bits + 3) // 4
    return (
        f"{fmt.name} word=0x{word:0{hexw}X} sign={s} code={c} (0x{c:X}) "
        f"L={L:+.7f} -> {value:.6g}"
    )
