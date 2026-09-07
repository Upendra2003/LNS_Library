"""LNS arithmetic -- add, subtract, multiply, multiply-accumulate.

**Every function in this module consumes and produces packed LNS words.**  No
routine here decodes a value to a float to compute a result: multiplication is
integer addition of log codes, and addition is a code comparison plus one
correction-term lookup (see :mod:`lns_arith.logadd`).  The only floats that
appear are the correction term itself and the log-domain gap it is indexed by,
which live entirely in the log domain and never touch the linear domain.

``tests/test_ops.py`` enforces the constraint mechanically by scanning this
module's source for any reference to ``lns_to_float``.

Sign and zero handling
----------------------
* ``mul``: sign is the XOR of the operand signs; either operand zero gives
  exact zero; the magnitude code is ``ca + cb - bias`` -- an *exact* integer
  operation, so multiplication adds no rounding error of its own.
* ``add``, same signs: ``result = larger + log2(1 + 2**d)``.
* ``add``, opposite signs: ``result = larger + log2(1 - 2**d)``.  When the two
  codes are *equal* the operands cancel exactly and the result is the zero
  sentinel -- the ``-inf`` singularity of the correction function is handled by
  a code comparison, never evaluated.
* ``add`` with a zero operand returns the other operand unchanged.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, MutableMapping, NamedTuple, Sequence

from .errors import warn_overflow, warn_underflow
from .formats import (
    LNSConfig,
    clamp_code,
    code_of,
    describe,
    get_format,
    is_zero,
    log_of_code,
    negate,
    pack,
    round_to_nearest,
    sign_of,
)
from .logadd import Mode, correction

__all__ = [
    "OpResult",
    "lns_neg",
    "lns_abs",
    "lns_mul",
    "lns_add",
    "lns_sub",
    "lns_mac",
    "lns_dot",
    "lns_mul_with_flags",
    "lns_add_with_flags",
    "lns_sub_with_flags",
    "lns_mac_with_flags",
]

Trace = MutableMapping[str, Any]


class OpResult(NamedTuple):
    """Result word plus the range flags raised while producing it.

    Attributes:
        word: The packed LNS result.
        overflow: A magnitude was clamped to ``max_code``.
        underflow: A magnitude was clamped to ``min_code``.
    """

    word: int
    overflow: bool
    underflow: bool


# ----------------------------------------------------------------------
# Unary helpers.
# ----------------------------------------------------------------------


def lns_neg(word: int, fmt: LNSConfig | str) -> int:
    """Return ``-word``: flips the sign bit, leaving the zero sentinel alone."""
    fmt = get_format(fmt)
    return negate(word, fmt)


def lns_abs(word: int, fmt: LNSConfig | str) -> int:
    """Return ``|word|``: clears the sign bit."""
    fmt = get_format(fmt)
    return word & fmt.code_mask


# ----------------------------------------------------------------------
# Multiplication: an exact integer add in the log domain.
# ----------------------------------------------------------------------


def lns_mul_with_flags(
    a_code: int,
    b_code: int,
    fmt: LNSConfig | str,
    *,
    trace: Trace | None = None,
) -> OpResult:
    """Multiply two LNS words, returning range flags.

    Args:
        a_code: First operand word.
        b_code: Second operand word.
        fmt: Format shared by both operands.
        trace: Optional dict, filled in with a step-by-step record of the
            computation for the simulator display.

    Returns:
        An :class:`OpResult`.
    """
    fmt = get_format(fmt)
    if trace is not None:
        trace.update(
            op="mul",
            format=fmt.name,
            a=describe(a_code, fmt),
            b=describe(b_code, fmt),
        )

    if is_zero(a_code, fmt) or is_zero(b_code, fmt):
        if trace is not None:
            trace.update(
                branch="zero-operand",
                steps=["one operand is the reserved zero code -> product is exact zero"],
                result=describe(0, fmt),
            )
        return OpResult(0, False, False)

    ca = code_of(a_code, fmt)
    cb = code_of(b_code, fmt)
    sign = sign_of(a_code, fmt) ^ sign_of(b_code, fmt)

    # L_a + L_b = (ca - bias)/scale + (cb - bias)/scale = (ca + cb - bias - bias)/scale
    # so the result code is ca + cb - bias, with no rounding whatsoever.
    raw_code = ca + cb - fmt.bias
    code, overflow, underflow = clamp_code(raw_code, fmt)
    word = pack(sign, code, fmt)

    if trace is not None:
        la = log_of_code(ca, fmt)
        lb = log_of_code(cb, fmt)
        trace.update(
            branch="log-domain add",
            sign=sign,
            code_a=ca,
            code_b=cb,
            log_a=la,
            log_b=lb,
            raw_code=raw_code,
            result_code=code,
            overflow=overflow,
            underflow=underflow,
            result=describe(word, fmt),
            steps=[
                f"sign = sign(a) XOR sign(b) = {sign_of(a_code, fmt)} XOR "
                f"{sign_of(b_code, fmt)} = {sign}",
                f"L_a + L_b = {la:+.7f} + {lb:+.7f} = {la + lb:+.7f}",
                f"code = c_a + c_b - bias = {ca} + {cb} - {fmt.bias} = {raw_code} (exact)",
                (
                    f"clamp to [{fmt.min_code}, {fmt.max_code}] -> {code}"
                    + (" (OVERFLOW)" if overflow else " (UNDERFLOW)" if underflow else "")
                ),
            ],
        )
    return OpResult(word, overflow, underflow)


def lns_mul(
    a_code: int,
    b_code: int,
    fmt: LNSConfig | str,
    *,
    warn: bool = True,
    trace: Trace | None = None,
) -> int:
    """Multiply two LNS words.

    Sign is the XOR of the operand signs, and the magnitude code is the exact
    integer sum ``c_a + c_b - bias``.  Multiplication therefore introduces *no*
    rounding error of its own -- the only error in a product is the error
    already present in its inputs.  This is the property that makes LNS
    attractive for DNN accelerators, whose inner loops are dominated by
    multiplies.

    Args:
        a_code: First operand word.
        b_code: Second operand word.
        fmt: Format shared by both operands.
        warn: Emit range warnings on clamping.
        trace: Optional dict to fill with a step-by-step record.

    Returns:
        The packed product word.
    """
    fmt = get_format(fmt)
    result = lns_mul_with_flags(a_code, b_code, fmt, trace=trace)
    if warn:
        if result.overflow:
            warn_overflow("lns_mul", fmt.name)
        elif result.underflow:
            warn_underflow("lns_mul", fmt.name)
    return result.word


# ----------------------------------------------------------------------
# Addition: comparison + correction term.
# ----------------------------------------------------------------------


def lns_add_with_flags(
    a_code: int,
    b_code: int,
    fmt: LNSConfig | str,
    mode: Mode | None = None,
    *,
    lut_size: int | None = None,
    trace: Trace | None = None,
) -> OpResult:
    """Add two LNS words, returning range flags.

    Args:
        a_code: First operand word.
        b_code: Second operand word.
        fmt: Format shared by both operands.
        mode: ``"exact"``, ``"lut"`` or ``None`` for the global default.
        lut_size: Table resolution when ``mode == "lut"``.
        trace: Optional dict, filled in with a step-by-step record.

    Returns:
        An :class:`OpResult`.
    """
    fmt = get_format(fmt)
    a_zero = is_zero(a_code, fmt)
    b_zero = is_zero(b_code, fmt)

    if trace is not None:
        trace.update(
            op="add",
            format=fmt.name,
            mode=mode,
            a=describe(a_code, fmt),
            b=describe(b_code, fmt),
        )

    # --- zero operands -------------------------------------------------
    if a_zero and b_zero:
        if trace is not None:
            trace.update(
                branch="both-zero",
                steps=["both operands are the zero sentinel -> result is exact zero"],
                result=describe(0, fmt),
            )
        return OpResult(0, False, False)
    if a_zero:
        if trace is not None:
            trace.update(
                branch="zero-operand",
                steps=["a is the zero sentinel -> result is b, bit for bit"],
                result=describe(b_code, fmt),
            )
        return OpResult(b_code, False, False)
    if b_zero:
        if trace is not None:
            trace.update(
                branch="zero-operand",
                steps=["b is the zero sentinel -> result is a, bit for bit"],
                result=describe(a_code, fmt),
            )
        return OpResult(a_code, False, False)

    ca = code_of(a_code, fmt)
    cb = code_of(b_code, fmt)
    sa = sign_of(a_code, fmt)
    sb = sign_of(b_code, fmt)
    same_sign = sa == sb

    # --- exact cancellation --------------------------------------------
    if not same_sign and ca == cb:
        if trace is not None:
            trace.update(
                branch="exact-cancellation",
                code_a=ca,
                code_b=cb,
                steps=[
                    "opposite signs with identical codes: |a| == |b| exactly",
                    "log2(1 - 2**0) = -inf, handled as a code comparison "
                    "-> result is the exact zero sentinel",
                ],
                result=describe(0, fmt),
            )
        return OpResult(0, False, False)

    # --- order the operands: the larger magnitude sets the sign ---------
    if ca >= cb:
        c_hi, c_lo, sign = ca, cb, sa
    else:
        c_hi, c_lo, sign = cb, ca, sb

    d_code = c_lo - c_hi  # <= 0, an exact multiple of the log-step by construction
    d = d_code / fmt.scale
    kind = "add" if same_sign else "sub"

    corr = correction(d, kind, fmt, mode, lut_size)
    if corr == -math.inf:
        # Defensive: the -inf singularity belongs to d == 0, which the exact
        # cancellation branch above already caught.  Anything else reaching
        # here is a degenerate correction table; treat it as full cancellation.
        if trace is not None:
            trace.update(
                branch="exact-cancellation",
                steps=["correction term is -inf -> result is the exact zero sentinel"],
                result=describe(0, fmt),
            )
        return OpResult(0, False, False)
    delta_code = round_to_nearest(corr * fmt.scale)
    raw_code = c_hi + delta_code
    code, overflow, underflow = clamp_code(raw_code, fmt)
    word = pack(sign, code, fmt)

    if trace is not None:
        formula = "log2(1 + 2**d)" if same_sign else "log2(1 - 2**d)"
        trace.update(
            branch="same-sign" if same_sign else "opposite-sign",
            kind=kind,
            formula=formula,
            sign=sign,
            code_hi=c_hi,
            code_lo=c_lo,
            log_hi=log_of_code(c_hi, fmt),
            log_lo=log_of_code(c_lo, fmt),
            d_code=d_code,
            d=d,
            correction=corr,
            delta_code=delta_code,
            raw_code=raw_code,
            result_code=code,
            overflow=overflow,
            underflow=underflow,
            result=describe(word, fmt),
            steps=[
                f"signs {'match' if same_sign else 'differ'} -> use {formula}",
                f"larger code c_hi = {c_hi} (L = {log_of_code(c_hi, fmt):+.7f}), "
                f"smaller c_lo = {c_lo} (L = {log_of_code(c_lo, fmt):+.7f})",
                f"d = (c_lo - c_hi)/scale = {d_code}/{fmt.scale} = {d:+.7f}",
                f"correction = {formula} = {corr:+.7f}  [{mode or 'global'} mode]",
                f"delta_code = round(correction * scale) = "
                f"round({corr * fmt.scale:+.4f}) = {delta_code:+d}",
                f"code = c_hi + delta_code = {c_hi} {delta_code:+d} = {raw_code}"
                + (
                    f" -> clamped to {code}"
                    + (" (OVERFLOW)" if overflow else " (UNDERFLOW)")
                    if (overflow or underflow)
                    else ""
                ),
            ],
        )
    return OpResult(word, overflow, underflow)


def lns_add(
    a_code: int,
    b_code: int,
    fmt: LNSConfig | str,
    mode: Mode | None = None,
    *,
    lut_size: int | None = None,
    warn: bool = True,
    trace: Trace | None = None,
) -> int:
    """Add two LNS words, staying entirely in the log domain.

    Full sign handling:

    * either operand is the zero sentinel -> the other operand is returned
      unchanged (both zero -> zero);
    * same signs -> ``c_hi + round(scale * log2(1 + 2**d))``;
    * opposite signs with equal codes -> the exact zero sentinel;
    * opposite signs otherwise -> ``c_hi + round(scale * log2(1 - 2**d))``,
      which can underflow for near-cancelling operands and is then clamped to
      the smallest nonzero code with an underflow flag.

    Args:
        a_code: First operand word.
        b_code: Second operand word.
        fmt: Format shared by both operands.
        mode: ``"exact"``, ``"lut"`` or ``None`` for the global default.
        lut_size: Table resolution when ``mode == "lut"``.
        warn: Emit range warnings on clamping.
        trace: Optional dict to fill with a step-by-step record.

    Returns:
        The packed sum word.
    """
    fmt = get_format(fmt)
    result = lns_add_with_flags(a_code, b_code, fmt, mode, lut_size=lut_size, trace=trace)
    if warn:
        if result.overflow:
            warn_overflow("lns_add", fmt.name)
        elif result.underflow:
            warn_underflow("lns_add (near-cancellation)", fmt.name)
    return result.word


# ----------------------------------------------------------------------
# Subtraction and MAC, defined on top of add/mul.
# ----------------------------------------------------------------------


def lns_sub_with_flags(
    a_code: int,
    b_code: int,
    fmt: LNSConfig | str,
    mode: Mode | None = None,
    *,
    lut_size: int | None = None,
    trace: Trace | None = None,
) -> OpResult:
    """Subtract ``b`` from ``a``, returning range flags."""
    fmt = get_format(fmt)
    negated = negate(b_code, fmt)
    if trace is not None:
        trace["negated_b"] = describe(negated, fmt)
    result = lns_add_with_flags(a_code, negated, fmt, mode, lut_size=lut_size, trace=trace)
    if trace is not None:
        trace["op"] = "sub"
        steps = list(trace.get("steps", ()))
        trace["steps"] = ["a - b is evaluated as a + (-b): flip b's sign bit", *steps]
    return result


def lns_sub(
    a_code: int,
    b_code: int,
    fmt: LNSConfig | str,
    mode: Mode | None = None,
    *,
    lut_size: int | None = None,
    warn: bool = True,
    trace: Trace | None = None,
) -> int:
    """Subtract two LNS words: ``lns_add(a, negate(b))``.

    Args:
        a_code: Minuend word.
        b_code: Subtrahend word.
        fmt: Format shared by both operands.
        mode: ``"exact"``, ``"lut"`` or ``None`` for the global default.
        lut_size: Table resolution when ``mode == "lut"``.
        warn: Emit range warnings on clamping.
        trace: Optional dict to fill with a step-by-step record.

    Returns:
        The packed difference word.
    """
    fmt = get_format(fmt)
    result = lns_sub_with_flags(a_code, b_code, fmt, mode, lut_size=lut_size, trace=trace)
    if warn:
        if result.overflow:
            warn_overflow("lns_sub", fmt.name)
        elif result.underflow:
            warn_underflow("lns_sub (near-cancellation)", fmt.name)
    return result.word


def lns_mac_with_flags(
    a_code: int,
    b_code: int,
    acc_code: int,
    fmt: LNSConfig | str,
    mode: Mode | None = None,
    *,
    lut_size: int | None = None,
    trace: Trace | None = None,
) -> OpResult:
    """Fused multiply-accumulate ``a * b + acc``, returning range flags.

    The product is rounded into the format before the accumulate, matching a
    hardware MAC unit whose accumulator is the same width as its operands.
    """
    fmt = get_format(fmt)
    mul_trace: Trace | None = {} if trace is not None else None
    add_trace: Trace | None = {} if trace is not None else None

    product = lns_mul_with_flags(a_code, b_code, fmt, trace=mul_trace)
    total = lns_add_with_flags(
        product.word, acc_code, fmt, mode, lut_size=lut_size, trace=add_trace
    )

    if trace is not None:
        assert mul_trace is not None and add_trace is not None
        trace.update(
            op="mac",
            format=fmt.name,
            mode=mode,
            a=describe(a_code, fmt),
            b=describe(b_code, fmt),
            acc=describe(acc_code, fmt),
            product=mul_trace.get("result"),
            mul=mul_trace,
            add=add_trace,
            branch=add_trace.get("branch"),
            result=add_trace.get("result"),
            steps=[
                "step 1 - multiply a * b (exact code addition):",
                *[f"    {s}" for s in mul_trace.get("steps", ())],
                "step 2 - accumulate (a*b) + acc:",
                *[f"    {s}" for s in add_trace.get("steps", ())],
            ],
        )
    return OpResult(
        total.word,
        product.overflow or total.overflow,
        product.underflow or total.underflow,
    )


def lns_mac(
    a_code: int,
    b_code: int,
    acc_code: int,
    fmt: LNSConfig | str,
    mode: Mode | None = None,
    *,
    lut_size: int | None = None,
    warn: bool = True,
    trace: Trace | None = None,
) -> int:
    """Multiply-accumulate: ``lns_add(lns_mul(a, b), acc)``.

    Args:
        a_code: First multiplicand word.
        b_code: Second multiplicand word.
        acc_code: Accumulator word.
        fmt: Format shared by all three operands.
        mode: ``"exact"``, ``"lut"`` or ``None`` for the global default.
        lut_size: Table resolution when ``mode == "lut"``.
        warn: Emit range warnings on clamping.
        trace: Optional dict to fill with a step-by-step record.

    Returns:
        The packed ``a * b + acc`` word.
    """
    fmt = get_format(fmt)
    result = lns_mac_with_flags(
        a_code, b_code, acc_code, fmt, mode, lut_size=lut_size, trace=trace
    )
    if warn:
        if result.overflow:
            warn_overflow("lns_mac", fmt.name)
        elif result.underflow:
            warn_underflow("lns_mac", fmt.name)
    return result.word


def lns_dot(
    a_codes: Sequence[int] | Iterable[int],
    b_codes: Sequence[int] | Iterable[int],
    fmt: LNSConfig | str,
    mode: Mode | None = None,
    *,
    lut_size: int | None = None,
    warn: bool = False,
) -> int:
    """Dot product of two LNS vectors, accumulated in the LNS domain.

    Provided for the DNN experiments this package is meant to feed: a layer's
    output is a dot product, and the sequential-MAC accumulation modelled here
    (round after every step) is what a real LNS PE array does.

    Args:
        a_codes: Words of the first vector.
        b_codes: Words of the second vector.
        fmt: Format shared by every operand.
        mode: Log-add mode.
        lut_size: Table resolution when ``mode == "lut"``.
        warn: Emit range warnings on clamping (off by default: a long
            accumulation would otherwise be very noisy).

    Returns:
        The packed dot-product word.
    """
    fmt = get_format(fmt)
    acc = 0  # the zero sentinel
    for a, b in zip(a_codes, b_codes):
        acc = lns_mac(a, b, acc, fmt, mode, lut_size=lut_size, warn=warn)
    return acc
