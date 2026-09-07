"""Tests for LNS arithmetic: multiply, add, subtract and MAC.

Coverage required by the assignment: exact powers of two, zero, negatives,
near-boundary overflow/underflow, and a seeded randomised batch -- plus a
structural test that ``ops.py`` never converts to float mid-operation.
"""

from __future__ import annotations

import ast
import math
import warnings
from pathlib import Path

import pytest

import lns_arith.ops as ops_module
from lns_arith import (
    LNS8,
    LNS16,
    LNSConfig,
    LNSOverflowWarning,
    LNSUnderflowWarning,
    code_of,
    default_lut_size,
    fp32_to_lns,
    get_lut,
    is_zero,
    lns_abs,
    lns_add,
    lns_add_with_flags,
    lns_dot,
    lns_mac,
    lns_mac_with_flags,
    lns_mul,
    lns_mul_with_flags,
    lns_neg,
    lns_sub,
    lns_to_float,
    relative_error,
    sign_of,
    use_mode,
)
from lns_arith.logadd import exact_mode, get_mode, lut_mode, set_mode
from lns_arith.utils import gen_random_linear

FORMATS = [LNS16, LNS8]
MODES = ["exact", "lut"]


def enc(x: float, fmt: LNSConfig) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fp32_to_lns(x, fmt)


def dec(word: int, fmt: LNSConfig) -> float:
    return lns_to_float(word, fmt)


# ----------------------------------------------------------------------
# The structural constraint: no float conversion inside the arithmetic core.
# ----------------------------------------------------------------------


def test_ops_module_never_calls_lns_to_float() -> None:
    """ops.py must not decode to float to compute a result.

    Parsed with :mod:`ast` rather than grepped, so the prose in the module
    docstring that *describes* this rule does not trip the test.
    """
    tree = ast.parse(Path(ops_module.__file__).read_text(encoding="utf-8"))
    banned = {"lns_to_float", "lns16_to_float", "lns8_to_float", "encode_with_flags"}

    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            if node.module and "convert" in node.module:
                pytest.fail(f"ops.py imports from {node.module}, the float boundary")
            referenced.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if "convert" in alias.name:
                    pytest.fail(f"ops.py imports {alias.name}, the float boundary")

    leaked = banned & referenced
    assert not leaked, f"ops.py references float-domain helpers: {sorted(leaked)}"


def test_ops_functions_accept_and_return_ints() -> None:
    """The public arithmetic API is code-in / code-out, never float-in."""
    a, b = enc(3.0, LNS16), enc(0.5, LNS16)
    for word in (
        lns_mul(a, b, LNS16),
        lns_add(a, b, LNS16),
        lns_sub(a, b, LNS16),
        lns_mac(a, b, b, LNS16),
        lns_neg(a, LNS16),
        lns_abs(a, LNS16),
    ):
        assert isinstance(word, int) and not isinstance(word, bool)
        assert 0 <= word <= LNS16.word_mask


# ----------------------------------------------------------------------
# Multiplication.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
@pytest.mark.parametrize(("p", "q"), [(0, 0), (1, 2), (-2, 3), (3, -3), (-1, -1)])
def test_multiplying_powers_of_two_is_exact(fmt: LNSConfig, p: int, q: int) -> None:
    a, b = 2.0**p, 2.0**q
    assert dec(lns_mul(enc(a, fmt), enc(b, fmt), fmt), fmt) == a * b


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_multiply_is_an_exact_integer_code_add(fmt: LNSConfig) -> None:
    """The product code is c_a + c_b - bias, with no rounding at all."""
    a, b = enc(3.0, fmt), enc(5.0, fmt)
    product = lns_mul(a, b, fmt)
    assert code_of(product, fmt) == code_of(a, fmt) + code_of(b, fmt) - fmt.bias


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
@pytest.mark.parametrize(("sa", "sb", "expected"), [(1, 1, 0), (1, -1, 1), (-1, -1, 0)])
def test_multiply_sign_is_xor(fmt: LNSConfig, sa: float, sb: float, expected: int) -> None:
    word = lns_mul(enc(3.0 * sa, fmt), enc(2.0 * sb, fmt), fmt)
    assert sign_of(word, fmt) == expected


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_multiply_by_one_is_the_identity(fmt: LNSConfig) -> None:
    one = enc(1.0, fmt)
    for value in (3.0, -0.25, 7.5):
        word = enc(value, fmt)
        assert lns_mul(word, one, fmt) == word


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_multiply_by_zero_is_exactly_zero(fmt: LNSConfig) -> None:
    z = enc(0.0, fmt)
    assert lns_mul(enc(7.0, fmt), z, fmt) == z
    assert lns_mul(z, enc(-7.0, fmt), fmt) == z
    assert lns_mul(z, z, fmt) == z
    assert is_zero(lns_mul(enc(1e30, fmt), z, fmt), fmt)


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_multiply_overflow_clamps_and_flags(fmt: LNSConfig) -> None:
    big = enc(fmt.max_value, fmt)
    result = lns_mul_with_flags(big, big, fmt)
    assert result.overflow and not result.underflow
    assert code_of(result.word, fmt) == fmt.max_code
    with pytest.warns(LNSOverflowWarning):
        lns_mul(big, big, fmt)


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_multiply_underflow_clamps_and_flags(fmt: LNSConfig) -> None:
    small = enc(fmt.min_value, fmt)
    result = lns_mul_with_flags(small, small, fmt)
    assert result.underflow and not result.overflow
    assert code_of(result.word, fmt) == fmt.min_code
    assert not is_zero(result.word, fmt)  # clamps, never flushes to zero
    with pytest.warns(LNSUnderflowWarning):
        lns_mul(small, small, fmt)


# ----------------------------------------------------------------------
# Addition: sign handling and the correction term.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
@pytest.mark.parametrize("mode", MODES)
def test_adding_equal_powers_of_two_doubles_exactly(fmt: LNSConfig, mode: str) -> None:
    """x + x is a one-code shift by log2(2) = 1, so it is exact."""
    for exponent in (-3, 0, 2):
        value = 2.0**exponent
        word = enc(value, fmt)
        assert dec(lns_add(word, word, fmt, mode), fmt) == 2.0 * value


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
@pytest.mark.parametrize("mode", MODES)
def test_zero_is_the_additive_identity(fmt: LNSConfig, mode: str) -> None:
    z = enc(0.0, fmt)
    for value in (3.0, -0.25, fmt.max_value, fmt.min_value):
        word = enc(value, fmt)
        assert lns_add(word, z, fmt, mode) == word
        assert lns_add(z, word, fmt, mode) == word
    assert lns_add(z, z, fmt, mode) == z


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
@pytest.mark.parametrize("mode", MODES)
def test_exact_cancellation_gives_the_zero_sentinel(fmt: LNSConfig, mode: str) -> None:
    """Opposite signs with identical codes must produce exact zero, not a tiny value."""
    for value in (1.0, -3.25, 7.0, fmt.max_value):
        word = enc(value, fmt)
        result = lns_add_with_flags(word, lns_neg(word, fmt), fmt, mode)
        assert is_zero(result.word, fmt)
        assert dec(result.word, fmt) == 0.0
        assert not result.overflow and not result.underflow
        assert lns_sub(word, word, fmt, mode) == 0


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
@pytest.mark.parametrize("mode", MODES)
def test_addition_is_commutative(fmt: LNSConfig, mode: str) -> None:
    values = [3.0, -1.5, 0.0, 0.125, -0.125, 20.0]
    for a in values:
        for b in values:
            wa, wb = enc(a, fmt), enc(b, fmt)
            assert lns_add(wa, wb, fmt, mode, warn=False) == lns_add(
                wb, wa, fmt, mode, warn=False
            )


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
@pytest.mark.parametrize("mode", MODES)
def test_larger_magnitude_sets_the_result_sign(fmt: LNSConfig, mode: str) -> None:
    positive_wins = lns_add(enc(5.0, fmt), enc(-2.0, fmt), fmt, mode)
    negative_wins = lns_add(enc(2.0, fmt), enc(-5.0, fmt), fmt, mode)
    assert sign_of(positive_wins, fmt) == 0
    assert sign_of(negative_wins, fmt) == 1
    assert dec(positive_wins, fmt) > 0 > dec(negative_wins, fmt)


@pytest.mark.parametrize("mode", MODES)
def test_addition_matches_fp32_within_the_expected_bound(mode: str) -> None:
    """Same-sign addition of in-range values stays within ~1.5 half-steps."""
    bound = 1.5 * LNS16.max_relative_step_error
    for a, b in [(1.0, 1.0), (3.0, 0.5), (100.0, 0.01), (-2.0, -8.0), (0.3, 0.7)]:
        got = dec(lns_add(enc(a, LNS16), enc(b, LNS16), LNS16, mode), LNS16)
        assert relative_error(a + b, got) <= bound


@pytest.mark.parametrize("mode", MODES)
def test_opposite_sign_addition_matches_fp32_away_from_cancellation(mode: str) -> None:
    bound = 2.0 * LNS16.max_relative_step_error
    for a, b in [(5.0, -2.0), (-9.0, 3.0), (100.0, -1.0), (0.75, -0.25)]:
        got = dec(lns_add(enc(a, LNS16), enc(b, LNS16), LNS16, mode), LNS16)
        assert relative_error(a + b, got) <= bound


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_far_apart_operands_leave_the_larger_unchanged(fmt: LNSConfig) -> None:
    """When d is beyond the table domain the correction rounds to zero."""
    big = enc(1.0, fmt)
    tiny = enc(2.0 ** (-fmt.lut_domain - 2), fmt)
    assert lns_add(big, tiny, fmt, "exact", warn=False) == big
    assert lns_add(big, tiny, fmt, "lut", warn=False) == big


def test_addition_overflow_clamps_and_flags() -> None:
    big = enc(LNS16.max_value, LNS16)
    result = lns_add_with_flags(big, big, LNS16, "exact")
    assert result.overflow
    assert code_of(result.word, LNS16) == LNS16.max_code
    with pytest.warns(LNSOverflowWarning):
        lns_add(big, big, LNS16, "exact")


def test_near_cancellation_underflows_and_flags() -> None:
    """Two tiny, nearly-equal opposite-sign operands drive the result below range."""
    a = enc(LNS16.min_value * 4, LNS16)
    b = LNS16.word_mask & (a ^ (1 << LNS16.sign_shift))  # -a, but one code apart
    b_shifted = (b & ~LNS16.code_mask) | (code_of(b, LNS16) - 1)
    result = lns_add_with_flags(a, b_shifted, LNS16, "exact")
    assert result.underflow
    assert code_of(result.word, LNS16) == LNS16.min_code
    assert not is_zero(result.word, LNS16)
    with pytest.warns(LNSUnderflowWarning):
        lns_add(a, b_shifted, LNS16, "exact")


# ----------------------------------------------------------------------
# Subtraction.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
@pytest.mark.parametrize("mode", MODES)
def test_sub_is_add_of_the_negation(fmt: LNSConfig, mode: str) -> None:
    for a, b in [(5.0, 2.0), (-3.0, 4.0), (0.0, 2.0), (2.0, 0.0), (1.5, -1.5)]:
        wa, wb = enc(a, fmt), enc(b, fmt)
        assert lns_sub(wa, wb, fmt, mode, warn=False) == lns_add(
            wa, lns_neg(wb, fmt), fmt, mode, warn=False
        )


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_negation_round_trips_and_leaves_zero_alone(fmt: LNSConfig) -> None:
    for value in (3.0, -3.0, 0.125):
        word = enc(value, fmt)
        assert lns_neg(lns_neg(word, fmt), fmt) == word
        assert dec(lns_neg(word, fmt), fmt) == -dec(word, fmt)
    z = enc(0.0, fmt)
    assert lns_neg(z, fmt) == z


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_abs_clears_the_sign(fmt: LNSConfig) -> None:
    assert lns_abs(enc(-3.0, fmt), fmt) == enc(3.0, fmt)
    assert lns_abs(enc(3.0, fmt), fmt) == enc(3.0, fmt)


def test_subtracting_powers_of_two_is_exact_when_the_result_is_one() -> None:
    """4 - 2 = 2 lands on a grid point, so it comes out exactly."""
    a, b = enc(4.0, LNS16), enc(2.0, LNS16)
    assert dec(lns_sub(a, b, LNS16, "exact"), LNS16) == 2.0


# ----------------------------------------------------------------------
# MAC.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
@pytest.mark.parametrize("mode", MODES)
def test_mac_equals_add_of_mul(fmt: LNSConfig, mode: str) -> None:
    for a, b, acc in [(2.0, 3.0, 1.0), (-2.0, 0.5, 4.0), (0.0, 5.0, 2.0), (1.5, -1.5, 0.0)]:
        wa, wb, wacc = enc(a, fmt), enc(b, fmt), enc(acc, fmt)
        expected = lns_add(lns_mul(wa, wb, fmt, warn=False), wacc, fmt, mode, warn=False)
        assert lns_mac(wa, wb, wacc, fmt, mode, warn=False) == expected


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
@pytest.mark.parametrize("mode", MODES)
def test_mac_with_powers_of_two_is_exact(fmt: LNSConfig, mode: str) -> None:
    # 2 * 4 + 8 = 16, all grid points.
    word = lns_mac(enc(2.0, fmt), enc(4.0, fmt), enc(8.0, fmt), fmt, mode)
    assert dec(word, fmt) == 16.0


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_mac_with_zero_accumulator_is_the_product(fmt: LNSConfig) -> None:
    wa, wb = enc(3.0, fmt), enc(-2.0, fmt)
    assert lns_mac(wa, wb, enc(0.0, fmt), fmt, "exact") == lns_mul(wa, wb, fmt)


def test_mac_propagates_the_product_overflow_flag() -> None:
    big = enc(LNS16.max_value, LNS16)
    result = lns_mac_with_flags(big, big, enc(1.0, LNS16), LNS16, "exact")
    assert result.overflow


def test_dot_product_accumulates_in_the_lns_domain() -> None:
    a = [enc(v, LNS16) for v in (1.0, 2.0, 3.0, 4.0)]
    b = [enc(v, LNS16) for v in (0.5, 0.25, 2.0, -1.0)]
    reference = sum(x * y for x, y in zip((1.0, 2.0, 3.0, 4.0), (0.5, 0.25, 2.0, -1.0)))
    got = dec(lns_dot(a, b, LNS16, "exact"), LNS16)
    assert relative_error(reference, got) < 0.01


# ----------------------------------------------------------------------
# Log-add correction term and its two modes.
# ----------------------------------------------------------------------


def test_exact_correction_matches_the_closed_form() -> None:
    for d in (-0.5, -1.0, -4.0, -16.0):
        assert exact_mode(d, "add") == pytest.approx(math.log2(1 + 2**d))
        assert exact_mode(d, "sub") == pytest.approx(math.log2(1 - 2**d))
    assert exact_mode(0.0, "add") == pytest.approx(1.0)
    assert exact_mode(0.0, "sub") == -math.inf


def test_correction_rejects_positive_gaps() -> None:
    with pytest.raises(ValueError):
        exact_mode(0.5, "add")
    with pytest.raises(ValueError):
        lut_mode(0.5, "add", LNS16)


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_lut_entries_lie_on_the_format_grid(fmt: LNSConfig) -> None:
    """A hardware ROM stores the correction at the format's own resolution."""
    table = get_lut(fmt, "add")
    assert len(table) == default_lut_size(fmt)
    for entry in table.entries:
        assert entry * fmt.scale == pytest.approx(round(entry * fmt.scale), abs=1e-9)


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_full_resolution_lut_reproduces_exact_mode(fmt: LNSConfig) -> None:
    """At one entry per log-step the LUT is exact on the format's grid."""
    values = [-4.0, -1.25, 0.5, 3.0, 0.125, -0.03125, 17.0]
    for a in values:
        for b in values:
            wa, wb = enc(a, fmt), enc(b, fmt)
            assert lns_add(wa, wb, fmt, "lut", warn=False) == lns_add(
                wa, wb, fmt, "exact", warn=False
            )


def test_decimated_lut_is_measurably_worse() -> None:
    """A table coarser than the format is where LUT quantisation error appears."""
    coarse = (default_lut_size(LNS16) - 1) // 8 + 1
    exact_errors, coarse_errors = [], []
    for a, b in zip(gen_random_linear(200, seed=3), gen_random_linear(200, seed=4)):
        if a + b == 0:
            continue
        wa, wb = enc(a, LNS16), enc(b, LNS16)
        exact_errors.append(
            relative_error(a + b, dec(lns_add(wa, wb, LNS16, "exact", warn=False), LNS16))
        )
        coarse_errors.append(
            relative_error(
                a + b,
                dec(
                    lns_add(wa, wb, LNS16, "lut", lut_size=coarse, warn=False),
                    LNS16,
                ),
            )
        )
    assert sum(coarse_errors) > sum(exact_errors)


def test_lut_beyond_its_domain_returns_zero_correction() -> None:
    for fmt in FORMATS:
        assert lut_mode(-fmt.lut_domain - 1.0, "add", fmt) == 0.0
        assert lut_mode(-fmt.lut_domain - 1.0, "sub", fmt) == 0.0


def test_decimated_sub_lut_never_annihilates_a_nonzero_gap() -> None:
    """Address 0 of a sub table is -inf; a nonzero gap must not round onto it."""
    coarse = 9  # very coarse: first step spans many log-steps
    a = enc(1.0, LNS16)
    b = LNS16.word_mask & (a ^ (1 << LNS16.sign_shift))
    b_near = (b & ~LNS16.code_mask) | (code_of(b, LNS16) - 1)  # one code apart
    word = lns_add(a, b_near, LNS16, "lut", lut_size=coarse, warn=False)
    assert not is_zero(word, LNS16)


def test_global_mode_switch_and_context_manager() -> None:
    previous = get_mode()
    try:
        set_mode("lut")
        assert get_mode() == "lut"
        with use_mode("exact"):
            assert get_mode() == "exact"
        assert get_mode() == "lut"
        with pytest.raises(ValueError):
            set_mode("bogus")
    finally:
        set_mode(previous)


def test_global_mode_is_used_when_mode_is_none() -> None:
    coarse_a, coarse_b = enc(1.0, LNS8), enc(0.6, LNS8)
    with use_mode("exact"):
        from_global = lns_add(coarse_a, coarse_b, LNS8, warn=False)
    explicit = lns_add(coarse_a, coarse_b, LNS8, "exact", warn=False)
    assert from_global == explicit


# ----------------------------------------------------------------------
# Tracing (used by both web front-ends).
# ----------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_trace_records_the_branch_and_the_steps(mode: str) -> None:
    trace: dict = {}
    lns_add(enc(3.0, LNS16), enc(-1.0, LNS16), LNS16, mode, trace=trace)
    assert trace["op"] == "add"
    assert trace["branch"] == "opposite-sign"
    assert trace["formula"] == "log2(1 - 2**d)"
    assert trace["d"] < 0
    assert trace["steps"]

    trace = {}
    lns_mac(enc(2.0, LNS16), enc(3.0, LNS16), enc(1.0, LNS16), LNS16, mode, trace=trace)
    assert trace["op"] == "mac"
    assert "mul" in trace and "add" in trace


def test_trace_marks_exact_cancellation() -> None:
    trace: dict = {}
    word = enc(2.0, LNS16)
    lns_add(word, lns_neg(word, LNS16), LNS16, "exact", trace=trace)
    assert trace["branch"] == "exact-cancellation"


# ----------------------------------------------------------------------
# Randomised batch, fixed seed.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
@pytest.mark.parametrize("mode", MODES)
def test_randomised_arithmetic_batch(fmt: LNSConfig, mode: str) -> None:
    """Seeded batch: multiplication error stays bounded and MAC stays sane."""
    mul_bound = 2.2 * fmt.max_relative_step_error  # two input roundings, no more
    a_values = gen_random_linear(500, seed=424242)
    b_values = gen_random_linear(500, seed=242424)
    checked = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a, b in zip(a_values, b_values):
            if not (fmt.min_value <= abs(a) <= fmt.max_value):
                continue
            if not (fmt.min_value <= abs(b) <= fmt.max_value):
                continue
            product = a * b
            if not (fmt.min_value <= abs(product) <= fmt.max_value):
                continue
            got = dec(lns_mul(enc(a, fmt), enc(b, fmt), fmt, warn=False), fmt)
            assert relative_error(product, got) <= mul_bound
            assert math.copysign(1.0, got) == math.copysign(1.0, product)
            checked += 1
    assert checked > 50


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_randomised_addition_respects_the_error_model(fmt: LNSConfig) -> None:
    """Addition error must match the analytic bound, cancellation included.

    With ``eps`` the half-step relative error and ``r = |a+b| / max(|a|,|b|)``
    the cancellation factor, each input contributes at most ``eps * |input|``
    of absolute error and the result is rounded once more::

        rel_err <= (1 + 2*eps/r) * (1 + eps) - 1

    The ``2/r`` term is exactly the cancellation blow-up: as ``r -> 0`` the
    bound diverges, which is why LNS relative error is flat *except* when
    subtracting nearly-equal operands.
    """
    eps = fmt.max_relative_step_error
    a_values = gen_random_linear(400, seed=101)
    b_values = gen_random_linear(400, seed=202)
    checked = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a, b in zip(a_values, b_values):
            total = a + b
            biggest = max(abs(a), abs(b))
            if total == 0.0 or biggest == 0.0:
                continue
            if not all(fmt.min_value <= abs(v) <= fmt.max_value for v in (a, b, total)):
                continue
            got = dec(lns_add(enc(a, fmt), enc(b, fmt), fmt, "exact", warn=False), fmt)
            cancellation_factor = abs(total) / biggest
            bound = (1 + 2 * eps / cancellation_factor) * (1 + eps) - 1
            assert relative_error(total, got) <= bound * 1.01, (
                f"a={a!r} b={b!r} r={cancellation_factor:.4f}"
            )
            checked += 1
    assert checked > 50


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_same_sign_addition_error_stays_near_one_half_step(fmt: LNSConfig) -> None:
    """With no cancellation possible, add error does not accumulate."""
    bound = ((1 + 2 * fmt.max_relative_step_error) ** 2 - 1) * 1.01
    checked = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a, b in zip(gen_random_linear(400, seed=11), gen_random_linear(400, seed=12)):
            a, b = abs(a), abs(b)  # force same sign: no cancellation branch
            total = a + b
            if not all(fmt.min_value <= v <= fmt.max_value for v in (a, b, total)):
                continue
            got = dec(lns_add(enc(a, fmt), enc(b, fmt), fmt, "exact", warn=False), fmt)
            assert relative_error(total, got) <= bound
            checked += 1
    assert checked > 50
