"""Tests for encoding, decoding and range handling.

Coverage required by the assignment: exact powers of two, zero, negatives,
near-format-boundary values that trigger overflow/underflow, and a seeded
randomised batch.
"""

from __future__ import annotations

import math
import warnings

import pytest

from lns_arith import (
    LNS8,
    LNS16,
    LNSConfig,
    LNSOverflowWarning,
    LNSUnderflowWarning,
    code_of,
    encode_with_flags,
    fp16_to_lns,
    fp16_to_lns8,
    fp16_to_lns16,
    fp32_to_lns,
    fp32_to_lns8,
    fp32_to_lns16,
    is_zero,
    lns8_to_float,
    lns16_to_float,
    lns_to_float,
    relative_error,
    sign_of,
    to_fp16,
)
from lns_arith.utils import gen_random, gen_random_linear

FORMATS = [LNS16, LNS8]


def roundtrip(x: float, fmt: LNSConfig) -> float:
    """Encode then decode, with range warnings suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lns_to_float(fp32_to_lns(x, fmt), fmt)


# ----------------------------------------------------------------------
# Format geometry -- the bit layout must be exactly what the spec states.
# ----------------------------------------------------------------------


def test_lns16_layout_matches_specification() -> None:
    assert (LNS16.total_bits, LNS16.int_bits, LNS16.frac_bits) == (16, 8, 7)
    assert LNS16.bias == 16384
    assert LNS16.code_bits == 15
    assert LNS16.max_code == 32767
    assert LNS16.log_step == pytest.approx(1 / 128)
    assert LNS16.max_log == pytest.approx(127.9921875)
    assert LNS16.min_log == pytest.approx(-127.9921875)


def test_lns8_layout_matches_specification() -> None:
    assert (LNS8.total_bits, LNS8.int_bits, LNS8.frac_bits) == (8, 4, 3)
    assert LNS8.bias == 64
    assert LNS8.code_bits == 7
    assert LNS8.max_code == 127
    assert LNS8.log_step == 0.125
    assert LNS8.max_log == pytest.approx(7.875)
    assert LNS8.min_log == pytest.approx(-7.875)


def test_zero_code_is_reserved_and_shrinks_the_bottom_of_the_range() -> None:
    """Reserving code 0 costs exactly one log-step of low-end range."""
    for fmt in FORMATS:
        naive_min_log = -fmt.bias / fmt.scale  # if code 0 were a normal slot
        assert fmt.min_log == pytest.approx(naive_min_log + fmt.log_step)
        assert fmt.min_value == pytest.approx(2.0**fmt.min_log)


# ----------------------------------------------------------------------
# Exact values.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
@pytest.mark.parametrize("exponent", [-7, -3, -1, 0, 1, 2, 3, 7])
def test_powers_of_two_round_trip_exactly(fmt: LNSConfig, exponent: int) -> None:
    """Powers of two land on grid points, so they survive a round trip exactly."""
    value = 2.0**exponent
    assert roundtrip(value, fmt) == value
    assert roundtrip(-value, fmt) == -value


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_one_encodes_to_the_bias(fmt: LNSConfig) -> None:
    word = fp32_to_lns(1.0, fmt)
    assert code_of(word, fmt) == fmt.bias
    assert sign_of(word, fmt) == 0
    assert lns_to_float(word, fmt) == 1.0


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_zero_maps_to_the_reserved_sentinel(fmt: LNSConfig) -> None:
    word = fp32_to_lns(0.0, fmt)
    assert word == 0
    assert is_zero(word, fmt)
    assert code_of(word, fmt) == fmt.zero_code
    assert lns_to_float(word, fmt) == 0.0


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_negative_zero_is_canonicalised_to_zero(fmt: LNSConfig) -> None:
    assert fp32_to_lns(-0.0, fmt) == 0
    assert lns_to_float(0, fmt) == 0.0


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_sign_bit_is_independent_of_magnitude(fmt: LNSConfig) -> None:
    positive = fp32_to_lns(3.0, fmt)
    negative = fp32_to_lns(-3.0, fmt)
    assert code_of(positive, fmt) == code_of(negative, fmt)
    assert sign_of(positive, fmt) == 0
    assert sign_of(negative, fmt) == 1
    assert lns_to_float(negative, fmt) == -lns_to_float(positive, fmt)


def test_nan_is_rejected() -> None:
    with pytest.raises(ValueError):
        fp32_to_lns(float("nan"), LNS16)


# ----------------------------------------------------------------------
# Rounding and precision bounds.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_round_to_nearest_picks_the_closer_grid_point(fmt: LNSConfig) -> None:
    """A value one third of a step above a grid point rounds back down to it."""
    base_log = 1.0  # value 2.0, exactly on the grid
    nudged = 2.0 ** (base_log + fmt.log_step / 3.0)
    assert roundtrip(nudged, fmt) == pytest.approx(2.0)

    nudged_up = 2.0 ** (base_log + 2.0 * fmt.log_step / 3.0)
    assert roundtrip(nudged_up, fmt) == pytest.approx(2.0 ** (base_log + fmt.log_step))


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_round_trip_error_respects_the_half_step_bound(fmt: LNSConfig) -> None:
    """Relative error is bounded by 2**(step/2) - 1 everywhere in range."""
    bound = fmt.max_relative_step_error * (1 + 1e-9)
    for value in gen_random_linear(500, seed=7):
        if not (fmt.min_value <= abs(value) <= fmt.max_value):
            continue
        assert relative_error(value, roundtrip(value, fmt)) <= bound


def test_relative_error_is_flat_across_the_whole_dynamic_range() -> None:
    """The defining LNS property: precision does not depend on magnitude."""
    errors = []
    for exponent in range(-100, 101, 10):
        value = 1.7 * 2.0**exponent  # deliberately off-grid
        errors.append(relative_error(value, roundtrip(value, LNS16)))
    assert max(errors) - min(errors) < 1e-12
    assert max(errors) <= LNS16.max_relative_step_error


# ----------------------------------------------------------------------
# Boundaries: overflow and underflow clamp, flag, and never wrap.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_extreme_representable_values_do_not_flag(fmt: LNSConfig) -> None:
    for value in (fmt.max_value, fmt.min_value, -fmt.max_value, -fmt.min_value):
        result = encode_with_flags(value, fmt, source="fp64")
        assert not result.overflow and not result.underflow
        assert lns_to_float(result.word, fmt) == pytest.approx(value)


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_overflow_clamps_flags_and_warns(fmt: LNSConfig) -> None:
    too_big = fmt.max_value * 4.0
    result = encode_with_flags(too_big, fmt, source="fp64")
    assert result.overflow and not result.underflow
    assert code_of(result.word, fmt) == fmt.max_code
    assert lns_to_float(result.word, fmt) == pytest.approx(fmt.max_value)

    with pytest.warns(LNSOverflowWarning):
        fp32_to_lns(too_big, fmt)


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_underflow_clamps_flags_and_warns(fmt: LNSConfig) -> None:
    too_small = fmt.min_value / 4.0
    result = encode_with_flags(too_small, fmt, source="fp64")
    assert result.underflow and not result.overflow
    assert code_of(result.word, fmt) == fmt.min_code
    # Underflow clamps to the smallest nonzero code -- it must NOT flush to zero,
    # because zero is a reserved exact value.
    assert not is_zero(result.word, fmt)
    assert lns_to_float(result.word, fmt) == pytest.approx(fmt.min_value)

    with pytest.warns(LNSUnderflowWarning):
        fp32_to_lns(too_small, fmt)


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_clamping_is_saturating_not_wrapping(fmt: LNSConfig) -> None:
    """A value far past the top must not reappear at the bottom."""
    huge = encode_with_flags(1e300, fmt, source="fp64")
    tiny = encode_with_flags(1e-300, fmt, source="fp64")
    assert code_of(huge.word, fmt) == fmt.max_code
    assert code_of(tiny.word, fmt) == fmt.min_code
    assert lns_to_float(huge.word, fmt) > lns_to_float(tiny.word, fmt)


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_infinity_saturates_with_an_overflow_flag(fmt: LNSConfig) -> None:
    result = encode_with_flags(math.inf, fmt, source="fp64")
    assert result.overflow
    assert code_of(result.word, fmt) == fmt.max_code
    assert sign_of(result.word, fmt) == 0

    negative = encode_with_flags(-math.inf, fmt, source="fp64")
    assert negative.overflow
    assert sign_of(negative.word, fmt) == 1


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_boundary_codes_are_reachable_and_distinct(fmt: LNSConfig) -> None:
    at_min = encode_with_flags(2.0**fmt.min_log, fmt, source="fp64")
    just_above = encode_with_flags(2.0 ** (fmt.min_log + fmt.log_step), fmt, source="fp64")
    at_max = encode_with_flags(2.0**fmt.max_log, fmt, source="fp64")
    assert code_of(at_min.word, fmt) == fmt.min_code == 1
    assert code_of(just_above.word, fmt) == 2
    assert code_of(at_max.word, fmt) == fmt.max_code


# ----------------------------------------------------------------------
# FP16 source path.
# ----------------------------------------------------------------------


def test_fp16_source_quantises_before_encoding() -> None:
    """A value FP16 cannot hold must reach the encoder already rounded."""
    value = 1.0 + 2.0**-12  # below FP16's resolution at 1.0
    assert to_fp16(value) == 1.0
    assert fp16_to_lns(value, LNS16) == fp16_to_lns(1.0, LNS16)


def test_fp16_saturation_becomes_an_lns_overflow() -> None:
    with pytest.warns(LNSOverflowWarning):
        word = fp16_to_lns(1e6, LNS16)  # above FP16's 65504 -> inf -> clamp
    assert code_of(word, LNS16) == LNS16.max_code


def test_fp16_and_fp32_paths_agree_on_exactly_representable_values() -> None:
    for value in (0.0, 1.0, -2.5, 0.125, 1024.0):
        assert fp16_to_lns(value, LNS16) == fp32_to_lns(value, LNS16)
        assert fp16_to_lns(value, LNS8, warn=False) == fp32_to_lns(value, LNS8, warn=False)


# ----------------------------------------------------------------------
# Named wrappers.
# ----------------------------------------------------------------------


def test_named_wrappers_match_the_parametrised_api() -> None:
    for value in (3.5, -0.75, 0.0, 12.0):
        assert fp32_to_lns16(value) == fp32_to_lns(value, LNS16)
        assert fp32_to_lns8(value, warn=False) == fp32_to_lns(value, LNS8, warn=False)
        assert fp16_to_lns16(value) == fp16_to_lns(value, LNS16)
        assert fp16_to_lns8(value, warn=False) == fp16_to_lns(value, LNS8, warn=False)
    assert lns16_to_float(fp32_to_lns16(3.5)) == lns_to_float(fp32_to_lns(3.5, LNS16), LNS16)
    assert lns8_to_float(fp32_to_lns8(3.5)) == lns_to_float(fp32_to_lns(3.5, LNS8), LNS8)


def test_format_can_be_named_by_string() -> None:
    assert fp32_to_lns(2.0, "LNS16") == fp32_to_lns(2.0, LNS16)
    assert fp32_to_lns(2.0, "lns8") == fp32_to_lns(2.0, LNS8)
    with pytest.raises(KeyError):
        fp32_to_lns(2.0, "LNS4")


# ----------------------------------------------------------------------
# Randomised batch, fixed seed.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_randomised_round_trip_batch(fmt: LNSConfig) -> None:
    """Seeded batch: every in-range value round-trips within the half-step bound."""
    bound = fmt.max_relative_step_error * (1 + 1e-9)
    checked = 0
    for value in gen_random(2000, seed=20240917):
        if not (fmt.min_value <= abs(value) <= fmt.max_value):
            continue
        decoded = roundtrip(value, fmt)
        assert math.copysign(1.0, decoded) == math.copysign(1.0, value)
        assert relative_error(value, decoded) <= bound
        checked += 1
    assert checked > 100, "seeded batch produced too few in-range samples to be meaningful"


def test_generators_are_reproducible() -> None:
    assert gen_random(50, seed=1) == gen_random(50, seed=1)
    assert gen_random(50, seed=1) != gen_random(50, seed=2)
