"""Runs the full benchmark experiment and asserts the results are sane.

This is the assignment's section-4 experiment executed as a test: a small but
complete sweep over formats, operations, log-add modes and input categories,
with assertions that pin the numbers to the theory rather than to whatever the
code happens to produce.  If a change to the arithmetic silently degrades
accuracy, or if the clamping behaviour regresses, this fails.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import run_experiments as bench  # noqa: E402
from lns_arith import LNS8, LNS16  # noqa: E402
from lns_arith.utils import CATEGORY_NAMES  # noqa: E402

SAMPLES = 200
SEED = 4242


@pytest.fixture(scope="module")
def report() -> bench.Report:
    """One shared experiment run for the whole module (seeded, reproducible)."""
    return bench.run_all(samples=SAMPLES, seed=SEED)


def cell(
    report: bench.Report, fmt_name: str, op: str, category: str, mode: str | None = None
) -> bench.ErrorRow:
    row = bench._lookup(report.error_rows, fmt_name, op, category, mode)
    assert row is not None, f"missing cell {fmt_name}/{op}/{category}/{mode}"
    return row


# ----------------------------------------------------------------------
# The sweep is complete.
# ----------------------------------------------------------------------


def test_every_format_operation_category_combination_is_measured(report: bench.Report) -> None:
    for fmt_name in ("LNS16", "LNS8", "FP16"):
        for op in bench.OPERATIONS:
            for category in CATEGORY_NAMES:
                assert bench._lookup(report.error_rows, fmt_name, op, category) is not None


def test_both_logadd_modes_are_exercised_for_mode_sensitive_ops(report: bench.Report) -> None:
    modes = {r.mode for r in report.error_rows if r.operation == "add" and r.format == "LNS16"}
    assert "exact" in modes
    assert any(m.startswith("lut[") for m in modes)


def test_mode_independent_ops_are_reported_once(report: bench.Report) -> None:
    """convert and mul never touch the correction table, so no per-mode rows."""
    for op in ("convert", "mul"):
        modes = {r.mode for r in report.error_rows if r.operation == op}
        assert modes == {"n/a"}


def test_run_is_reproducible() -> None:
    first = bench.run_error_experiments(samples=50, seed=7, categories=["random"])
    second = bench.run_error_experiments(samples=50, seed=7, categories=["random"])
    assert [r.as_dict() for r in first] == [r.as_dict() for r in second]


# ----------------------------------------------------------------------
# Conversion error must respect the half-step bound.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("fmt_name", "fmt"), [("LNS16", LNS16), ("LNS8", LNS8)])
def test_conversion_error_is_bounded_by_half_a_log_step(
    report: bench.Report, fmt_name: str, fmt
) -> None:
    """On in-range inputs, no sample may exceed 2**(step/2) - 1."""
    for category in ("positive", "negative", "random_linear"):
        row = cell(report, fmt_name, "convert", category)
        if row.clamped:
            continue  # clamped samples measure range, not rounding
        assert row.max_error <= fmt.max_relative_step_error * 1.001, (
            f"{fmt_name}/{category}: {row.max_error} > bound"
        )
        assert row.mean_error <= fmt.max_relative_step_error


def test_conversion_error_is_flat_across_categories(report: bench.Report) -> None:
    """LNS16's median error must not depend on the magnitude of its inputs."""
    medians = [
        cell(report, "LNS16", "convert", category).median_error
        for category in ("positive", "negative", "small", "large", "random_log", "random")
    ]
    assert max(medians) / min(medians) < 1.5, f"error is not flat across range: {medians}"


def test_lns16_is_more_accurate_than_lns8_everywhere(report: bench.Report) -> None:
    for op in bench.OPERATIONS:
        for category in ("positive", "negative", "random_linear"):
            lns16 = cell(report, "LNS16", op, category)
            lns8 = cell(report, "LNS8", op, category)
            assert lns16.median_error <= lns8.median_error


def test_lns16_precision_sits_between_fp16_and_the_lns8_step(report: bench.Report) -> None:
    """LNS16 is coarser than FP16 but far finer than LNS8 -- as the bit budgets imply."""
    lns16 = cell(report, "LNS16", "convert", "random_linear").median_error
    lns8 = cell(report, "LNS8", "convert", "random_linear").median_error
    fp16 = cell(report, "FP16", "convert", "random_linear").median_error
    assert fp16 < lns16 < lns8


# ----------------------------------------------------------------------
# Zero must stay exact.
# ----------------------------------------------------------------------


def test_zero_operands_produce_exact_results(report: bench.Report) -> None:
    """Every sample in the zero category whose FP32 reference is 0 must be exact.

    The category deliberately mixes exact zeros with ordinary values so that
    ``0 + 0``, ``0 + x`` and ``x + 0`` are all covered, so the aggregate cell
    also contains ordinary conversion error.  This walks the samples directly
    and checks the cases that must be bit-exact.
    """
    from lns_arith.utils import make_pairs

    triples = make_pairs("zero", 200, SEED)
    assert any(a == 0.0 and b == 0.0 for a, b, _ in triples)
    assert any(a == 0.0 and b != 0.0 for a, b, _ in triples)
    assert any(a != 0.0 and b == 0.0 for a, b, _ in triples)

    for fmt_name, fmt in (("LNS16", LNS16), ("LNS8", LNS8)):
        for op in bench.OPERATIONS:
            for a, b, acc in triples:
                ref = bench.reference(op, a, b, acc, bench.to_fp32)
                if ref != 0.0:
                    continue
                got = bench.evaluate_lns(op, a, b, acc, fmt, "exact", None)
                assert got.value == 0.0, f"{fmt_name}/{op}: {a} {b} {acc} -> {got.value}"


def test_zero_category_error_never_exceeds_a_conversion(report: bench.Report) -> None:
    """Adding zero passes the other operand through, so no error is added."""
    for fmt_name, fmt in (("LNS16", LNS16), ("LNS8", LNS8)):
        for op in ("convert", "mul", "add"):
            mode = "exact" if op == "add" else None
            row = cell(report, fmt_name, op, "zero", mode)
            assert row.max_error <= fmt.max_relative_step_error * 1.001, row.worst_case


# ----------------------------------------------------------------------
# Multiplication adds no rounding of its own.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("fmt_name", "fmt"), [("LNS16", LNS16), ("LNS8", LNS8)])
def test_multiplication_error_is_at_most_two_input_roundings(
    report: bench.Report, fmt_name: str, fmt
) -> None:
    """Product error = combined input error; the code sum itself is exact."""
    bound = (1 + fmt.max_relative_step_error) ** 2 - 1
    for category in ("positive", "negative", "random_linear"):
        row = cell(report, fmt_name, "mul", category)
        if row.clamped:
            continue
        assert row.max_error <= bound * 1.001, f"{fmt_name}/{category}: {row.worst_case}"


def test_multiplication_has_no_catastrophic_tail(report: bench.Report) -> None:
    """Unlike addition, multiplication can never lose the whole result."""
    for fmt_name, fmt in (("LNS16", LNS16), ("LNS8", LNS8)):
        row = cell(report, fmt_name, "mul", "random_linear")
        if not row.clamped:
            assert row.max_error < 0.2


def test_addition_has_a_cancellation_tail(report: bench.Report) -> None:
    """Addition's max error is far above its median -- that gap is cancellation."""
    row = cell(report, "LNS16", "add", "random_linear", "exact")
    assert row.max_error > 10 * row.median_error


# ----------------------------------------------------------------------
# Exact vs LUT modes.
# ----------------------------------------------------------------------


def test_full_resolution_lut_matches_exact_mode(report: bench.Report) -> None:
    """One table entry per log-step is exact with respect to the format's grid."""
    for fmt_name, fmt in (("LNS16", LNS16), ("LNS8", LNS8)):
        full = f"lut[{bench.default_lut_size(fmt)}]"
        for op in ("add", "mac"):
            for category in CATEGORY_NAMES:
                exact_row = cell(report, fmt_name, op, category, "exact")
                lut_row = cell(report, fmt_name, op, category, full)
                assert lut_row.mean_error == pytest.approx(exact_row.mean_error, rel=1e-12)
                assert lut_row.max_error == pytest.approx(exact_row.max_error, rel=1e-12)


def test_decimated_lut_is_strictly_worse_than_exact_mode(report: bench.Report) -> None:
    """A table coarser than the format is the realistic-hardware error source."""
    coarse16 = f"lut[{bench.coarse_lut_size(LNS16, 8)}]"
    coarse8 = f"lut[{bench.coarse_lut_size(LNS8, 4)}]"
    for fmt_name, coarse in (("LNS16", coarse16), ("LNS8", coarse8)):
        exact_row = cell(report, fmt_name, "add", "random_linear", "exact")
        lut_row = cell(report, fmt_name, "add", "random_linear", coarse)
        assert lut_row.median_error > exact_row.median_error


# ----------------------------------------------------------------------
# Range: clamping happens where it should, and only where it should.
# ----------------------------------------------------------------------


def test_lns16_range_is_fp32_class(report: bench.Report) -> None:
    """LNS16 spans ~256 binades against FP32's 254 normal binades, sitting lower.

    It reaches about two binades further down than FP32's smallest normal and
    stops one log-step short of FP32's maximum -- so on the 1e-38..1e38 test
    categories it never clamps, but it is *not* strictly wider than FP32.
    """
    assert LNS16.max_log - LNS16.min_log == pytest.approx(255.984375)
    assert LNS16.min_value < 1.18e-38, "must reach below FP32's smallest normal"
    assert LNS16.max_value < 3.4028e38, "top end stops just short of FP32's maximum"
    assert LNS16.max_value > 3.3e38

    for category in CATEGORY_NAMES:
        row = cell(report, "LNS16", "convert", category)
        assert row.clamped == 0, f"LNS16 clamped on {category}: {row.worst_case}"


def test_lns8_clamps_on_out_of_range_categories(report: bench.Report) -> None:
    """LNS8's ~4.3e-3 .. 2.3e2 window cannot hold the small/large categories."""
    for category in bench.RANGE_STRESS_CATEGORIES:
        row = cell(report, "LNS8", "convert", category)
        assert row.clamped == row.scored
        # Saturating far below a huge reference gives a relative error of ~1;
        # saturating far above a tiny one gives an enormous relative error.
        assert row.max_error >= 0.99


def test_clamping_report_flags_everything_out_of_range(report: bench.Report) -> None:
    by_key = {(r["format"], r["probe"]): r for r in report.clamping}
    for fmt_name in ("LNS16", "LNS8"):
        beyond_max = by_key[(fmt_name, "beyond-max")]
        beyond_min = by_key[(fmt_name, "beyond-min")]
        assert beyond_max["overflow"] == beyond_max["values"]
        assert beyond_min["underflow"] == beyond_min["values"]
        assert beyond_max["in_range"] == 0 and beyond_min["in_range"] == 0

    # 1e6 and 1e-6 are comfortable for LNS16 and impossible for LNS8.
    assert by_key[("LNS16", "large(1e6)")]["in_range"] == 3
    assert by_key[("LNS16", "small(1e-6)")]["in_range"] == 3
    assert by_key[("LNS8", "large(1e6)")]["overflow"] == 3
    assert by_key[("LNS8", "small(1e-6)")]["underflow"] == 3


def test_fp16_saturates_where_lns16_does_not(report: bench.Report) -> None:
    """FP16's 65504 ceiling makes its error infinite on the 'large' category."""
    fp16_large = cell(report, "FP16", "convert", "large")
    lns16_large = cell(report, "LNS16", "convert", "large")
    assert fp16_large.max_error == math.inf
    assert lns16_large.max_error <= LNS16.max_relative_step_error * 1.001


# ----------------------------------------------------------------------
# Cancellation study.
# ----------------------------------------------------------------------


def test_cancellation_error_grows_as_operands_converge(report: bench.Report) -> None:
    rows = [r for r in report.cancellation if r["format"] == "LNS16"]
    errors = [r["error"] for r in rows]
    assert errors == sorted(errors), "error must increase monotonically as the gap shrinks"
    assert errors[0] < 0.05, "a wide gap must subtract accurately"
    assert errors[-1] == pytest.approx(1.0), "a sub-log-step gap must collapse to zero"


def test_lns8_loses_cancellation_earlier_than_lns16(report: bench.Report) -> None:
    def collapse_gap(fmt_name: str) -> float:
        return max(
            r["relative_gap"]
            for r in report.cancellation
            if r["format"] == fmt_name and r["error"] >= 1.0
        )

    assert collapse_gap("LNS8") > collapse_gap("LNS16")


# ----------------------------------------------------------------------
# Report plumbing: the numbers must survive serialisation and rendering.
# ----------------------------------------------------------------------


def test_discussion_is_generated_and_mentions_the_key_findings(report: bench.Report) -> None:
    text = report.discussion
    for expected in ("Range", "Precision", "Exact vs LUT log-add", "Cancellation", "Clamping"):
        assert expected in text
    assert "n/a" not in text, "a statistic the discussion quotes is missing from the run"


def test_tables_render(report: bench.Report) -> None:
    assert "format" in bench.format_error_table(report.error_rows)
    assert "median err" in bench.format_error_table(report.error_rows, by="operation")
    assert bench.build_readme_section(report).startswith("_Generated by")


def test_outputs_are_written_and_reloadable(report: bench.Report, tmp_path: Path) -> None:
    written = bench.write_outputs(report, tmp_path)
    assert {p.name for p in written} >= {"error_report.csv", "report.json"}
    for path in written:
        assert path.exists() and path.stat().st_size > 0
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["samples"] == SAMPLES
    assert len(payload["error_rows"]) == len(report.error_rows)
    assert payload["formats"]["LNS16"]["bias"] == 16384


def test_cli_runs_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = bench.main(
        ["--samples", "40", "--seed", "1", "--outdir", str(tmp_path), "--quiet"]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "PRECISION SUMMARY" in out
    assert "DISCUSSION" in out
    assert (tmp_path / "error_report.csv").exists()
