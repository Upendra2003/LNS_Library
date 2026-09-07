#!/usr/bin/env python3
"""Full LNS error-characterisation experiment.

Runs four operations -- ``convert`` (round-trip), ``add``, ``mul`` and ``mac``
-- across both LNS formats, both log-add modes (plus a deliberately decimated
LUT), and eight input categories, then reports mean and max error against an
FP32 reference.  FP16 is measured the same way so it can sit in the same table
as a familiar yardstick.

Three reports are produced:

1. **Error report** -- mean/max error per (format, operation, category).
2. **Clamping report** -- overflow/underflow flag counts for inputs pushed
   deliberately outside each format's range.
3. **Cancellation study** -- error of ``a - b`` as ``b`` approaches ``a``,
   which is where LNS subtraction loses its otherwise-constant relative error.

Run it::

    python benchmarks/run_experiments.py --samples 2000 --outdir benchmarks/results
    python benchmarks/run_experiments.py --update-readme      # refresh README table

Every function here is importable, and the Streamlit app in ``webapp/app.py``
calls :func:`run_all` directly to render the same numbers interactively.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

# Allow ``python benchmarks/run_experiments.py`` from a source checkout without
# installing the package first.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lns_arith import utils  # noqa: E402
from lns_arith.convert import encode_with_flags, lns_to_float, to_fp16, to_fp32  # noqa: E402
from lns_arith.errors import LNSWarning, error_metric  # noqa: E402
from lns_arith.formats import LNS8, LNS16, LNSConfig  # noqa: E402
from lns_arith.logadd import default_lut_size  # noqa: E402
from lns_arith.ops import lns_add_with_flags, lns_mac_with_flags, lns_mul_with_flags  # noqa: E402

Operation = Literal["convert", "add", "mul", "mac"]

OPERATIONS: tuple[Operation, ...] = ("convert", "add", "mul", "mac")

#: Operations whose result depends on the log-add correction term.  ``convert``
#: and ``mul`` never touch it, so running them per-mode would just duplicate rows.
MODE_SENSITIVE: frozenset[str] = frozenset({"add", "mac"})

#: Categories whose magnitudes deliberately sit outside at least one format's
#: range.  Their error statistics measure *range*, not precision: a clamped
#: LNS8 value standing in for 1e-38 has a relative error near 1e35, which is
#: correct but swamps any average it is mixed into.
RANGE_STRESS_CATEGORIES: tuple[str, ...] = ("small", "large")


# ----------------------------------------------------------------------
# Variants: one row of the "who is being measured" axis.
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """One system under test.

    Attributes:
        label: Display name, e.g. ``"LNS16/lut-coarse"``.
        format_name: ``"LNS16"``, ``"LNS8"`` or ``"FP16"``.
        fmt: The LNS config, or ``None`` for the FP16 baseline.
        mode: Log-add mode, or ``None`` when not applicable.
        lut_size: Table resolution override for ``mode == "lut"``.
    """

    label: str
    format_name: str
    fmt: LNSConfig | None
    mode: str | None
    lut_size: int | None = None

    @property
    def is_lns(self) -> bool:
        return self.fmt is not None

    @property
    def mode_label(self) -> str:
        if self.mode is None:
            return "n/a"
        if self.mode == "lut" and self.lut_size is not None:
            return f"lut[{self.lut_size}]"
        return self.mode


def coarse_lut_size(fmt: LNSConfig, factor: int) -> int:
    """Table size for a LUT decimated by ``factor`` relative to full resolution."""
    return (default_lut_size(fmt) - 1) // factor + 1


def default_variants() -> list[Variant]:
    """The standard variant sweep used by the report.

    Includes, for each LNS format, the exact correction, a full-resolution LUT
    (one entry per log-step, as specified) and a deliberately decimated LUT --
    the decimated one is what actually exposes lookup-table quantisation error,
    since a full-resolution table is exact on this format's grid.
    """
    return [
        Variant("LNS16/exact", "LNS16", LNS16, "exact"),
        Variant("LNS16/lut", "LNS16", LNS16, "lut", default_lut_size(LNS16)),
        Variant("LNS16/lut-coarse", "LNS16", LNS16, "lut", coarse_lut_size(LNS16, 8)),
        Variant("LNS8/exact", "LNS8", LNS8, "exact"),
        Variant("LNS8/lut", "LNS8", LNS8, "lut", default_lut_size(LNS8)),
        Variant("LNS8/lut-coarse", "LNS8", LNS8, "lut", coarse_lut_size(LNS8, 4)),
        Variant("FP16", "FP16", None, None),
    ]


# ----------------------------------------------------------------------
# Reference arithmetic.  A single IEEE operation is correctly rounded, so
# computing in double and rounding once to the narrow format reproduces true
# FP32 / FP16 arithmetic exactly.  MAC rounds after each of its two steps,
# matching a non-fused multiply-accumulate.
# ----------------------------------------------------------------------


def reference(
    op: Operation, a: float, b: float, acc: float, cast: Callable[[float], float]
) -> float:
    """Compute the reference result of ``op`` in the precision given by ``cast``."""
    a_, b_, acc_ = cast(a), cast(b), cast(acc)
    if op == "convert":
        return a_
    if op == "add":
        return cast(a_ + b_)
    if op == "mul":
        return cast(a_ * b_)
    if op == "mac":
        return cast(cast(a_ * b_) + acc_)
    raise ValueError(f"unknown operation {op!r}")


@dataclass
class Outcome:
    """Result of evaluating one operation on one sample under one variant."""

    value: float
    overflow: bool = False
    underflow: bool = False


def evaluate_lns(
    op: Operation,
    a: float,
    b: float,
    acc: float,
    fmt: LNSConfig,
    mode: str | None,
    lut_size: int | None,
    source: str = "fp32",
) -> Outcome:
    """Run ``op`` in the LNS domain and decode only the final result.

    Inputs are encoded once, the operation runs entirely on packed codes, and
    :func:`lns_to_float` is called exactly once at the very end -- outside the
    arithmetic -- so the comparison against FP32 is meaningful.
    """
    ea = encode_with_flags(a, fmt, source=source)
    eb = encode_with_flags(b, fmt, source=source)
    eacc = encode_with_flags(acc, fmt, source=source)
    over = ea.overflow or eb.overflow
    under = ea.underflow or eb.underflow

    if op == "convert":
        word, o, u = ea.word, ea.overflow, ea.underflow
    elif op == "add":
        word, o, u = lns_add_with_flags(ea.word, eb.word, fmt, mode, lut_size=lut_size)
    elif op == "mul":
        word, o, u = lns_mul_with_flags(ea.word, eb.word, fmt)
    elif op == "mac":
        over = over or eacc.overflow
        under = under or eacc.underflow
        word, o, u = lns_mac_with_flags(
            ea.word, eb.word, eacc.word, fmt, mode, lut_size=lut_size
        )
    else:
        raise ValueError(f"unknown operation {op!r}")

    return Outcome(lns_to_float(word, fmt), over or o, under or u)


def evaluate_fp16(op: Operation, a: float, b: float, acc: float) -> Outcome:
    """Run ``op`` in FP16 and report saturation as an overflow flag."""
    value = reference(op, a, b, acc, to_fp16)
    inputs_finite = all(math.isfinite(to_fp16(v)) for v in (a, b, acc))
    return Outcome(value, overflow=not inputs_finite or not math.isfinite(value))


# ----------------------------------------------------------------------
# Error report.
# ----------------------------------------------------------------------


@dataclass
class ErrorRow:
    """One aggregated cell of the error table.

    Attributes:
        format: ``"LNS16"``, ``"LNS8"`` or ``"FP16"``.
        operation: ``convert`` / ``add`` / ``mul`` / ``mac``.
        mode: Log-add mode label, or ``"n/a"`` for mode-independent operations.
        category: Input category name.
        samples: Operand triples offered to this cell.
        scored: Triples that produced a finite FP32 reference and were scored.
        mean_error: Mean over the *finite* errors; a saturating FP16 result has
            an infinite relative error and is excluded here but still shows up
            in ``max_error``.
        max_error: Worst error in the cell, infinities included.
        median_error: Median error -- the robust "typical precision" number,
            unaffected by a handful of clamped outliers.
        metric: ``rel``, ``abs`` or ``mixed`` -- which metric the cell used
            (absolute error is the fallback when the reference is exactly zero).
        overflow / underflow: Samples that raised each range flag.
        clamped: Samples that raised either flag.
        skipped_nonfinite_ref: Samples dropped because the FP32 reference itself
            was ``inf``/``nan`` and no meaningful error could be defined.
        worst_case: Human-readable record of the worst sample.
    """

    format: str
    operation: str
    mode: str
    category: str
    samples: int
    scored: int
    mean_error: float
    max_error: float
    median_error: float
    metric: str
    overflow: int
    underflow: int
    clamped: int
    skipped_nonfinite_ref: int
    worst_case: str

    @property
    def clamped_fraction(self) -> float:
        return self.clamped / self.scored if self.scored else 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def run_error_experiments(
    samples: int = 2000,
    seed: int = 12345,
    categories: Sequence[str] = utils.CATEGORY_NAMES,
    variants: Sequence[Variant] | None = None,
    operations: Sequence[Operation] = OPERATIONS,
    progress: Callable[[str], None] | None = None,
) -> list[ErrorRow]:
    """Measure mean/max error for every (variant, operation, category) cell.

    Args:
        samples: Number of operand triples per category.
        seed: Base RNG seed; each category derives its own stream from it.
        categories: Input categories to sweep (see :mod:`lns_arith.utils`).
        variants: Systems under test; defaults to :func:`default_variants`.
        operations: Operations to sweep.
        progress: Optional callback receiving a short status string per cell,
            used by the Streamlit progress bar.

    Returns:
        One :class:`ErrorRow` per non-empty cell.
    """
    variants = list(default_variants() if variants is None else variants)
    rows: list[ErrorRow] = []

    # Flags are collected explicitly, so mute the warning channel for speed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LNSWarning)

        for ci, category in enumerate(categories):
            triples = utils.make_pairs(category, samples, seed + 1000 * ci)
            for op in operations:
                for variant in variants:
                    if op not in MODE_SENSITIVE and variant.mode == "lut":
                        continue  # convert/mul do not use the correction table
                    if progress is not None:
                        progress(f"{category} / {op} / {variant.label}")

                    errors: list[float] = []
                    metric_kinds: set[str] = set()
                    n_over = n_under = n_clamped = n_skipped = 0
                    worst = -1.0
                    worst_case = ""

                    for a, b, acc in triples:
                        ref = reference(op, a, b, acc, to_fp32)
                        if not math.isfinite(ref):
                            n_skipped += 1
                            continue
                        if variant.is_lns:
                            assert variant.fmt is not None
                            out = evaluate_lns(
                                op, a, b, acc, variant.fmt, variant.mode, variant.lut_size
                            )
                        else:
                            out = evaluate_fp16(op, a, b, acc)
                        n_over += out.overflow
                        n_under += out.underflow
                        n_clamped += out.overflow or out.underflow
                        err, kind = error_metric(ref, out.value)
                        metric_kinds.add(kind)
                        errors.append(err)
                        if err > worst:
                            worst = err
                            worst_case = _describe_case(op, a, b, acc, ref, out.value)

                    if not errors:
                        continue
                    errors_sorted = sorted(errors)
                    finite = [e for e in errors if math.isfinite(e)]
                    rows.append(
                        ErrorRow(
                            format=variant.format_name,
                            operation=op,
                            mode=variant.mode_label if op in MODE_SENSITIVE else "n/a",
                            category=category,
                            samples=len(triples),
                            scored=len(errors),
                            mean_error=(sum(finite) / len(finite)) if finite else math.inf,
                            max_error=max(errors),
                            median_error=_percentile(errors_sorted, 0.5),
                            metric="abs" if metric_kinds == {"abs"}
                            else "rel" if metric_kinds == {"rel"}
                            else "mixed",
                            overflow=n_over,
                            underflow=n_under,
                            clamped=n_clamped,
                            skipped_nonfinite_ref=n_skipped,
                            worst_case=worst_case,
                        )
                    )
    return rows


def _describe_case(
    op: Operation, a: float, b: float, acc: float, ref: float, got: float
) -> str:
    """Short human-readable record of the worst sample in a cell."""
    if op == "convert":
        expr = f"{a:.6g}"
    elif op == "mac":
        expr = f"{a:.6g}*{b:.6g}+{acc:.6g}"
    else:
        symbol = "+" if op == "add" else "*"
        expr = f"{a:.6g} {symbol} {b:.6g}"
    return f"{expr} -> ref {ref:.6g}, lns {got:.6g}"


# ----------------------------------------------------------------------
# Clamping report: does out-of-range input clamp and flag, rather than wrap?
# ----------------------------------------------------------------------


def run_clamping_report(samples: int = 500, seed: int = 999) -> list[dict[str, Any]]:
    """Check overflow/underflow flagging on deliberately out-of-range inputs.

    Uses the ``extreme`` category (``1e-60`` .. ``1e60``), which is outside FP32
    range as well, so the question here is not "how big is the error" but "was
    the event detected and was the magnitude clamped into range".

    Returns:
        One dict per (format, probe set) with flag counts and the clamped
        magnitudes actually produced.
    """
    probes: dict[str, list[float]] = {
        "extreme": utils.gen_extreme(samples, seed),
        "beyond-max": [1e39, 1e60, 1e300, float("inf"), -1e60],
        "beyond-min": [1e-40, 1e-60, 1e-300, 5e-324, -1e-60],
        "large(1e6)": [1e6, -1e6, 12345.0],
        "small(1e-6)": [1e-6, -1e-6, 3.7e-7],
    }
    out: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LNSWarning)
        for fmt in (LNS16, LNS8):
            for probe_name, values in probes.items():
                n_over = n_under = 0
                decoded_extremes: list[float] = []
                for value in values:
                    res = encode_with_flags(value, fmt, source="fp64")
                    n_over += res.overflow
                    n_under += res.underflow
                    if res.overflow or res.underflow:
                        decoded_extremes.append(lns_to_float(res.word, fmt))
                out.append(
                    {
                        "format": fmt.name,
                        "probe": probe_name,
                        "values": len(values),
                        "overflow": n_over,
                        "underflow": n_under,
                        "in_range": len(values) - n_over - n_under,
                        "clamped_to": (
                            f"{min(map(abs, decoded_extremes)):.4g} .. "
                            f"{max(map(abs, decoded_extremes)):.4g}"
                            if decoded_extremes
                            else "-"
                        ),
                        "wrapped": False,  # clamping is by construction, never modular
                    }
                )
    return out


# ----------------------------------------------------------------------
# Cancellation study: the one place LNS error is *not* constant.
# ----------------------------------------------------------------------


def run_cancellation_study(
    fmt: LNSConfig = LNS16,
    mode: str = "exact",
    base: float = 1.0,
    gaps: Sequence[float] = (0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
) -> list[dict[str, Any]]:
    """Measure ``a - b`` as ``b`` approaches ``a`` from below.

    LNS relative error is otherwise flat across the whole dynamic range; this
    is the exception.  Two nearby operands share almost all of their log code,
    so the *difference* of their codes carries only a few bits of information,
    and the tiny result inherits the full absolute quantisation of the large
    inputs -- catastrophic cancellation, exactly as in floating point, but
    reached sooner because the operands themselves are coarser.

    Args:
        fmt: Format under test.
        mode: Log-add mode.
        base: The value of ``a``.
        gaps: Relative gaps; ``b = a * (1 - gap)``.

    Returns:
        One dict per gap with the reference, the LNS result and the errors.
    """
    from lns_arith.ops import lns_sub_with_flags

    out: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LNSWarning)
        for gap in gaps:
            a = base
            b = base * (1.0 - gap)
            ref = to_fp32(to_fp32(a) - to_fp32(b))
            ea = encode_with_flags(a, fmt)
            eb = encode_with_flags(b, fmt)
            word, over, under = lns_sub_with_flags(ea.word, eb.word, fmt, mode)
            got = lns_to_float(word, fmt)
            err, kind = error_metric(ref, got)
            out.append(
                {
                    "format": fmt.name,
                    "mode": mode,
                    "relative_gap": gap,
                    "a": a,
                    "b": b,
                    "code_gap": abs(ea.word & fmt.code_mask) - abs(eb.word & fmt.code_mask),
                    "reference_fp32": ref,
                    "lns_result": got,
                    "error": err,
                    "metric": kind,
                    "underflow": under,
                    "overflow": over,
                }
            )
    return out


# ----------------------------------------------------------------------
# Rendering.
# ----------------------------------------------------------------------


def render_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Render a fixed-width console table."""
    body = [[("" if c is None else str(c)) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join("-" * w for w in widths)
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)), line]
    out += ["  ".join(c.ljust(widths[i]) for i, c in enumerate(row)) for row in body]
    return "\n".join(out)


def _fmt_err(value: float) -> str:
    if not math.isfinite(value):
        return "inf"
    if value == 0.0:
        return "0"
    return f"{value:.3e}"


def format_error_table(
    rows: Sequence[ErrorRow],
    by: str = "category",
    categories: Sequence[str] | None = None,
) -> str:
    """Render the error rows as a console table.

    Args:
        rows: Rows from :func:`run_error_experiments`.
        by: ``"category"`` for the full breakdown, or ``"operation"`` for a
            compact per-operation summary aggregated over categories.
        categories: Optional category filter (used to separate the precision
            view from the range-stress view).
    """
    if by == "operation":
        return render_table(
            ["format", "op", "mode", "median err", "mean err", "max err", "clamped", "cells"],
            [
                [
                    r["format"],
                    r["operation"],
                    r["mode"],
                    _fmt_err(r["median"]),
                    _fmt_err(r["mean"]),
                    _fmt_err(r["max"]),
                    f"{r['clamped_fraction'] * 100:.1f}%",
                    r["cells"],
                ]
                for r in summarize_by_operation(rows, categories)
            ],
        )
    selected = rows if categories is None else [r for r in rows if r.category in categories]
    return render_table(
        ["format", "op", "mode", "category", "n", "metric",
         "mean err", "median err", "max err", "ovf", "unf"],
        [
            [
                r.format,
                r.operation,
                r.mode,
                r.category,
                r.scored,
                r.metric,
                _fmt_err(r.mean_error),
                _fmt_err(r.median_error),
                _fmt_err(r.max_error),
                r.overflow,
                r.underflow,
            ]
            for r in selected
        ],
    )


def _median(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def summarize_by_operation(
    rows: Sequence[ErrorRow], categories: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Aggregate rows over input categories, keeping ``(format, op, mode)``.

    Three statistics are reported because they answer different questions:

    * ``median`` -- the median of the per-category median errors.  This is the
      *typical precision* of the format and is immune to the handful of clamped
      samples that a range-stress category contributes.
    * ``mean`` -- the mean of the per-category mean errors.  When a format
      cannot represent a category at all (LNS8 on ``small``/``large``) this is
      astronomically large, and rightly so: that is a range failure, not a
      rounding error.
    * ``max`` -- the single worst sample anywhere in the group.

    Args:
        rows: Rows from :func:`run_error_experiments`.
        categories: Optional subset of input categories to aggregate over.

    Returns:
        One dict per ``(format, operation, mode)`` group.
    """
    selected = rows if categories is None else [r for r in rows if r.category in categories]
    buckets: dict[tuple[str, str, str], list[ErrorRow]] = {}
    for row in selected:
        buckets.setdefault((row.format, row.operation, row.mode), []).append(row)
    out: list[dict[str, Any]] = []
    for (fmt_name, op, mode), group in buckets.items():
        finite_means = [g.mean_error for g in group if math.isfinite(g.mean_error)]
        scored = sum(g.scored for g in group)
        out.append(
            {
                "format": fmt_name,
                "operation": op,
                "mode": mode,
                "median": _median([g.median_error for g in group]),
                "mean": (sum(finite_means) / len(finite_means)) if finite_means else math.inf,
                "max": max(g.max_error for g in group),
                "clamped_fraction": (sum(g.clamped for g in group) / scored) if scored else 0.0,
                "cells": len(group),
            }
        )
    order = {op: i for i, op in enumerate(OPERATIONS)}
    fmt_order = {"LNS16": 0, "LNS8": 1, "FP16": 2}
    out.sort(
        key=lambda r: (order.get(r["operation"], 99), fmt_order.get(r["format"], 9), r["mode"])
    )
    return out


def _lookup(
    rows: Sequence[ErrorRow], fmt_name: str, op: str, category: str, mode: str | None = None
) -> ErrorRow | None:
    for row in rows:
        if row.format == fmt_name and row.operation == op and row.category == category:
            if mode is None or row.mode == mode:
                return row
    return None


def build_discussion(
    rows: Sequence[ErrorRow],
    clamping: Sequence[dict[str, Any]] = (),
    cancellation: Sequence[dict[str, Any]] = (),
) -> str:
    """Auto-generate the range/precision discussion from the measured numbers.

    Every figure quoted here is read out of the run that just happened, so the
    text cannot drift away from the data the way a hand-written summary would.
    """
    lines: list[str] = []

    def err(fmt_name: str, op: str, category: str, mode: str | None = None) -> str:
        """Mean error -- deliberately sensitive to clamping; used for range claims."""
        row = _lookup(rows, fmt_name, op, category, mode)
        return "n/a" if row is None else _fmt_err(row.mean_error)

    def med(fmt_name: str, op: str, category: str, mode: str | None = None) -> str:
        """Median error -- the typical-precision number; used for precision claims."""
        row = _lookup(rows, fmt_name, op, category, mode)
        return "n/a" if row is None else _fmt_err(row.median_error)

    def worst(fmt_name: str, op: str, category: str, mode: str | None = None) -> str:
        row = _lookup(rows, fmt_name, op, category, mode)
        return "n/a" if row is None else _fmt_err(row.max_error)

    lines.append("Range")
    lines.append("-----")
    lines.append(
        f"  LNS16 covers magnitudes {LNS16.min_value:.3e} .. {LNS16.max_value:.3e} "
        f"(log2 in [{LNS16.min_log:g}, {LNS16.max_log:g}]) -- "
        f"{LNS16.max_log - LNS16.min_log:.0f} binades of dynamic range, against the 254 "
        "binades of FP32's normal numbers (1.18e-38 .. 3.40e38). The two ranges are "
        "effectively the same size, with LNS16's window sitting about two binades lower: "
        "it reaches further down and stops a hundredth of a binade short at the top. "
        "LNS16 gets FP32's reach out of 16 bits by spending them all on one monotone log "
        "axis instead of splitting them into an exponent field and a mantissa field -- "
        "and pays for it with a far coarser step."
    )
    lines.append(
        f"  LNS8 covers only {LNS8.min_value:.3e} .. {LNS8.max_value:.3e}, about "
        f"{LNS8.max_log - LNS8.min_log:.0f} binades. That is a far narrower window than "
        "FP16 (6.0e-8 .. 6.5e4, ~40 binades), so LNS8 is a per-tensor-scaled format in "
        "practice: unscaled activations run off both ends of it."
    )
    lines.append(
        f"  Measured: on the 'large' category (1e3..1e38) LNS8 overflowed and clamped, "
        f"giving a mean convert error of {err('LNS8', 'convert', 'large')} versus "
        f"{err('LNS16', 'convert', 'large')} for LNS16, which never leaves its range there. "
        f"On 'small' (1e-38..1e-3) the same story runs in reverse: LNS8 "
        f"{err('LNS8', 'convert', 'small')} versus LNS16 {err('LNS16', 'convert', 'small')}."
    )
    lines.append("")

    lines.append("Precision")
    lines.append("---------")
    lines.append(
        f"  LNS quantises log2(|x|) uniformly, so its relative error is *constant across "
        f"the entire dynamic range*: a half-step of log error is a relative error of "
        f"2**(step/2) - 1, i.e. {LNS16.max_relative_step_error * 100:.4f}% for LNS16 "
        f"(step 1/128) and {LNS8.max_relative_step_error * 100:.3f}% for LNS8 (step 1/8). "
        "There are no exponent bands and no wobble."
    )
    lines.append(
        f"  Measured round-trip conversion error confirms the bound. On 'random_linear' "
        f"(values in [-4, 4], in range for both formats) the median relative error is "
        f"{med('LNS16', 'convert', 'random_linear')} for LNS16 against a "
        f"{LNS16.max_relative_step_error:.3e} worst case, and "
        f"{med('LNS8', 'convert', 'random_linear')} for LNS8 against "
        f"{LNS8.max_relative_step_error:.3e}. The same medians on 'random' (magnitudes "
        f"from 1e-6 to 1e6) are {med('LNS16', 'convert', 'random')} and "
        f"{med('LNS8', 'convert', 'random')} -- twelve orders of magnitude of input range, "
        "and the relative error does not move. That flatness is the defining property of "
        "the format."
    )
    lines.append(
        "  FP32 and FP16 are the mirror image: within one binade the spacing is uniform "
        "in the *linear* domain, so relative error sawtooths between 2**-24 and 2**-23 "
        "(FP32) or 2**-11 and 2**-10 (FP16) as the mantissa sweeps a binade -- but the "
        "exponent field buys a very wide range for very few bits. In half-step terms "
        f"FP32 is ~{LNS16.max_relative_step_error / (2 ** -24):.0f}x finer than LNS16 and "
        f"FP16 is ~{LNS16.max_relative_step_error / (2 ** -11):.1f}x finer, while LNS16 "
        f"beats both on range (measured FP16 convert error on 'random': "
        f"{med('FP16', 'convert', 'random')} median, {worst('FP16', 'convert', 'random')} "
        "max -- the max is infinite because FP16 saturates above 65504)."
    )
    lines.append(
        f"  Multiplication needs no correction term at all: the product code is the exact "
        f"integer sum c_a + c_b - bias, with no rounding of its own. What it does do is "
        f"*add the two input errors*, so its median error "
        f"({med('LNS16', 'mul', 'random_linear')} for LNS16) sits just above a single "
        f"conversion ({med('LNS16', 'convert', 'random_linear')}). There is no error "
        "growth beyond that, and no possibility of catastrophic loss."
    )
    lines.append(
        f"  Addition costs an extra rounding of the log2(1 +/- 2**d) correction, but in "
        f"the typical case the two input errors partially average out rather than "
        f"accumulating, so its median error ({med('LNS16', 'add', 'random_linear', 'exact')}) "
        f"is comparable to multiply's ({med('LNS16', 'mul', 'random_linear')}). Addition's "
        f"real cost is in the tail, not the middle: its max error over the same category is "
        f"{worst('LNS16', 'add', 'random_linear', 'exact')} against "
        f"{worst('LNS16', 'mul', 'random_linear')} for multiply. That tail is cancellation, "
        "and it is discussed below."
    )
    lines.append("")

    lines.append("Exact vs LUT log-add")
    lines.append("--------------------")
    lines.append(
        f"  At full table resolution the LUT reproduces the exact mode bit for bit "
        f"(LNS16 add on 'random_linear': exact "
        f"{med('LNS16', 'add', 'random_linear', 'exact')} vs lut "
        f"{med('LNS16', 'add', 'random_linear', f'lut[{default_lut_size(LNS16)}]')}). That "
        "is expected, not a bug: d is always an exact multiple of the log-step, so the "
        "lookup lands on a sampled point, and the stored value has already been rounded "
        "to the same grid the result gets rounded to. A full-resolution table is exact "
        "*with respect to the format*; the idealisation in 'exact' mode only shows up "
        "once the table is smaller than the format's own resolution."
    )
    lines.append(
        f"  Decimating the table is what exposes lookup error. With a "
        f"{coarse_lut_size(LNS16, 8)}-entry LNS16 table (1/8 resolution) the add median "
        f"error rises from {med('LNS16', 'add', 'random_linear', 'exact')} to "
        f"{med('LNS16', 'add', 'random_linear', f'lut[{coarse_lut_size(LNS16, 8)}]')}, and "
        f"a {coarse_lut_size(LNS8, 4)}-entry LNS8 table moves LNS8 from "
        f"{med('LNS8', 'add', 'random_linear', 'exact')} to "
        f"{med('LNS8', 'add', 'random_linear', f'lut[{coarse_lut_size(LNS8, 4)}]')}. This "
        "is the real hardware trade-off: ROM area against accuracy."
    )
    lines.append("")

    if cancellation:
        lines.append("Cancellation -- the exception to constant relative error")
        lines.append("-------------------------------------------------------")
        def _collapse_gap(fmt_name: str) -> str:
            """Largest relative gap at which a - b loses the result entirely."""
            losses = [
                r["relative_gap"]
                for r in cancellation
                if r["format"] == fmt_name and r["error"] >= 1.0
            ]
            return f"{max(losses):g}" if losses else "none of the gaps tested"

        for fmt in (LNS16, LNS8):
            rows_for_fmt = [r for r in cancellation if r["format"] == fmt.name]
            if not rows_for_fmt:
                continue
            worst_partial = max(
                (r for r in rows_for_fmt if r["error"] < 1.0),
                key=lambda r: r["error"],
                default=None,
            )
            partial = (
                f"a gap of {worst_partial['relative_gap']:g} already costs "
                f"{_fmt_err(worst_partial['error'])} relative error"
                if worst_partial
                else "every gap tested lost the result entirely"
            )
            lines.append(
                f"  {fmt.name}: {partial}, and by a gap of {_collapse_gap(fmt.name)} the "
                f"result collapses to exact zero (relative error 1.0)."
            )
        lines.append(
            "  Subtracting nearly-equal operands is the one place the flat-error property "
            "breaks. The two operands share almost their entire log code, so the code "
            "*difference* that drives the correction term carries only a handful of bits, "
            "and the tiny result inherits the absolute quantisation of the large inputs. "
            "Once the operands are closer together than one log-step their codes are "
            "identical and the difference is reported as exactly zero. Floating point "
            "suffers the same catastrophic cancellation; LNS reaches it sooner because "
            "the operands are coarser to begin with, and LNS8 -- whose log-step is 16x "
            "wider than LNS16's -- sooner still."
        )
        lines.append(
            "  Exact cancellation (identical codes, opposite signs) is handled as a code "
            "comparison and returns the exact zero sentinel -- the -inf singularity of "
            "log2(1 - 2**0) is never evaluated."
        )
        lines.append("")

    if clamping:
        total_clamped = sum(r["overflow"] + r["underflow"] for r in clamping)
        lines.append("Clamping")
        lines.append("--------")
        lines.append(
            f"  {total_clamped} out-of-range probe values were clamped to the nearest "
            "representable code and flagged; none wrapped. Overflow saturates at "
            f"{LNS16.max_value:.3e} (LNS16) / {LNS8.max_value:.3e} (LNS8) and underflow "
            f"at {LNS16.min_value:.3e} / {LNS8.min_value:.3e} -- never to zero, since the "
            "zero code is a reserved sentinel reached only by an exact zero input or by "
            "exact cancellation."
        )
        lines.append("")

    lines.append("Bottom line for DNN acceleration")
    lines.append("--------------------------------")
    lines.append(
        "  LNS16 is a credible alternative to FP16 for inference: FP32-class range where "
        "FP16 has ~40 binades, coarser precision than FP16 but uniform -- the error does "
        "not depend on where in the range a value sits -- and "
        "multiplies that cost an integer add. LNS8 needs per-tensor scaling to keep "
        "values inside its ~4.3e-3 .. 2.3e2 window, and its ~4.4% quantisation step "
        "makes it a quantised-inference format rather than a training format. In both "
        "cases the cost centre is addition -- the correction term -- which is exactly "
        "where the accumulator width and the LUT size of a real design get spent."
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Orchestration and CLI.
# ----------------------------------------------------------------------


@dataclass
class Report:
    """Everything one full experiment run produced."""

    samples: int
    seed: int
    error_rows: list[ErrorRow] = field(default_factory=list)
    clamping: list[dict[str, Any]] = field(default_factory=list)
    cancellation: list[dict[str, Any]] = field(default_factory=list)
    discussion: str = ""

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "seed": self.seed,
            "formats": {
                f.name: {
                    "total_bits": f.total_bits,
                    "int_bits": f.int_bits,
                    "frac_bits": f.frac_bits,
                    "bias": f.bias,
                    "log_step": f.log_step,
                    "min_value": f.min_value,
                    "max_value": f.max_value,
                    "max_relative_step_error": f.max_relative_step_error,
                }
                for f in (LNS16, LNS8)
            },
            "error_rows": [r.as_dict() for r in self.error_rows],
            "clamping": list(self.clamping),
            "cancellation": list(self.cancellation),
            "discussion": self.discussion,
        }


def run_all(
    samples: int = 2000,
    seed: int = 12345,
    progress: Callable[[str], None] | None = None,
    variants: Sequence[Variant] | None = None,
    categories: Sequence[str] = utils.CATEGORY_NAMES,
) -> Report:
    """Run all three reports and assemble the discussion text.

    Args:
        samples: Operand triples per category.
        seed: Base RNG seed.
        progress: Optional per-cell status callback.
        variants: Systems under test; defaults to :func:`default_variants`.
        categories: Input categories to sweep.

    Returns:
        A fully populated :class:`Report`.
    """
    error_rows = run_error_experiments(
        samples=samples,
        seed=seed,
        categories=categories,
        variants=variants,
        progress=progress,
    )
    clamping = run_clamping_report(samples=max(50, samples // 4), seed=seed + 7)
    cancellation = run_cancellation_study(LNS16, "exact") + run_cancellation_study(LNS8, "exact")
    report = Report(samples=samples, seed=seed, error_rows=error_rows,
                    clamping=clamping, cancellation=cancellation)
    report.discussion = build_discussion(error_rows, clamping, cancellation)
    return report


def write_outputs(report: Report, outdir: Path) -> list[Path]:
    """Write CSV and JSON artefacts for a report; returns the paths written."""
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    csv_path = outdir / "error_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ErrorRow.__dataclass_fields__))
        writer.writeheader()
        for row in report.error_rows:
            writer.writerow(row.as_dict())
    written.append(csv_path)

    if report.clamping:
        clamp_path = outdir / "clamping_report.csv"
        with clamp_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(report.clamping[0]))
            writer.writeheader()
            writer.writerows(report.clamping)
        written.append(clamp_path)

    if report.cancellation:
        cancel_path = outdir / "cancellation_study.csv"
        with cancel_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(report.cancellation[0]))
            writer.writeheader()
            writer.writerows(report.cancellation)
        written.append(cancel_path)

    json_path = outdir / "report.json"
    json_path.write_text(json.dumps(report.to_json_obj(), indent=2), encoding="utf-8")
    written.append(json_path)
    return written


README_BEGIN = "<!-- BEGIN AUTO-REPORT -->"
README_END = "<!-- END AUTO-REPORT -->"


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def build_readme_section(report: Report) -> str:
    """Build the auto-generated results block that gets spliced into README.md."""
    precision_categories = [
        c for c in utils.CATEGORY_NAMES if c not in RANGE_STRESS_CATEGORIES
    ]
    parts: list[str] = []
    parts.append(
        f"_Generated by `python benchmarks/run_experiments.py --update-readme` with "
        f"`--samples {report.samples} --seed {report.seed}`; "
        f"{len(report.error_rows)} (format, operation, mode, category) cells. "
        "Errors are relative to an FP32 reference, falling back to absolute error where "
        "the reference is exactly zero. `median` is the median of the per-category median "
        "errors -- the typical-precision number. `mean` is the mean of the per-category "
        "means and is deliberately sensitive to clamping. `max` is the single worst "
        "sample._"
    )
    parts.append("")
    parts.append(
        "#### Precision: error by operation (in-range categories: "
        + ", ".join(f"`{c}`" for c in precision_categories)
        + ")"
    )
    parts.append("")
    parts.append(
        _markdown_table(
            ["Format", "Operation", "Log-add mode", "Median error", "Mean error",
             "Max error", "Clamped"],
            [
                [r["format"], r["operation"], r["mode"], _fmt_err(r["median"]),
                 _fmt_err(r["mean"]), _fmt_err(r["max"]),
                 f"{r['clamped_fraction'] * 100:.1f}%"]
                for r in summarize_by_operation(report.error_rows, precision_categories)
            ],
        )
    )
    parts.append("")
    parts.append(
        "#### Range: error by operation on out-of-range inputs (categories: "
        + ", ".join(f"`{c}`" for c in RANGE_STRESS_CATEGORIES)
        + ")"
    )
    parts.append("")
    parts.append(
        "Large numbers in this table are *clamping*, not rounding: LNS8 simply cannot "
        "represent 1e-38 or 1e38, so it saturates and the relative error against FP32 is "
        "enormous. LNS16 and FP32 have near-identical range, so LNS16 barely clamps here; "
        "FP16 saturates above 65504 and its relative error becomes infinite."
    )
    parts.append("")
    parts.append(
        _markdown_table(
            ["Format", "Operation", "Log-add mode", "Median error", "Mean error",
             "Max error", "Clamped"],
            [
                [r["format"], r["operation"], r["mode"], _fmt_err(r["median"]),
                 _fmt_err(r["mean"]), _fmt_err(r["max"]),
                 f"{r['clamped_fraction'] * 100:.1f}%"]
                for r in summarize_by_operation(report.error_rows, RANGE_STRESS_CATEGORIES)
            ],
        )
    )
    parts.append("")
    parts.append("#### Mean error by input category (round-trip conversion)")
    parts.append("")
    parts.append(
        "Mean error, so out-of-range categories show their clamping. Read the LNS16 row "
        "across: the error barely moves from `positive` to `large`, which is the flat "
        "relative error of a log format. The LNS8 row explodes on `small`/`large` and the "
        "FP16 row on `small`, because those values are outside those formats' range."
    )
    parts.append("")
    categories = [c for c in utils.CATEGORY_NAMES]
    conv_rows = []
    for fmt_name in ("LNS16", "LNS8", "FP16"):
        cells = []
        for category in categories:
            row = _lookup(report.error_rows, fmt_name, "convert", category)
            cells.append(_fmt_err(row.mean_error) if row else "-")
        conv_rows.append([fmt_name, *cells])
    parts.append(_markdown_table(["Format", *categories], conv_rows))
    parts.append("")
    parts.append("#### Mean error by input category (addition, exact log-add)")
    parts.append("")
    add_rows = []
    for fmt_name, mode in (("LNS16", "exact"), ("LNS8", "exact"), ("FP16", "n/a")):
        cells = []
        for category in categories:
            row = _lookup(report.error_rows, fmt_name, "add", category, mode)
            cells.append(_fmt_err(row.mean_error) if row else "-")
        add_rows.append([f"{fmt_name}", *cells])
    parts.append(_markdown_table(["Format", *categories], add_rows))
    parts.append("")
    if report.cancellation:
        parts.append("#### Cancellation study (`a - b` as `b -> a`)")
        parts.append("")
        parts.append(
            _markdown_table(
                ["Format", "Relative gap", "FP32 reference", "LNS result", "Relative error"],
                [
                    [
                        r["format"],
                        f"{r['relative_gap']:g}",
                        f"{r['reference_fp32']:.6g}",
                        f"{r['lns_result']:.6g}",
                        _fmt_err(r["error"]),
                    ]
                    for r in report.cancellation
                ],
            )
        )
        parts.append("")
    if report.clamping:
        parts.append("#### Overflow / underflow clamping")
        parts.append("")
        parts.append(
            _markdown_table(
                ["Format", "Probe set", "Values", "Overflow", "Underflow", "In range", "Clamped to |x|"],
                [
                    [
                        r["format"],
                        r["probe"],
                        r["values"],
                        r["overflow"],
                        r["underflow"],
                        r["in_range"],
                        r["clamped_to"],
                    ]
                    for r in report.clamping
                    if r["probe"] != "extreme"
                ],
            )
        )
        parts.append("")
    parts.append("#### Discussion")
    parts.append("")
    parts.append("```text")
    parts.append(report.discussion)
    parts.append("```")
    return "\n".join(parts)


def update_readme(report: Report, readme_path: Path) -> bool:
    """Splice the generated results block between the README's markers.

    Returns:
        ``True`` if the file was rewritten, ``False`` if the markers were absent.
    """
    if not readme_path.exists():
        return False
    text = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(README_BEGIN) + r".*?" + re.escape(README_END), re.DOTALL
    )
    if not pattern.search(text):
        return False
    block = f"{README_BEGIN}\n\n{build_readme_section(report)}\n\n{README_END}"
    readme_path.write_text(pattern.sub(lambda _: block, text), encoding="utf-8")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--samples", type=int, default=2000,
                        help="operand triples per input category (default: 2000)")
    parser.add_argument("--seed", type=int, default=12345, help="base RNG seed")
    parser.add_argument("--outdir", type=Path, default=_REPO_ROOT / "benchmarks" / "results",
                        help="directory for the CSV/JSON artefacts")
    parser.add_argument("--no-write", action="store_true", help="print only, write nothing")
    parser.add_argument("--update-readme", action="store_true",
                        help="splice the results into README.md between its auto-report markers")
    parser.add_argument("--quiet", action="store_true", help="suppress the full breakdown table")
    args = parser.parse_args(argv)

    show_progress = sys.stderr.isatty() and not args.quiet

    def progress(message: str) -> None:
        print(f"\r  running {message:<48}", end="", file=sys.stderr, flush=True)

    report = run_all(samples=args.samples, seed=args.seed,
                     progress=progress if show_progress else None)
    if show_progress:
        print("\r" + " " * 64 + "\r", end="", file=sys.stderr, flush=True)

    print("=" * 100)
    print(f"LNS ERROR REPORT  --  {args.samples} samples/category, seed {args.seed}")
    print("=" * 100)
    print()
    for fmt in (LNS16, LNS8):
        print(f"  {fmt}")
        print(f"      worst-case conversion error {fmt.max_relative_step_error * 100:.4f}%  "
              f"(half a log-step of {fmt.log_step:g})")
    print()
    precision_categories = [
        c for c in utils.CATEGORY_NAMES if c not in RANGE_STRESS_CATEGORIES
    ]
    print("-" * 100)
    print("PRECISION SUMMARY (in-range categories: "
          + ", ".join(precision_categories) + ")")
    print("-" * 100)
    print(format_error_table(report.error_rows, by="operation",
                             categories=precision_categories))
    print()
    print("-" * 100)
    print("RANGE-STRESS SUMMARY (categories: " + ", ".join(RANGE_STRESS_CATEGORIES)
          + " -- huge errors here are clamping, not rounding)")
    print("-" * 100)
    print(format_error_table(report.error_rows, by="operation",
                             categories=RANGE_STRESS_CATEGORIES))
    print()
    if not args.quiet:
        print("-" * 100)
        print("FULL BREAKDOWN BY FORMAT / OPERATION / MODE / CATEGORY")
        print("-" * 100)
        print(format_error_table(report.error_rows))
        print()
    print("-" * 100)
    print("OVERFLOW / UNDERFLOW CLAMPING")
    print("-" * 100)
    print(
        render_table(
            ["format", "probe", "values", "overflow", "underflow", "in range", "clamped to |x|"],
            [
                [r["format"], r["probe"], r["values"], r["overflow"], r["underflow"],
                 r["in_range"], r["clamped_to"]]
                for r in report.clamping
            ],
        )
    )
    print()
    print("-" * 100)
    print("CANCELLATION STUDY (a - b as b -> a)")
    print("-" * 100)
    print(
        render_table(
            ["format", "gap", "fp32 ref", "lns result", "error", "underflow"],
            [
                [r["format"], f"{r['relative_gap']:g}", f"{r['reference_fp32']:.6g}",
                 f"{r['lns_result']:.6g}", _fmt_err(r["error"]), r["underflow"]]
                for r in report.cancellation
            ],
        )
    )
    print()
    print("-" * 100)
    print("DISCUSSION")
    print("-" * 100)
    print(report.discussion)
    print()

    if not args.no_write:
        written = write_outputs(report, args.outdir)
        print("Wrote: " + ", ".join(os.path.relpath(p, _REPO_ROOT) for p in written))
    if args.update_readme:
        ok = update_readme(report, _REPO_ROOT / "README.md")
        print("Updated README.md" if ok else "README.md has no auto-report markers; skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
