"""Streamlit simulator for ``lns_arith`` -- the full-integration front end.

This app imports the real package and calls the real arithmetic, so what you
see on screen is exactly what the library computes.  It is the source of truth;
``webapp/simulator.html`` is a dependency-free JavaScript mirror of the same
pipeline for quick sharing.

Run it from the repository root::

    pip install -e ".[web]"
    streamlit run webapp/app.py

Three tabs:

``Walkthrough``
    Encode two (or three) numbers, run one operation, and show every step:
    the sign/log encoding of each input, the log-domain gap ``d``, which
    correction-term branch was taken, the value of the correction, the raw and
    clamped result codes, and the decoded float -- next to FP32 and FP16
    references with the errors computed.

``Batch experiment``
    Runs ``benchmarks/run_experiments.py`` on demand at an adjustable sample
    size and renders the mean/max error tables, comparison charts, the clamping
    and cancellation studies, and the auto-generated discussion.

``Format reference``
    The bit layouts, ranges and precision bounds, plus a plot of the log-add
    correction function against the quantised LUT that approximates it.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "benchmarks")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_experiments as bench  # noqa: E402
from lns_arith import (  # noqa: E402
    LNS8,
    LNS16,
    LNSConfig,
    code_of,
    encode_with_flags,
    lns_add_with_flags,
    lns_mac_with_flags,
    lns_mul_with_flags,
    lns_sub_with_flags,
    lns_to_float,
    log_of_code,
    sign_of,
    to_fp16,
    to_fp32,
)
from lns_arith.errors import error_metric  # noqa: E402
from lns_arith.logadd import default_lut_size, exact_mode, get_lut  # noqa: E402
from lns_arith.utils import CATEGORY_NAMES  # noqa: E402

st.set_page_config(page_title="LNS arithmetic simulator", page_icon="🔢", layout="wide")

FORMATS: dict[str, LNSConfig] = {"LNS16": LNS16, "LNS8": LNS8}

OPERATIONS = {
    "convert (round-trip)": "convert",
    "add  (a + b)": "add",
    "subtract  (a - b)": "sub",
    "multiply  (a * b)": "mul",
    "MAC  (a * b + acc)": "mac",
}

PRESETS: dict[str, tuple[str, str, str]] = {
    "typical values": ("3.5", "-1.25", "0.5"),
    "powers of two (exact)": ("8", "0.25", "2"),
    "zero + value": ("0", "2.75", "1"),
    "both zero": ("0", "0", "0"),
    "exact cancellation": ("2.5", "-2.5", "0"),
    "near cancellation": ("1.0", "-0.999", "0"),
    "very small (~1e-6)": ("1e-6", "3e-7", "0"),
    "very large (~1e6)": ("1e6", "2.5e5", "0"),
    "beyond LNS8 range": ("5000", "-0.0001", "1"),
    "negative pair": ("-6.25", "-2.5", "-1.5"),
}


# ----------------------------------------------------------------------
# Small helpers.
# ----------------------------------------------------------------------


def parse_float(text: str, label: str) -> float | None:
    """Parse a user-entered number, showing an inline error instead of raising."""
    try:
        value = float(text.strip())
    except (TypeError, ValueError):
        st.error(f"`{label}` is not a number: {text!r}")
        return None
    if math.isnan(value):
        st.error(f"`{label}` is NaN, which has no LNS representation.")
        return None
    return value


def fmt_float(x: float) -> str:
    """Compact, readable float formatting for the trace tables."""
    if x == 0.0:
        return "0"
    if not math.isfinite(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.8g}"


def fmt_error(x: float) -> str:
    if not math.isfinite(x):
        return "inf"
    return "0" if x == 0.0 else f"{x:.4e}"


def binary_word(word: int, fmt: LNSConfig) -> str:
    """Render a packed word as ``sign | code`` binary, the way a datasheet would."""
    sign = sign_of(word, fmt)
    code = code_of(word, fmt)
    return f"{sign:b} {code:0{fmt.code_bits}b}"


def encoding_table(
    values: dict[str, float], fmt: LNSConfig, source: str
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    """Encode each named input and describe the result row by row."""
    rows = []
    encoded: dict[str, Any] = {}
    flagged = False
    for name, value in values.items():
        result = encode_with_flags(value, fmt, source=source)
        encoded[name] = result
        flagged = flagged or result.overflow or result.underflow
        code = code_of(result.word, fmt)
        is_zero_word = code == fmt.zero_code
        rows.append(
            {
                "operand": name,
                f"as {source}": fmt_float(result.source_value),
                "sign": "-" if sign_of(result.word, fmt) else "+",
                "log2|x|": "-inf (zero)" if is_zero_word else f"{result.log2_magnitude:+.7f}",
                "code": code,
                "L = (code-bias)/scale": (
                    "reserved zero" if is_zero_word else f"{log_of_code(code, fmt):+.7f}"
                ),
                "word (bin)": binary_word(result.word, fmt),
                "word (hex)": f"0x{result.word:0{(fmt.total_bits + 3) // 4}X}",
                "decodes to": fmt_float(lns_to_float(result.word, fmt)),
                "flag": (
                    "OVERFLOW -> clamped"
                    if result.overflow
                    else "UNDERFLOW -> clamped"
                    if result.underflow
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows), encoded, flagged


def run_operation(
    op: str,
    words: dict[str, int],
    fmt: LNSConfig,
    mode: str,
    lut_size: int | None,
) -> tuple[int, bool, bool, dict[str, Any]]:
    """Dispatch one traced operation; returns ``(word, overflow, underflow, trace)``."""
    trace: dict[str, Any] = {}
    if op == "convert":
        return words["a"], False, False, {
            "op": "convert",
            "branch": "encode / decode round trip",
            "steps": [
                "conversion is the encode step shown above; no arithmetic is performed",
                "the decoded value below is the round-trip result",
            ],
        }
    if op == "add":
        result = lns_add_with_flags(words["a"], words["b"], fmt, mode,
                                    lut_size=lut_size, trace=trace)
    elif op == "sub":
        result = lns_sub_with_flags(words["a"], words["b"], fmt, mode,
                                    lut_size=lut_size, trace=trace)
    elif op == "mul":
        result = lns_mul_with_flags(words["a"], words["b"], fmt, trace=trace)
    elif op == "mac":
        result = lns_mac_with_flags(words["a"], words["b"], words["acc"], fmt, mode,
                                    lut_size=lut_size, trace=trace)
    else:
        raise ValueError(f"unknown operation {op!r}")
    return result.word, result.overflow, result.underflow, trace


def float_reference(op: str, a: float, b: float, acc: float, cast) -> float:
    """The FP reference for the operation, including subtraction."""
    if op == "sub":
        a_, b_ = cast(a), cast(b)
        return cast(a_ - b_)
    return bench.reference(op, a, b, acc, cast)


# ----------------------------------------------------------------------
# Sidebar: the format and mode selection shared by every tab.
# ----------------------------------------------------------------------

st.title("Logarithmic Number System arithmetic simulator")
st.caption(
    "Every number below is computed by the `lns_arith` package itself -- inputs are "
    "encoded once, the operation runs entirely on packed LNS codes, and the result is "
    "decoded only at the very end for comparison."
)

with st.sidebar:
    st.header("Configuration")
    fmt_name = st.selectbox("Target LNS format", list(FORMATS), index=0)
    fmt = FORMATS[fmt_name]
    source = st.selectbox(
        "Source float format", ["fp32", "fp16"], index=0,
        help="Inputs are rounded to this IEEE format before the logarithm is taken.",
    )
    mode = st.selectbox(
        "Log-add correction mode", ["exact", "lut"], index=0,
        help=(
            "'exact' evaluates log2(1 +/- 2**d) in double precision -- a software "
            "idealisation. 'lut' reads a table quantised to this format's log grid, "
            "which is what hardware does."
        ),
    )

    full_size = default_lut_size(fmt)
    lut_size: int | None = None
    if mode == "lut":
        decimate = st.select_slider(
            "LUT resolution", options=[1, 2, 4, 8, 16, 32],
            value=1, format_func=lambda d: f"1/{d} of format step" if d > 1 else "full",
            help=(
                "A full-resolution table is exact on this format's grid, because d is "
                "always a whole number of log-steps. Decimating it is what produces "
                "realistic lookup-table error."
            ),
        )
        lut_size = (full_size - 1) // decimate + 1
        st.caption(f"Table: {lut_size} entries over d in [-{fmt.lut_domain:g}, 0]")

    st.divider()
    st.markdown(
        f"**{fmt.name}** — 1 sign + {fmt.int_bits} int + {fmt.frac_bits} frac bits  \n"
        f"bias **{fmt.bias}**, log-step **1/{fmt.scale}** = {fmt.log_step:g}  \n"
        f"magnitudes **{fmt.min_value:.3e} … {fmt.max_value:.3e}**  \n"
        f"worst-case rounding **{fmt.max_relative_step_error * 100:.4f}%**"
    )
    st.caption(
        "Code 0 is reserved as the exact-zero sentinel, which costs one log-step at the "
        "bottom of the range and buys an exact zero — worth it for DNN tensors, which "
        "are full of them."
    )

walkthrough_tab, batch_tab, reference_tab = st.tabs(
    ["Walkthrough", "Batch experiment", "Format reference"]
)


# ----------------------------------------------------------------------
# Tab 1: single-operation walkthrough.
# ----------------------------------------------------------------------


for _key, _default in (("a_text", "3.5"), ("b_text", "-1.25"), ("acc_text", "0.5")):
    st.session_state.setdefault(_key, _default)


def _apply_preset() -> None:
    choice = st.session_state.get("preset")
    if choice in PRESETS:
        a, b, acc = PRESETS[choice]
        st.session_state["a_text"] = a
        st.session_state["b_text"] = b
        st.session_state["acc_text"] = acc


with walkthrough_tab:
    st.subheader("Step-by-step LNS pipeline")

    controls = st.columns([2, 2, 1, 1, 1])
    with controls[0]:
        st.selectbox("Preset", list(PRESETS), key="preset", on_change=_apply_preset)
    with controls[1]:
        op_label = st.selectbox("Operation", list(OPERATIONS), index=1)
        op = OPERATIONS[op_label]
    with controls[2]:
        st.text_input("a", key="a_text")
    with controls[3]:
        st.text_input("b", key="b_text", disabled=op == "convert")
    with controls[4]:
        st.text_input("acc", key="acc_text", disabled=op != "mac")

    a = parse_float(st.session_state["a_text"], "a")
    b = parse_float(st.session_state["b_text"], "b") if op != "convert" else 0.0
    acc = parse_float(st.session_state["acc_text"], "acc") if op == "mac" else 0.0

    if a is None or b is None or acc is None:
        st.stop()

    operands = {"a": a}
    if op != "convert":
        operands["b"] = b
    if op == "mac":
        operands["acc"] = acc

    st.markdown("#### 1 · Encode the inputs")
    st.caption(
        f"`code = round(log2(|x|) × {fmt.scale} + {fmt.bias})`, clamped to "
        f"[{fmt.min_code}, {fmt.max_code}]. Code 0 is the reserved zero."
    )
    enc_df, encoded, enc_flagged = encoding_table(operands, fmt, source)
    st.dataframe(enc_df, hide_index=True, width="stretch")
    if enc_flagged:
        st.warning(
            "At least one input fell outside the format's range and was clamped to the "
            "nearest representable code. The library flags this rather than wrapping."
        )

    words = {name: result.word for name, result in encoded.items()}
    words.setdefault("b", 0)
    words.setdefault("acc", 0)

    st.markdown("#### 2 · Run the operation in the log domain")
    result_word, overflow, underflow, trace = run_operation(op, words, fmt, mode, lut_size)

    branch = trace.get("branch", "-")
    formula = trace.get("formula")
    meta = st.columns(4)
    meta[0].metric("Branch taken", branch)
    meta[1].metric("Correction formula", formula or "none needed")
    meta[2].metric(
        "ΔL (log-domain gap)",
        f"{trace['d']:+.7f}" if "d" in trace else "-",
    )
    meta[3].metric(
        "Correction term",
        f"{trace['correction']:+.7f}" if "correction" in trace else "-",
    )

    st.code("\n".join(trace.get("steps", ["(no steps recorded)"])), language="text")

    if "d_code" in trace:
        st.caption(
            f"ΔL is exact: it is the integer code difference {trace['d_code']} divided by "
            f"the scale {fmt.scale}. That is why a full-resolution LUT lookup can never "
            "miss an entry."
        )

    st.markdown("#### 3 · Result and comparison")
    lns_value = lns_to_float(result_word, fmt)
    ref32 = float_reference(op, a, b, acc, to_fp32)
    ref16 = float_reference(op, a, b, acc, to_fp16)

    result_cols = st.columns(3)
    with result_cols[0]:
        st.markdown("**LNS result**")
        st.metric(f"{fmt.name} decoded", fmt_float(lns_value))
        st.code(
            f"word  = 0x{result_word:0{(fmt.total_bits + 3) // 4}X}\n"
            f"bits  = {binary_word(result_word, fmt)}\n"
            f"sign  = {sign_of(result_word, fmt)}\n"
            f"code  = {code_of(result_word, fmt)}"
            + (
                "  (reserved zero)"
                if code_of(result_word, fmt) == fmt.zero_code
                else f"\nL     = {log_of_code(code_of(result_word, fmt), fmt):+.7f}"
            ),
            language="text",
        )
    with result_cols[1]:
        st.markdown("**FP32 reference**")
        err32, kind32 = error_metric(ref32, lns_value)
        st.metric("value", fmt_float(ref32))
        st.metric(
            f"{'relative' if kind32 == 'rel' else 'absolute'} error",
            fmt_error(err32),
            help="Relative error, falling back to absolute error when the reference is 0.",
        )
    with result_cols[2]:
        st.markdown("**FP16 reference**")
        err16, kind16 = error_metric(ref16, lns_value)
        st.metric("value", fmt_float(ref16))
        st.metric(
            f"{'relative' if kind16 == 'rel' else 'absolute'} error",
            fmt_error(err16),
        )

    if overflow:
        st.error("Result overflowed the format range and was clamped to the maximum code.")
    if underflow:
        st.warning(
            "Result underflowed and was clamped to the smallest nonzero code. This is the "
            "near-cancellation regime: the result is not flushed to zero, because zero is "
            "a reserved exact value."
        )
    if branch == "exact-cancellation":
        st.info(
            "The two operands had identical codes and opposite signs, so they cancel "
            "exactly. The `-inf` singularity of `log2(1 - 2**0)` is handled by comparing "
            "codes, never by evaluating the correction function."
        )

    with st.expander("Compare every configuration for these inputs"):
        st.caption(
            "The same operands run through both formats and every log-add mode. This is "
            "the quickest way to see where the format's range or the table's resolution "
            "starts to matter."
        )
        comparison_rows = []
        for other_name, other_fmt in FORMATS.items():
            other_words = {
                name: encode_with_flags(value, other_fmt, source=source).word
                for name, value in operands.items()
            }
            other_words.setdefault("b", 0)
            other_words.setdefault("acc", 0)
            full = default_lut_size(other_fmt)
            configs: list[tuple[str, str, int | None]] = [("exact", "exact", None)]
            if op in ("add", "sub", "mac"):
                configs += [
                    (f"lut (full, {full})", "lut", full),
                    (f"lut (1/8, {(full - 1) // 8 + 1})", "lut", (full - 1) // 8 + 1),
                ]
            for label, cfg_mode, cfg_size in configs:
                word, ovf, unf, _ = run_operation(op, other_words, other_fmt,
                                                  cfg_mode, cfg_size)
                value = lns_to_float(word, other_fmt)
                error, kind = error_metric(ref32, value)
                comparison_rows.append(
                    {
                        "format": other_name,
                        "log-add mode": label,
                        "result": fmt_float(value),
                        f"{kind} error vs FP32": fmt_error(error),
                        "flags": " ".join(
                            f for f, on in (("overflow", ovf), ("underflow", unf)) if on
                        )
                        or "-",
                    }
                )
        st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, width="stretch")


# ----------------------------------------------------------------------
# Tab 2: batch experiment.
# ----------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def cached_experiment(samples: int, seed: int, categories: tuple[str, ...]) -> dict[str, Any]:
    """Run the benchmark and return plain data (so Streamlit can cache it)."""
    report = bench.run_all(samples=samples, seed=seed, categories=list(categories))
    return report.to_json_obj()


with batch_tab:
    st.subheader("Full error experiment")
    st.caption(
        "This is `benchmarks/run_experiments.py` run in-process: four operations × two "
        "LNS formats × exact/full-LUT/decimated-LUT × the input categories you choose, "
        "scored against an FP32 reference, with FP16 measured the same way for scale."
    )

    settings = st.columns([1, 1, 3])
    with settings[0]:
        samples = st.slider("Samples per category", 50, 5000, 500, step=50)
    with settings[1]:
        seed = st.number_input("Seed", value=12345, step=1)
    with settings[2]:
        categories = st.multiselect(
            "Input categories", list(CATEGORY_NAMES), default=list(CATEGORY_NAMES)
        )

    if not categories:
        st.info("Select at least one input category.")
    elif st.button("Run experiment", type="primary"):
        with st.spinner(f"Running {samples} samples per category…"):
            st.session_state["batch"] = cached_experiment(
                int(samples), int(seed), tuple(categories)
            )

    payload = st.session_state.get("batch")
    if payload is None:
        st.info("Press **Run experiment** to generate the report.")
    else:
        rows = pd.DataFrame(payload["error_rows"])
        st.markdown("#### Mean / max error by format, operation, mode and category")
        st.caption(
            "`median_error` is the typical-precision number. `mean_error` is deliberately "
            "sensitive to clamping — where a format cannot represent a category at all, "
            "it explodes, and that is a range failure rather than a rounding error."
        )
        st.dataframe(
            rows[
                ["format", "operation", "mode", "category", "scored", "metric",
                 "median_error", "mean_error", "max_error", "clamped"]
            ],
            hide_index=True,
            width="stretch",
            height=380,
        )

        in_range = [c for c in categories if c not in bench.RANGE_STRESS_CATEGORIES]

        st.markdown("#### LNS8 vs LNS16 across operations")
        st.caption(
            "Median relative error on in-range categories, log scale. The gap between the "
            "two formats is the 16× difference in their log-steps, and it is the same at "
            "every operation — which is the point of LNS."
        )
        by_op = rows[
            rows["category"].isin(in_range) & rows["mode"].isin(["exact", "n/a"])
        ]
        if by_op.empty:
            st.info("No in-range categories selected, so there is nothing to compare.")
        else:
            grouped = (
                by_op.groupby(["operation", "format"], as_index=False)["median_error"]
                .median()
            )
            order = [o for o in bench.OPERATIONS]
            chart = (
                alt.Chart(grouped)
                .mark_bar()
                .encode(
                    x=alt.X("operation:N", sort=order, title="operation"),
                    xOffset=alt.XOffset("format:N", sort=["LNS16", "LNS8", "FP16"]),
                    y=alt.Y(
                        "median_error:Q",
                        title="median relative error (log scale)",
                        scale=alt.Scale(type="log"),
                    ),
                    color=alt.Color("format:N", sort=["LNS16", "LNS8", "FP16"]),
                    tooltip=["format", "operation", alt.Tooltip("median_error:Q", format=".3e")],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, width="stretch")

        st.markdown("#### Error by input category")
        st.caption(
            "Conversion error per category. Flat bars across categories are the whole "
            "story of LNS: the relative error does not care how big the number is — "
            "until the number leaves the format's range, at which point the bar shoots up "
            "because the value has been clamped."
        )
        by_cat = rows[(rows["operation"] == "convert")]
        chart2 = (
            alt.Chart(by_cat)
            .mark_bar()
            .encode(
                x=alt.X("category:N", sort=list(categories), title="input category"),
                xOffset=alt.XOffset("format:N", sort=["LNS16", "LNS8", "FP16"]),
                y=alt.Y(
                    "median_error:Q",
                    title="median error (log scale)",
                    scale=alt.Scale(type="log"),
                ),
                color=alt.Color("format:N", sort=["LNS16", "LNS8", "FP16"]),
                tooltip=[
                    "format", "category",
                    alt.Tooltip("median_error:Q", format=".3e"),
                    alt.Tooltip("max_error:Q", format=".3e"),
                    "clamped",
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(chart2, width="stretch")

        st.markdown("#### Exact vs LUT log-add")
        lut_rows = rows[rows["operation"].isin(["add", "mac"])]
        if not lut_rows.empty:
            lut_grouped = (
                lut_rows[lut_rows["category"].isin(in_range)]
                .groupby(["format", "mode"], as_index=False)["median_error"]
                .median()
            )
            chart3 = (
                alt.Chart(lut_grouped)
                .mark_bar()
                .encode(
                    x=alt.X("mode:N", title="log-add mode"),
                    xOffset=alt.XOffset("format:N"),
                    y=alt.Y(
                        "median_error:Q",
                        title="median error (log scale)",
                        scale=alt.Scale(type="log"),
                    ),
                    color=alt.Color("format:N", sort=["LNS16", "LNS8"]),
                    tooltip=["format", "mode", alt.Tooltip("median_error:Q", format=".3e")],
                )
                .properties(height=280)
            )
            st.altair_chart(chart3, width="stretch")
            st.caption(
                "The full-resolution table matches exact mode exactly — a table sampled at "
                "the format's own log-step cannot be wrong, because ΔL always lands on an "
                "entry. Only the decimated table shows lookup error."
            )

        left, right = st.columns(2)
        with left:
            st.markdown("#### Overflow / underflow clamping")
            st.dataframe(pd.DataFrame(payload["clamping"]), hide_index=True,
                         width="stretch")
        with right:
            st.markdown("#### Cancellation study")
            cancel = pd.DataFrame(payload["cancellation"])
            st.dataframe(
                cancel[["format", "relative_gap", "reference_fp32", "lns_result", "error"]],
                hide_index=True,
                width="stretch",
            )

        st.markdown("#### Discussion (auto-generated from this run)")
        st.code(payload["discussion"], language="text")

        downloads = st.columns(2)
        downloads[0].download_button(
            "Download error report (CSV)",
            rows.to_csv(index=False).encode("utf-8"),
            file_name="lns_error_report.csv",
            mime="text/csv",
        )
        downloads[1].download_button(
            "Download full report (JSON)",
            json.dumps(payload, indent=2).encode("utf-8"),
            file_name="lns_report.json",
            mime="application/json",
        )


# ----------------------------------------------------------------------
# Tab 3: format reference.
# ----------------------------------------------------------------------

with reference_tab:
    st.subheader("Format reference")

    spec_rows = []
    for name, config in FORMATS.items():
        spec_rows.append(
            {
                "format": name,
                "total bits": config.total_bits,
                "layout": f"1 sign + {config.int_bits} int + {config.frac_bits} frac",
                "bias": config.bias,
                "log-step": f"1/{config.scale} = {config.log_step:g}",
                "log2 range": f"[{config.min_log:g}, {config.max_log:g}]",
                "magnitude range": f"{config.min_value:.4e} … {config.max_value:.4e}",
                "binades": f"{config.max_log - config.min_log:.2f}",
                "worst-case rel. error": f"{config.max_relative_step_error * 100:.4f}%",
                "codes": f"0 = zero sentinel, 1…{config.max_code} = magnitudes",
            }
        )
    st.dataframe(pd.DataFrame(spec_rows), hide_index=True, width="stretch")

    st.markdown(
        """
**Encoding.** `L = (code - bias) / 2**frac_bits` is the base-2 logarithm of the
magnitude; the sign lives in a separate bit. Encoding rounds `log2(|x|) * scale + bias`
to the nearest integer and clamps into range — it never wraps.

**Zero.** Code 0 is reserved as an exact-zero sentinel rather than used as the
most-negative magnitude slot. That costs one log-step of range at the bottom and buys
an exact zero, which matters for DNN tensors full of ReLU outputs, padding and masks.

**Multiplication** is an integer add of the codes and needs no correction at all.
**Addition** needs `log2(1 ± 2**d)`, plotted below.
        """
    )

    st.markdown("#### The log-add correction term")
    st.caption(
        "The smooth curve is the exact function; the step trace is the table a hardware "
        "implementation would hold, quantised to this format's log grid. Where they "
        "coincide, LUT mode and exact mode give identical results."
    )

    kind = st.radio(
        "Branch", ["add: log2(1 + 2**d)", "sub: log2(1 - 2**d)"], horizontal=True
    )
    kind_key = "add" if kind.startswith("add") else "sub"
    span = st.slider("Plot |d| up to", 1.0, float(fmt.lut_domain), min(6.0, fmt.lut_domain))

    table = get_lut(fmt, kind_key, lut_size)
    points = 400
    curve_rows = []
    for i in range(points + 1):
        d = -span * i / points
        if kind_key == "sub" and d == 0.0:
            continue
        exact_value = exact_mode(d, kind_key)
        lut_value = table.lookup(d)
        if not math.isfinite(exact_value) or exact_value < -40:
            continue
        curve_rows.append({"d": d, "correction": exact_value, "series": "exact"})
        curve_rows.append({"d": d, "correction": lut_value, "series": f"LUT ({len(table)} entries)"})

    curve_df = pd.DataFrame(curve_rows)
    curve_chart = (
        alt.Chart(curve_df)
        .mark_line()
        .encode(
            x=alt.X("d:Q", title="d = L_small - L_large  (log-domain gap)"),
            y=alt.Y("correction:Q", title="correction (log2 units)"),
            color=alt.Color("series:N", title=None),
            strokeDash=alt.StrokeDash("series:N", title=None),
        )
        .properties(height=340)
    )
    st.altair_chart(curve_chart, width="stretch")

    st.info(
        "**Exact mode is a software idealisation** — no accelerator computes a logarithm "
        "per addition. **LUT mode is the realistic one**: a ROM sampled and quantised at "
        "the format's resolution. At full resolution the two agree exactly, because ΔL is "
        "always a whole number of log-steps; decimate the table in the sidebar to see the "
        "error a smaller ROM would introduce.",
        icon="ℹ️",
    )

    st.caption(
        "Companion demo: `webapp/simulator.html` is a dependency-free JavaScript port of "
        "the same encode/decode/add/multiply/MAC pipeline that opens straight in a "
        "browser. This Streamlit app is the source of truth — it calls the real package."
    )
