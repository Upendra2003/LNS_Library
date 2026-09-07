"""Differential test: the JavaScript port must match the Python implementation.

``webapp/simulator.html`` reimplements the arithmetic core in JavaScript so the
demo can run from a ``file://`` URL with no Python.  A reimplementation is only
useful if it agrees with the original, so this test extracts the
``<script id="lns-core">`` block from the HTML, runs it under Node, and compares
packed result words, range flags and encodings against :mod:`lns_arith` across a
sweep of formats, source precisions, operations, log-add modes and operands.

Skipped automatically when Node is not installed, so it never blocks a run on a
machine that only has Python.

It can also be run directly::

    python tests/test_js_parity.py
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lns_arith import (  # noqa: E402
    LNS8,
    LNS16,
    LNSConfig,
    encode_with_flags,
    lns_add_with_flags,
    lns_mac_with_flags,
    lns_mul_with_flags,
    lns_sub_with_flags,
    to_fp16,
)
from lns_arith.logadd import default_lut_size  # noqa: E402
from lns_arith.utils import gen_random  # noqa: E402

SIMULATOR = _REPO_ROOT / "webapp" / "simulator.html"
FORMATS = {"LNS16": LNS16, "LNS8": LNS8}

NODE = shutil.which("node") or shutil.which("nodejs")
requires_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")

DRIVER = """
const LNS = require(process.argv[2]);
const fs = require('fs');
const payload = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));

function runOp(op, words, fmt, mode, lutSize) {
  switch (op) {
    case 'convert': return { word: words.a, overflow: false, underflow: false };
    case 'add': return LNS.add(words.a, words.b, fmt, mode, lutSize, null);
    case 'sub': return LNS.sub(words.a, words.b, fmt, mode, lutSize, null);
    case 'mul': return LNS.mul(words.a, words.b, fmt, null);
    case 'mac': return LNS.mac(words.a, words.b, words.acc, fmt, mode, lutSize, null);
    default: throw new Error('unknown op ' + op);
  }
}

const results = payload.cases.map(function (c) {
  const fmt = LNS.FORMATS[c.format];
  const ea = LNS.encode(c.a, fmt, c.source);
  const eb = LNS.encode(c.b, fmt, c.source);
  const eacc = LNS.encode(c.acc, fmt, c.source);
  const out = runOp(c.op, { a: ea.word, b: eb.word, acc: eacc.word }, fmt, c.mode, c.lutSize);
  return {
    encA: [ea.word, ea.overflow, ea.underflow],
    encB: [eb.word, eb.overflow, eb.underflow],
    encAcc: [eacc.word, eacc.overflow, eacc.underflow],
    word: out.word,
    overflow: !!out.overflow,
    underflow: !!out.underflow
  };
});

// JSON has no Infinity, so non-finite values travel as strings.
function jsonSafe(v) { return Number.isFinite(v) ? v : String(v); }

const fp16 = payload.fp16Probe.map(function (x) {
  return [jsonSafe(LNS.toFp16(x)), jsonSafe(LNS.fp16Manual(x))];
});

const luts = payload.lutProbe.map(function (p) {
  const fmt = LNS.FORMATS[p.format];
  return LNS.getLut(fmt, p.kind, p.size).entries.map(jsonSafe);
});

console.log(JSON.stringify({ results: results, fp16: fp16, luts: luts }));
"""


def extract_core(html_path: Path = SIMULATOR) -> str:
    """Pull the ``lns-core`` script block out of the standalone simulator."""
    html = html_path.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="lns-core">(.*?)</script>', html, re.DOTALL
    )
    assert match, "simulator.html has no <script id=\"lns-core\"> block"
    return match.group(1)


def build_cases() -> list[dict[str, Any]]:
    """A sweep wide enough to hit every branch in both implementations."""
    fixed: list[tuple[float, float, float]] = [
        (3.5, -1.25, 0.5),
        (8.0, 0.25, 2.0),
        (1.0, 1.0, 1.0),
        (0.0, 2.75, 1.0),
        (2.75, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        (2.5, -2.5, 0.0),          # exact cancellation
        (1.0, -0.999, 0.0),        # near cancellation
        (-6.25, -2.5, -1.5),
        (1e-6, 3e-7, 0.0),         # underflows LNS8
        (1e6, 2.5e5, 0.0),         # overflows LNS8
        (5000.0, -0.0001, 1.0),
        (1e30, 1e30, 1e30),        # overflows on multiply
        (1e-30, 1e-30, 1.0),
        (65600.0, 2.0, 1.0),       # beyond FP16 range
        (0.125, -0.125, 0.0),
        (-1.0, 1.0000001, 0.0),
    ]
    randoms = gen_random(160, seed=31337)
    for i in range(0, len(randoms) - 2, 3):
        fixed.append((randoms[i], randoms[i + 1], randoms[i + 2]))

    cases: list[dict[str, Any]] = []
    for fmt_name, fmt in FORMATS.items():
        full = default_lut_size(fmt)
        modes: list[tuple[str, int | None]] = [
            ("exact", None),
            ("lut", full),
            ("lut", (full - 1) // 8 + 1),
            ("lut", 9),
        ]
        for source in ("fp32", "fp16"):
            for op in ("convert", "add", "sub", "mul", "mac"):
                for mode, lut_size in modes:
                    if op in ("convert", "mul") and mode != "exact":
                        continue
                    for a, b, acc in fixed:
                        cases.append(
                            {
                                "format": fmt_name,
                                "source": source,
                                "op": op,
                                "mode": mode,
                                "lutSize": lut_size,
                                "a": a,
                                "b": b,
                                "acc": acc,
                            }
                        )
    return cases


def python_result(case: dict[str, Any]) -> dict[str, Any]:
    """Compute the same case with the Python package."""
    fmt: LNSConfig = FORMATS[case["format"]]
    source = case["source"]
    mode = case["mode"]
    lut_size = case["lutSize"]

    ea = encode_with_flags(case["a"], fmt, source=source)
    eb = encode_with_flags(case["b"], fmt, source=source)
    eacc = encode_with_flags(case["acc"], fmt, source=source)

    op = case["op"]
    if op == "convert":
        word, overflow, underflow = ea.word, False, False
    elif op == "add":
        word, overflow, underflow = lns_add_with_flags(
            ea.word, eb.word, fmt, mode, lut_size=lut_size
        )
    elif op == "sub":
        word, overflow, underflow = lns_sub_with_flags(
            ea.word, eb.word, fmt, mode, lut_size=lut_size
        )
    elif op == "mul":
        word, overflow, underflow = lns_mul_with_flags(ea.word, eb.word, fmt)
    elif op == "mac":
        word, overflow, underflow = lns_mac_with_flags(
            ea.word, eb.word, eacc.word, fmt, mode, lut_size=lut_size
        )
    else:  # pragma: no cover - build_cases never emits this
        raise ValueError(op)

    return {
        "encA": [ea.word, ea.overflow, ea.underflow],
        "encB": [eb.word, eb.overflow, eb.underflow],
        "encAcc": [eacc.word, eacc.overflow, eacc.underflow],
        "word": word,
        "overflow": overflow,
        "underflow": underflow,
    }


FP16_PROBE = [
    0.0, 1.0, -1.0, 0.5, 1.0 + 2.0**-11, 1.0 + 2.0**-12, 65504.0, 65519.0, 65520.0,
    70000.0, -70000.0, 6.103515625e-05, 6e-08, 2.0**-24, 2.0**-25, 1.5 * 2.0**-25,
    3.14159265358979, -2.718281828, 1e-7, 12345.6789, 0.1, 1023.5, 2048.5,
]

LUT_PROBE = [
    {"format": "LNS16", "kind": "add", "size": default_lut_size(LNS16)},
    {"format": "LNS16", "kind": "sub", "size": default_lut_size(LNS16)},
    {"format": "LNS8", "kind": "add", "size": default_lut_size(LNS8)},
    {"format": "LNS8", "kind": "sub", "size": default_lut_size(LNS8)},
    {"format": "LNS16", "kind": "add", "size": 129},
]


def run_node(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Execute the extracted JS core under Node and return its results."""
    assert NODE is not None
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        core_path = tmpdir / "lns_core.js"
        core_path.write_text(extract_core(), encoding="utf-8")
        driver_path = tmpdir / "driver.js"
        driver_path.write_text(DRIVER, encoding="utf-8")
        payload_path = tmpdir / "payload.json"
        payload_path.write_text(
            json.dumps({"cases": cases, "fp16Probe": FP16_PROBE, "lutProbe": LUT_PROBE}),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [NODE, str(driver_path), str(core_path), str(payload_path)],
            capture_output=True,
            text=True,
            timeout=180,
        )
    if completed.returncode != 0:
        raise AssertionError(f"node failed:\n{completed.stderr}")
    return json.loads(completed.stdout)


@pytest.fixture(scope="module")
def parity() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = build_cases()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cases, run_node(cases)


@requires_node
def test_js_arithmetic_matches_python_bit_for_bit(parity) -> None:
    cases, node_output = parity
    results = node_output["results"]
    assert len(results) == len(cases)

    mismatches: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for case, js in zip(cases, results):
            py = python_result(case)
            for key in ("encA", "encB", "encAcc", "word", "overflow", "underflow"):
                if py[key] != js[key]:
                    mismatches.append(
                        f"{case['format']}/{case['source']}/{case['op']}/{case['mode']}"
                        f"[{case['lutSize']}] a={case['a']!r} b={case['b']!r} "
                        f"acc={case['acc']!r}: {key} python={py[key]!r} js={js[key]!r}"
                    )
                    break
    assert not mismatches, (
        f"{len(mismatches)} of {len(cases)} cases differ:\n  " + "\n  ".join(mismatches[:12])
    )


def _from_json_number(value: Any) -> float:
    """Undo the driver's non-finite-as-string encoding."""
    return float(value)


@requires_node
def test_js_fp16_rounding_matches_python(parity) -> None:
    """Both the native Float16Array path and the manual fallback must agree."""
    _, node_output = parity
    for value, (native_raw, manual_raw) in zip(FP16_PROBE, node_output["fp16"]):
        expected = to_fp16(value)
        native = _from_json_number(native_raw)
        manual = _from_json_number(manual_raw)
        assert native == expected or (
            math.isinf(native) and math.isinf(expected)
            and math.copysign(1, native) == math.copysign(1, expected)
        ), f"native fp16({value}) = {native}, expected {expected}"
        assert manual == expected or (
            math.isinf(manual) and math.isinf(expected)
            and math.copysign(1, manual) == math.copysign(1, expected)
        ), f"manual fp16({value}) = {manual}, expected {expected}"


@requires_node
def test_js_lut_tables_match_python(parity) -> None:
    """The quantised correction tables must be identical entry for entry."""
    from lns_arith.logadd import get_lut

    _, node_output = parity
    for probe, js_entries in zip(LUT_PROBE, node_output["luts"]):
        table = get_lut(FORMATS[probe["format"]], probe["kind"], probe["size"])
        assert len(js_entries) == len(table.entries), probe
        for index, (py_value, js_raw) in enumerate(zip(table.entries, js_entries)):
            js_value = _from_json_number(js_raw)
            if math.isinf(py_value):
                assert math.isinf(js_value) and js_value < 0, (probe, index)
            else:
                assert js_value == pytest.approx(py_value, rel=0, abs=1e-12), (probe, index)


@requires_node
def test_simulator_html_is_self_contained() -> None:
    """The demo must open from file:// with no network and no build step."""
    html = SIMULATOR.read_text(encoding="utf-8")
    assert "<script src=" not in html, "simulator.html must not load external scripts"
    assert "<link " not in html, "simulator.html must not load external stylesheets"
    for forbidden in ("http://", "https://"):
        assert forbidden not in html, f"simulator.html references {forbidden}"
    assert "streamlit run webapp/app.py" in html, (
        "the demo must point at the full Streamlit app"
    )


if __name__ == "__main__":  # pragma: no cover - manual invocation
    if NODE is None:
        raise SystemExit("Node.js is not installed; cannot run the parity check.")
    all_cases = build_cases()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = run_node(all_cases)
        bad = 0
        for c, j in zip(all_cases, output["results"]):
            p = python_result(c)
            if any(p[k] != j[k] for k in ("encA", "encB", "encAcc", "word", "overflow", "underflow")):
                bad += 1
                if bad <= 10:
                    print("MISMATCH", c, "python:", p, "js:", j)
    print(f"{len(all_cases) - bad}/{len(all_cases)} cases match")
    raise SystemExit(1 if bad else 0)
