# lns_arith — Logarithmic Number System arithmetic for DNN acceleration

A simulator for **LNS arithmetic**: a Python package that implements 16-bit and 8-bit
logarithmic number formats and their add / subtract / multiply / multiply-accumulate
operations, an error-characterisation experiment that measures them against FP32 and
FP16, and two interactive simulators — a full Streamlit app and a dependency-free HTML
page — that show the arithmetic step by step.

---

## What LNS is, and why an accelerator would want it

A floating-point number stores a sign, an exponent and a mantissa. A **logarithmic number
system** value stores a sign and a single fixed-point number: the base-2 logarithm of the
magnitude. That one change rewrites the cost of arithmetic.

Multiplying two numbers means **adding their logarithms**, so an LNS multiplier is an
integer adder. Division is a subtraction, squaring is a left shift, and square root is a
right shift. In a DNN accelerator, where the inner loop of every convolution and every
matmul is a multiply, replacing a multiplier array with an adder array is a large win in
area and energy — which is the entire motivation for the format.

The bill comes due at addition. There is no way to add two logarithms directly, so LNS
addition needs a correction term, `log2(1 ± 2^d)`, evaluated on the gap `d` between the
two operands' logs. Real hardware reads that term from a small ROM. So an LNS design
trades cheap multiplies against more expensive adds — a good trade for neural networks,
where multiplies dominate and the additions are accumulations that can share one
correction unit across many multipliers.

The format has a second property that suits DNNs: because it quantises the *logarithm*
uniformly, its **relative error is constant across its whole dynamic range**. A floating
point format's relative error sawtooths within each exponent band; an LNS format treats a
weight of 0.001 and an activation of 1000 with exactly the same fractional accuracy. That
matches how neural networks actually care about their numbers — they care about ratios,
not absolute magnitudes.

---

## The two formats

### LNS16 — 16 bits

```
 bit 15   bits 14..0
┌──────┬─────────────────────┐
│ sign │  code (15 bits)     │      L = (code − 16384) / 128
└──────┴─────────────────────┘      x = (−1)^sign × 2^L
```

| | |
|---|---|
| Layout | 1 sign + 8 integer + 7 fractional bits |
| Bias | **16384** (= 2^14) |
| Log-step | **1/128 ≈ 0.0078125** |
| Log2 range | `L ∈ [−127.9921875, +127.9921875]` |
| Magnitude range | **2.955e−39 … 3.384e+38** (≈ 256 binades) |
| Worst-case relative error | `2^(step/2) − 1` = **0.2711 %** |
| Codes | `0` = exact zero (reserved), `1 … 32767` = magnitudes |

### LNS8 — 8 bits

```
 bit 7    bits 6..0
┌──────┬─────────────┐
│ sign │ code (7 b)  │            L = (code − 64) / 8
└──────┴─────────────┘            x = (−1)^sign × 2^L
```

| | |
|---|---|
| Layout | 1 sign + 4 integer + 3 fractional bits |
| Bias | **64** (= 2^6) |
| Log-step | **1/8 = 0.125** |
| Log2 range | `L ∈ [−7.875, +7.875]` |
| Magnitude range | **4.260e−03 … 2.348e+02** (≈ 16 binades) |
| Worst-case relative error | `2^(step/2) − 1` = **4.4274 %** |
| Codes | `0` = exact zero (reserved), `1 … 127` = magnitudes |

### Zero, and what reserving it costs

A logarithm cannot represent zero, so one code point has to be sacrificed. This library
reserves **code 0** — the most-negative log slot — as an exact-zero sentinel.

The cost is exactly one log-step of range at the bottom: the smallest nonzero LNS16
magnitude is `2^−127.9921875` instead of `2^−128`, and the smallest LNS8 magnitude is
`2^−7.875 ≈ 4.26e−3` instead of `2^−8 ≈ 3.91e−3`. That is a negligible loss — a
half-percent of a binade — and it buys an exact zero, which matters enormously for DNN
tensors: ReLU outputs, padded borders, masked attention entries and pruned weights are all
exactly zero, and a format that turned them into `2^−128` instead would leak error into
every accumulation. The canonical zero word is `sign=0, code=0`; a word with `code=0` and
the sign bit set decodes to zero as well and is normalised to the canonical form.

### Overflow, underflow, and rounding

Encoding **rounds to nearest** on the fixed-point code (`floor(x + 0.5)`, ties away from
−∞ — the cheap incrementer a hardware encoder would use).

A magnitude outside the representable range is **clamped to the nearest representable
code and flagged** — it never wraps. Both a flag-based and a warning-based interface are
available:

```python
word = fp32_to_lns(1e30, LNS8)                       # warns LNSOverflowWarning, clamps
result = encode_with_flags(1e30, LNS8)               # result.overflow is True, no warning
```

Underflow clamps to the smallest **nonzero** code — it does not flush to zero, because
zero is reserved for values that really are zero. The only ways to reach the zero
sentinel are an exact zero input and exact cancellation in a subtraction.

---

## Repository layout

```
.
├── lns_arith/            the library
│   ├── __init__.py
│   ├── formats.py
│   ├── convert.py
│   ├── ops.py
│   ├── logadd.py
│   ├── errors.py
│   └── utils.py
├── tests/
│   ├── test_convert.py
│   ├── test_ops.py
│   ├── test_error_report.py
│   └── test_js_parity.py
├── benchmarks/
│   └── run_experiments.py
├── webapp/
│   ├── app.py            full Streamlit simulator
│   └── simulator.html    standalone HTML/JS demo
├── pyproject.toml
├── Makefile
├── README.md
└── LICENSE
```

### File by file

**`lns_arith/__init__.py`** — the public API. Re-exports everything a user needs from the
five implementation modules under one namespace, so `from lns_arith import LNS16,
fp32_to_lns, lns_mac` works without knowing the internal layout. Its docstring carries the
quickstart snippet and states the design rule that governs the whole package: functions in
`ops.py` take and return LNS codes, and `lns_to_float` exists only for tests, reports and
displays.

**`lns_arith/formats.py`** — the format definitions and the bit-level plumbing.
`LNSConfig` is a frozen dataclass that derives everything (bias, scale, code width, min and
max codes, the log and magnitude ranges, the worst-case relative error) from just the
integer and fractional bit counts, and `LNS16`/`LNS8` are the two instances the assignment
specifies — also exported under the names `LNS16Config` and `LNS8Config`. Around them sit
the word helpers every other module builds on: `pack`, `sign_of`, `code_of`, `is_zero`,
`negate`, `log_of_code`, `code_of_log`, `clamp_code`, and the `round_to_nearest` rule. An
LNS "word" is just a Python `int` holding the packed sign and code bits, which is what
makes the no-float-mid-operation constraint enforceable by construction.

**`lns_arith/convert.py`** — the float boundary, and the only module where floats and LNS
codes meet. `fp32_to_lns` and `fp16_to_lns` round the incoming value to binary32 or
binary16 first (using `struct`'s `'f'` and `'e'` codes, so the package needs no
third-party dependency), take the base-2 logarithm, quantise, clamp and pack. The named
wrappers `fp32_to_lns16`, `fp32_to_lns8`, `fp16_to_lns16`, `fp16_to_lns8`,
`lns16_to_float` and `lns8_to_float` are here too. `encode_with_flags` is the flag-based
variant used by the benchmark and the simulators; it also returns the pre-quantisation
`log2` value, which is what the step-by-step trace displays.

**`lns_arith/ops.py`** — the arithmetic core, operating on packed codes end to end.
`lns_mul` is an exact integer add (`c_a + c_b − bias`) with an XOR of the sign bits and
zero special-cased. `lns_add` implements the full sign logic: a zero operand returns the
other operand unchanged; same signs use `log2(1 + 2^d)`; opposite signs with equal codes
return the exact zero sentinel; opposite signs otherwise use `log2(1 − 2^d)` and may
underflow-clamp in the near-cancellation regime. `lns_sub` is `lns_add(a, negate(b))` and
`lns_mac` is `lns_add(lns_mul(a, b), acc)`; `lns_dot` chains MACs for the DNN experiments
this package is meant to feed. Every function takes an optional `trace` dict that it fills
with a step-by-step record — the branch taken, the codes, `ΔL`, the correction value, the
rounded delta and the clamping — which is what both web front-ends render.

**`lns_arith/logadd.py`** — the correction term, in both modes. `exact_mode(d, kind)`
evaluates `log2(1 ± 2^d)` in double precision, using `log1p` and `expm1` so it stays
accurate at both ends of the domain; this is the *software idealisation*. `CorrectionLUT`
precomputes the same function sampled over `d ∈ [−16, 0]` for LNS16 and `[−8, 0]` for
LNS8, with its **values quantised to the format's own fixed-point log grid**, exactly as a
hardware ROM would hold them; `lut_mode` reads the nearest entry. `lut_size` overrides the
table resolution, `set_mode` / `get_mode` / `use_mode` provide the global mode switch, and
tables are cached so a sweep does not rebuild them. Beyond the table's domain the
correction is smaller than half a log-step and is taken as zero.

**`lns_arith/errors.py`** — warnings and metrics. `LNSOverflowWarning` and
`LNSUnderflowWarning` (also exported as `OverflowWarning` / `UnderflowWarning`) derive from
a common `LNSWarning`, so a caller can promote every range event to an exception with one
`simplefilter` call — which the test suite does, to prove the library never clamps
silently. `relative_error`, `abs_error` and `error_metric` implement the experiment's
scoring rule: relative error where it is defined, absolute error when the reference is
exactly zero.

**`lns_arith/utils.py`** — reproducible test-vector generators. Each category
(`positive`, `negative`, `zero`, `small`, `large`, `random_log`, `random_linear`,
`random`, `extreme`) takes an explicit seed and uses its own `random.Random`, so results
never depend on global RNG state. `make_pairs` assembles `(a, b, acc)` triples with
category-appropriate pairing: extremes are paired with typical values so the FP32
*reference* stays finite and the comparison measures something, and the zero category
alternates so that `0 + 0`, `0 + x` and `x + 0` are all covered.

**`tests/test_convert.py`** — encoding and decoding. Checks the bit layouts against the
specification, exact round-trips of powers of two, the zero sentinel and negative zero,
sign independence, round-to-nearest behaviour, the half-step error bound, the flatness of
relative error across 200 binades, and the boundary cases: extreme representable values
that must *not* flag, out-of-range values that must clamp and warn, saturation rather than
wrapping, infinities, and a seeded 2000-sample randomised batch.

**`tests/test_ops.py`** — the arithmetic. Exact results for powers of two, the identity
and zero laws, commutativity, exact cancellation, sign selection by the larger operand,
overflow and underflow on every operation, subtraction as add-of-negation, MAC as
add-of-multiply, both log-add modes, the trace contents, and seeded randomised batches
whose bounds come from the analytic error model rather than from observed output. It also
contains the structural test that enforces the assignment's central constraint: it parses
`ops.py` with `ast` and fails if the module references `lns_to_float` or imports anything
from `convert.py`.

**`tests/test_error_report.py`** — the section-4 experiment run as a test. Executes the
full sweep at a small sample size and asserts the results against theory: conversion error
within the half-step bound, error flat across categories, multiplication bounded by two
input roundings, addition showing a cancellation tail, the full-resolution LUT matching
exact mode, the decimated LUT being measurably worse, LNS16 never clamping on FP32-range
inputs, LNS8 always clamping on out-of-range ones, FP16 saturating where LNS16 does not,
and the CSV/JSON/README plumbing round-tripping.

**`tests/test_js_parity.py`** — proves the HTML demo is not a separate, drifting
implementation. Extracts the `<script id="lns-core">` block from `simulator.html`, runs it
under Node, and compares packed result words, range flags, encodings, FP16 rounding (both
the native `Float16Array` path and the manual fallback) and every LUT entry against the
Python package across ~3900 cases. Skipped automatically when Node is not installed.

**`benchmarks/run_experiments.py`** — the error-characterisation experiment and its
report. Sweeps four operations × {LNS16, LNS8, FP16} × {exact, full LUT, decimated LUT} ×
eight input categories, scoring every sample against an FP32 reference, and produces three
reports: the error table (mean, median and max, with overflow/underflow counts), a
clamping report that verifies out-of-range inputs saturate and flag rather than wrap, and
a cancellation study that measures `a − b` as `b` approaches `a`. It prints to the
console, writes CSV and JSON, auto-generates the discussion text from the numbers it just
measured, and with `--update-readme` splices the results into this file. Every function is
importable, and the Streamlit app calls `run_all` directly.

**`webapp/app.py`** — the full Streamlit simulator, and the source of truth. Imports the
real package and calls the real arithmetic. The *Walkthrough* tab encodes the operands and
shows the complete pipeline — sign, `log2|x|`, code, packed word in binary and hex, the
branch taken, `ΔL`, the correction term, the rounded delta, the clamped result code and
the decoded float — beside FP32 and FP16 references with errors, plus a panel comparing
every format/mode combination on the same inputs. The *Batch experiment* tab runs
`run_experiments.py` in-process at an adjustable sample size and renders the error tables,
log-scale comparison charts, the clamping and cancellation studies, the auto-generated
discussion, and CSV/JSON downloads. The *Format reference* tab shows the bit layouts and
plots the correction function against its quantised table.

**`webapp/simulator.html`** — the standalone demo. A single file with no dependencies, no
build step and no server: a JavaScript port of `formats.py`, `convert.py`, `logadd.py` and
`ops.py` plus the same step-by-step trace UI, so it can be opened straight from a
`file://` URL or emailed as an attachment. It supports the same single-pair walkthrough
(encode/decode, add, subtract, multiply, MAC, exact vs LUT, both formats, FP32/FP16
sources, adjustable LUT decimation) and the same all-configurations comparison table, and
it plots the correction curve against its table. It deliberately omits the batch
experiment — it says so on the page and points at the Streamlit app — and
`tests/test_js_parity.py` keeps it bit-for-bit identical to the Python.

**`pyproject.toml`** — packaging. Editable-installable with `pip install -e .`; the core
has **no runtime dependencies**, with `[web]` (streamlit, pandas, altair) and `[dev]`
(pytest) extras. It also configures pytest to turn every uncaught `LNSWarning` into a test
failure, so the suite proves the library never clamps silently.

**`Makefile`** — the shortcuts: `make test`, `make report` (regenerates the results below),
`make report-quick`, `make web`, `make demo`, `make dev`, `make clean`.

---

## Install

```bash
pip install -e .                # core library only, zero dependencies
pip install -e ".[dev,web]"     # + pytest, streamlit, pandas, altair
```

Python 3.10 or newer.

## Quickstart

```python
from lns_arith import (
    LNS16, LNS8,
    fp32_to_lns, lns_to_float,
    lns_add, lns_sub, lns_mul, lns_mac,
)

# --- convert -----------------------------------------------------------
a = fp32_to_lns(3.5, LNS16)          # a packed 16-bit LNS word (an int)
b = fp32_to_lns(-1.25, LNS16)
acc = fp32_to_lns(0.5, LNS16)

lns_to_float(a, LNS16)               # 3.493534772398338    (0.185% low)

# --- arithmetic, entirely on codes -------------------------------------
lns_to_float(lns_mul(a, b, LNS16), LNS16)             # -4.362030930661031   (exact: -4.375)
lns_to_float(lns_add(a, b, LNS16, "exact"), LNS16)    #  2.2408755048192135  (exact:  2.25)
lns_to_float(lns_sub(a, b, LNS16, "exact"), LNS16)    #  4.731138843937364   (exact:  4.75)
lns_to_float(lns_mac(a, b, acc, LNS16, "lut"), LNS16) # -3.8721235869845887  (exact: -3.875)

# --- inspect the pipeline ---------------------------------------------
trace = {}
lns_add(a, b, LNS16, "exact", trace=trace)
print("\n".join(trace["steps"]))
# signs differ -> use log2(1 - 2**d)
# larger code c_hi = 16615 (L = +1.8046875), smaller c_lo = 16425 (L = +0.3203125)
# d = (c_lo - c_hi)/scale = -190/128 = -1.4843750
# correction = log2(1 - 2**d) = -0.6380146  [exact mode]
# delta_code = round(correction * scale) = round(-81.6659) = -82
# code = c_hi + delta_code = 16615 -82 = 16533

# --- range events are flagged, never silent ----------------------------
from lns_arith import encode_with_flags
r = encode_with_flags(1e30, LNS8)
r.overflow, lns_to_float(r.word, LNS8)                # (True, 234.75303506039583)

# --- a dot product accumulated in the LNS domain -----------------------
from lns_arith import lns_dot
w = [fp32_to_lns(v, LNS16) for v in (0.5, -0.25, 2.0)]
x = [fp32_to_lns(v, LNS16) for v in (1.5,  4.0,  0.125)]
lns_to_float(lns_dot(w, x, LNS16, "exact"), LNS16)    # 0.0 — 0.75 - 1.0 + 0.25 cancels
                                                      # exactly, and LNS returns exact zero
```

## Run the simulators

**Full Streamlit app** (imports the real package; includes the batch experiment):

```bash
pip install -e ".[web]"
streamlit run webapp/app.py          # or: make web
```

It opens at <http://localhost:8501>.

**Standalone HTML demo** (no Python, no server, no network):

```bash
xdg-open webapp/simulator.html       # Linux
open webapp/simulator.html           # macOS
start webapp\simulator.html          # Windows
```

or just drag the file into a browser tab. `make demo` prints the `file://` URL.

## Run the experiments and the tests

```bash
make test                            # pytest tests/ -q
make report                          # full sweep -> console, CSV, JSON, README
make report-quick                    # 200 samples, print only

python benchmarks/run_experiments.py --samples 5000 --seed 7 --outdir results/
```

`make report` regenerates the results section below, so the numbers in this README always
come from an actual run rather than being typed in by hand.

---

## Results

<!-- BEGIN AUTO-REPORT -->

_Generated by `python benchmarks/run_experiments.py --update-readme` with `--samples 2000 --seed 12345`; 160 (format, operation, mode, category) cells. Errors are relative to an FP32 reference, falling back to absolute error where the reference is exactly zero. `median` is the median of the per-category median errors -- the typical-precision number. `mean` is the mean of the per-category means and is deliberately sensitive to clamping. `max` is the single worst sample._

#### Precision: error by operation (in-range categories: `positive`, `negative`, `zero`, `random_log`, `random_linear`, `random`)

| Format | Operation | Log-add mode | Median error | Mean error | Max error | Clamped |
|---|---|---|---|---|---|---|
| LNS16 | convert | n/a | 1.336e-03 | 1.206e-03 | 2.709e-03 | 0.0% |
| LNS8 | convert | n/a | 2.715e-02 | 3.721e+01 | 4.225e+03 | 34.7% |
| FP16 | convert | n/a | 1.752e-04 | 2.892e-04 | inf | 4.7% |
| LNS16 | add | exact | 1.483e-03 | 2.915e-03 | 1.577e+00 | 0.0% |
| LNS16 | add | lut[2049] | 1.483e-03 | 2.915e-03 | 1.577e+00 | 0.0% |
| LNS16 | add | lut[257] | 1.792e-03 | 1.491e-02 | 1.917e+01 | 0.0% |
| LNS8 | add | exact | 3.268e-02 | 2.063e+00 | 3.509e+03 | 35.0% |
| LNS8 | add | lut[17] | 3.969e-02 | 2.101e+00 | 3.509e+03 | 35.0% |
| LNS8 | add | lut[65] | 3.268e-02 | 2.063e+00 | 3.509e+03 | 35.0% |
| FP16 | add | n/a | 2.218e-04 | 3.847e-04 | inf | 4.7% |
| LNS16 | mul | n/a | 1.597e-03 | 1.514e-03 | 5.397e-03 | 0.0% |
| LNS8 | mul | n/a | 1.777e-01 | 1.177e+06 | 2.899e+09 | 41.2% |
| FP16 | mul | n/a | 2.699e-04 | 2.243e-02 | inf | 7.3% |
| LNS16 | mac | exact | 1.724e-03 | 3.581e-03 | 6.653e+00 | 0.0% |
| LNS16 | mac | lut[2049] | 1.724e-03 | 3.581e-03 | 6.653e+00 | 0.0% |
| LNS16 | mac | lut[257] | 1.992e-03 | 1.513e-02 | 5.891e+01 | 0.0% |
| LNS8 | mac | exact | 3.873e-02 | 2.732e-01 | 3.726e+02 | 41.3% |
| LNS8 | mac | lut[17] | 4.348e-02 | 4.038e-01 | 1.364e+03 | 41.3% |
| LNS8 | mac | lut[65] | 3.873e-02 | 2.732e-01 | 3.726e+02 | 41.3% |
| FP16 | mac | n/a | 2.646e-04 | 5.980e-04 | inf | 7.3% |

#### Range: error by operation on out-of-range inputs (categories: `small`, `large`)

Large numbers in this table are *clamping*, not rounding: LNS8 simply cannot represent 1e-38 or 1e38, so it saturates and the relative error against FP32 is enormous. LNS16 and FP32 have near-identical range, so LNS16 barely clamps here; FP16 saturates above 65504 and its relative error becomes infinite.

| Format | Operation | Log-add mode | Median error | Mean error | Max error | Clamped |
|---|---|---|---|---|---|---|
| LNS16 | convert | n/a | 1.385e-03 | 1.374e-03 | 2.710e-03 | 0.0% |
| LNS8 | convert | n/a | 5.184e+17 | 2.365e+33 | 4.204e+35 | 100.0% |
| FP16 | convert | n/a | inf | 4.420e-01 | inf | 47.3% |
| LNS16 | add | exact | 1.360e-03 | 1.368e-03 | 3.910e-03 | 0.0% |
| LNS16 | add | lut[2049] | 1.360e-03 | 1.368e-03 | 3.910e-03 | 0.0% |
| LNS16 | add | lut[257] | 1.360e-03 | 1.368e-03 | 3.910e-03 | 0.0% |
| LNS8 | add | exact | 5.111e-01 | 5.128e-01 | 6.485e+00 | 100.0% |
| LNS8 | add | lut[17] | 5.111e-01 | 5.127e-01 | 6.485e+00 | 100.0% |
| LNS8 | add | lut[65] | 5.111e-01 | 5.128e-01 | 6.485e+00 | 100.0% |
| FP16 | add | n/a | inf | 1.843e-04 | inf | 47.3% |
| LNS16 | mul | n/a | 1.620e-03 | 1.829e-03 | 5.281e-03 | 0.0% |
| LNS8 | mul | n/a | 6.735e+17 | 3.906e+33 | 1.297e+36 | 100.0% |
| FP16 | mul | n/a | inf | 4.442e-01 | inf | 47.9% |
| LNS16 | mac | exact | 1.494e-03 | 1.591e-03 | 5.232e-03 | 0.0% |
| LNS16 | mac | lut[2049] | 1.494e-03 | 1.591e-03 | 5.232e-03 | 0.0% |
| LNS16 | mac | lut[257] | 1.494e-03 | 1.591e-03 | 5.232e-03 | 0.0% |
| LNS8 | mac | exact | 5.116e-01 | 5.294e-01 | 6.204e+01 | 100.0% |
| LNS8 | mac | lut[17] | 5.116e-01 | 5.280e-01 | 5.698e+01 | 100.0% |
| LNS8 | mac | lut[65] | 5.116e-01 | 5.294e-01 | 6.204e+01 | 100.0% |
| FP16 | mac | n/a | inf | 2.254e-04 | inf | 47.9% |

#### Mean error by input category (round-trip conversion)

Mean error, so out-of-range categories show their clamping. Read the LNS16 row across: the error barely moves from `positive` to `large`, which is the flat relative error of a log format. The LNS8 row explodes on `small`/`large` and the FP16 row on `small`, because those values are outside those formats' range.

| Format | positive | negative | zero | small | large | random_log | random_linear | random |
|---|---|---|---|---|---|---|---|---|
| LNS16 | 1.335e-03 | 1.368e-03 | 4.489e-04 | 1.376e-03 | 1.373e-03 | 1.347e-03 | 1.360e-03 | 1.378e-03 |
| LNS8 | 2.082e-01 | 1.961e-01 | 7.259e-03 | 4.730e+33 | 9.973e-01 | 1.351e+02 | 2.176e-02 | 8.773e+01 |
| FP16 | 1.807e-04 | 1.792e-04 | 5.634e-05 | 8.837e-01 | 1.613e-04 | 6.907e-04 | 1.721e-04 | 4.563e-04 |

#### Mean error by input category (addition, exact log-add)

| Format | positive | negative | zero | small | large | random_log | random_linear | random |
|---|---|---|---|---|---|---|---|---|
| LNS16 | 1.636e-03 | 1.646e-03 | 9.127e-04 | 1.358e-03 | 1.378e-03 | 1.821e-03 | 7.414e-03 | 4.060e-03 |
| LNS8 | 1.420e-01 | 1.321e-01 | 1.523e-02 | 2.828e-02 | 9.973e-01 | 8.517e+00 | 8.391e-02 | 3.486e+00 |
| FP16 | 2.198e-04 | 2.317e-04 | 1.149e-04 | 1.703e-04 | 1.983e-04 | 3.339e-04 | 9.401e-04 | 4.678e-04 |

#### Cancellation study (`a - b` as `b -> a`)

| Format | Relative gap | FP32 reference | LNS result | Relative error |
|---|---|---|---|---|
| LNS16 | 0.5 | 0.5 | 0.5 | 0 |
| LNS16 | 0.1 | 0.1 | 0.0979669 | 2.033e-02 |
| LNS16 | 0.01 | 0.00999999 | 0.0107534 | 7.534e-02 |
| LNS16 | 0.001 | 0.000999987 | 0 | 1.000e+00 |
| LNS16 | 0.0001 | 0.000100017 | 0 | 1.000e+00 |
| LNS16 | 1e-05 | 1.00136e-05 | 0 | 1.000e+00 |
| LNS16 | 1e-06 | 1.01328e-06 | 0 | 1.000e+00 |
| LNS8 | 0.5 | 0.5 | 0.5 | 0 |
| LNS8 | 0.1 | 0.1 | 0.0810525 | 1.895e-01 |
| LNS8 | 0.01 | 0.00999999 | 0 | 1.000e+00 |
| LNS8 | 0.001 | 0.000999987 | 0 | 1.000e+00 |
| LNS8 | 0.0001 | 0.000100017 | 0 | 1.000e+00 |
| LNS8 | 1e-05 | 1.00136e-05 | 0 | 1.000e+00 |
| LNS8 | 1e-06 | 1.01328e-06 | 0 | 1.000e+00 |

#### Overflow / underflow clamping

| Format | Probe set | Values | Overflow | Underflow | In range | Clamped to |x| |
|---|---|---|---|---|---|---|
| LNS16 | beyond-max | 5 | 5 | 0 | 0 | 3.384e+38 .. 3.384e+38 |
| LNS16 | beyond-min | 5 | 0 | 5 | 0 | 2.955e-39 .. 2.955e-39 |
| LNS16 | large(1e6) | 3 | 0 | 0 | 3 | - |
| LNS16 | small(1e-6) | 3 | 0 | 0 | 3 | - |
| LNS8 | beyond-max | 5 | 5 | 0 | 0 | 234.8 .. 234.8 |
| LNS8 | beyond-min | 5 | 0 | 5 | 0 | 0.00426 .. 0.00426 |
| LNS8 | large(1e6) | 3 | 3 | 0 | 0 | 234.8 .. 234.8 |
| LNS8 | small(1e-6) | 3 | 0 | 3 | 0 | 0.00426 .. 0.00426 |

#### Discussion

```text
Range
-----
  LNS16 covers magnitudes 2.955e-39 .. 3.384e+38 (log2 in [-127.992, 127.992]) -- 256 binades of dynamic range, against the 254 binades of FP32's normal numbers (1.18e-38 .. 3.40e38). The two ranges are effectively the same size, with LNS16's window sitting about two binades lower: it reaches further down and stops a hundredth of a binade short at the top. LNS16 gets FP32's reach out of 16 bits by spending them all on one monotone log axis instead of splitting them into an exponent field and a mantissa field -- and pays for it with a far coarser step.
  LNS8 covers only 4.260e-03 .. 2.348e+02, about 16 binades. That is a far narrower window than FP16 (6.0e-8 .. 6.5e4, ~40 binades), so LNS8 is a per-tensor-scaled format in practice: unscaled activations run off both ends of it.
  Measured: on the 'large' category (1e3..1e38) LNS8 overflowed and clamped, giving a mean convert error of 9.973e-01 versus 1.373e-03 for LNS16, which never leaves its range there. On 'small' (1e-38..1e-3) the same story runs in reverse: LNS8 4.730e+33 versus LNS16 1.376e-03.

Precision
---------
  LNS quantises log2(|x|) uniformly, so its relative error is *constant across the entire dynamic range*: a half-step of log error is a relative error of 2**(step/2) - 1, i.e. 0.2711% for LNS16 (step 1/128) and 4.427% for LNS8 (step 1/8). There are no exponent bands and no wobble.
  Measured round-trip conversion error confirms the bound. On 'random_linear' (values in [-4, 4], in range for both formats) the median relative error is 1.341e-03 for LNS16 against a 2.711e-03 worst case, and 2.124e-02 for LNS8 against 4.427e-02. The same medians on 'random' (magnitudes from 1e-6 to 1e6) are 1.372e-03 and 3.271e-02 -- twelve orders of magnitude of input range, and the relative error does not move. That flatness is the defining property of the format.
  FP32 and FP16 are the mirror image: within one binade the spacing is uniform in the *linear* domain, so relative error sawtooths between 2**-24 and 2**-23 (FP32) or 2**-11 and 2**-10 (FP16) as the mantissa sweeps a binade -- but the exponent field buys a very wide range for very few bits. In half-step terms FP32 is ~45488x finer than LNS16 and FP16 is ~5.6x finer, while LNS16 beats both on range (measured FP16 convert error on 'random': 1.905e-04 median, inf max -- the max is infinite because FP16 saturates above 65504).
  Multiplication needs no correction term at all: the product code is the exact integer sum c_a + c_b - bias, with no rounding of its own. What it does do is *add the two input errors*, so its median error (1.687e-03 for LNS16) sits just above a single conversion (1.341e-03). There is no error growth beyond that, and no possibility of catastrophic loss.
  Addition costs an extra rounding of the log2(1 +/- 2**d) correction, but in the typical case the two input errors partially average out rather than accumulating, so its median error (1.949e-03) is comparable to multiply's (1.687e-03). Addition's real cost is in the tail, not the middle: its max error over the same category is 1.577e+00 against 5.339e-03 for multiply. That tail is cancellation, and it is discussed below.

Exact vs LUT log-add
--------------------
  At full table resolution the LUT reproduces the exact mode bit for bit (LNS16 add on 'random_linear': exact 1.949e-03 vs lut 1.949e-03). That is expected, not a bug: d is always an exact multiple of the log-step, so the lookup lands on a sampled point, and the stored value has already been rounded to the same grid the result gets rounded to. A full-resolution table is exact *with respect to the format*; the idealisation in 'exact' mode only shows up once the table is smaller than the format's own resolution.
  Decimating the table is what exposes lookup error. With a 257-entry LNS16 table (1/8 resolution) the add median error rises from 1.949e-03 to 4.495e-03, and a 17-entry LNS8 table moves LNS8 from 3.430e-02 to 4.630e-02. This is the real hardware trade-off: ROM area against accuracy.

Cancellation -- the exception to constant relative error
-------------------------------------------------------
  LNS16: a gap of 0.01 already costs 7.534e-02 relative error, and by a gap of 0.001 the result collapses to exact zero (relative error 1.0).
  LNS8: a gap of 0.1 already costs 1.895e-01 relative error, and by a gap of 0.01 the result collapses to exact zero (relative error 1.0).
  Subtracting nearly-equal operands is the one place the flat-error property breaks. The two operands share almost their entire log code, so the code *difference* that drives the correction term carries only a handful of bits, and the tiny result inherits the absolute quantisation of the large inputs. Once the operands are closer together than one log-step their codes are identical and the difference is reported as exactly zero. Floating point suffers the same catastrophic cancellation; LNS reaches it sooner because the operands are coarser to begin with, and LNS8 -- whose log-step is 16x wider than LNS16's -- sooner still.
  Exact cancellation (identical codes, opposite signs) is handled as a code comparison and returns the exact zero sentinel -- the -inf singularity of log2(1 - 2**0) is never evaluated.

Clamping
--------
  1026 out-of-range probe values were clamped to the nearest representable code and flagged; none wrapped. Overflow saturates at 3.384e+38 (LNS16) / 2.348e+02 (LNS8) and underflow at 2.955e-39 / 4.260e-03 -- never to zero, since the zero code is a reserved sentinel reached only by an exact zero input or by exact cancellation.

Bottom line for DNN acceleration
--------------------------------
  LNS16 is a credible alternative to FP16 for inference: FP32-class range where FP16 has ~40 binades, coarser precision than FP16 but uniform -- the error does not depend on where in the range a value sits -- and multiplies that cost an integer add. LNS8 needs per-tensor scaling to keep values inside its ~4.3e-3 .. 2.3e2 window, and its ~4.4% quantisation step makes it a quantised-inference format rather than a training format. In both cases the cost centre is addition -- the correction term -- which is exactly where the accumulator width and the LUT size of a real design get spent.
```

<!-- END AUTO-REPORT -->

---

## Two notes on what is and is not idealised

**Exact-mode log-add is a software idealisation.** No accelerator evaluates a logarithm
per addition. `mode="exact"` calls `math.log2` in double precision and is there as a
reference point — the best any implementation of this format could do. `mode="lut"` is the
realistic one: a table sampled over the correction function's useful domain with its
entries rounded to the format's own fixed-point log grid, read by nearest-neighbour
lookup, which is what a ROM in a datapath actually does.

There is a subtlety worth stating plainly, because the measurements show it. At *full*
table resolution — one entry per log-step, as specified — LUT mode reproduces exact mode
**bit for bit**. That is not a bug and not a coincidence: `ΔL` in LNS addition is the
difference of two integer codes divided by the scale, so it is always a whole number of
log-steps and the lookup always lands exactly on a sampled point; and the stored value has
already been rounded to the same grid the result will be rounded to. A full-resolution
table is *exact with respect to the format*. Lookup-table error only appears once the
table is smaller than the format's own resolution, which is the interesting engineering
regime — ROM area against accuracy — so the benchmark sweeps decimated tables as well
(1/8 resolution for LNS16, 1/4 for LNS8) and the simulators let you decimate interactively.

**The "no float conversion mid-operation" constraint is enforced by construction.** Every
function in `ops.py` accepts and returns packed LNS words — plain Python `int`s holding
the sign and code bits — and the module imports nothing from `convert.py`. Multiplication
is integer addition of codes; addition is an integer comparison, an integer subtraction to
get `ΔL`, one correction-term evaluation in the log domain, and an integer add. No operand
is ever raised to a power of two and no result is ever built from a linear-domain
quantity. `lns_to_float` is called exactly once per experiment sample, at the very end,
outside the arithmetic, purely to compare against the FP32 reference. This is not merely a
convention: `tests/test_ops.py::test_ops_module_never_calls_lns_to_float` parses `ops.py`
with `ast` and fails the build if the module so much as references the float-domain
helpers.

## License

MIT — see [LICENSE](LICENSE).
