"""``lns_arith`` -- a Logarithmic Number System arithmetic simulator for DNNs.

LNS stores a number as the fixed-point ``log2`` of its magnitude plus a sign
bit.  Multiplication becomes an integer add, division a subtract, and powers a
shift -- which is why LNS is attractive for the multiply-dominated inner loops
of neural-network inference.  The price is addition, which needs a correction
term (see :mod:`lns_arith.logadd`).

Quickstart
----------
::

    from lns_arith import LNS16, fp32_to_lns, lns_to_float, lns_add, lns_mul, lns_mac

    a = fp32_to_lns(3.5, LNS16)
    b = fp32_to_lns(-1.25, LNS16)

    lns_to_float(lns_mul(a, b, LNS16), LNS16)          # ~ -4.375
    lns_to_float(lns_add(a, b, LNS16, "exact"), LNS16) # ~  2.25
    acc = fp32_to_lns(0.5, LNS16)
    lns_to_float(lns_mac(a, b, acc, LNS16, "lut"), LNS16)  # ~ -3.875

Design rule: every function in :mod:`lns_arith.ops` takes and returns *LNS
words* (packed ``int`` codes).  Nothing in the arithmetic core converts to
float mid-operation; :func:`lns_to_float` is for tests, reports and displays.
"""

from __future__ import annotations

from .convert import (
    EncodeResult,
    encode_with_flags,
    fp16_to_lns,
    fp16_to_lns8,
    fp16_to_lns16,
    fp32_to_lns,
    fp32_to_lns8,
    fp32_to_lns16,
    lns8_to_float,
    lns16_to_float,
    lns_to_float,
    quantize_source,
    to_fp16,
    to_fp32,
)
from .errors import (
    LNSOverflowWarning,
    LNSUnderflowWarning,
    LNSWarning,
    abs_error,
    error_metric,
    relative_error,
)
from .formats import (
    FORMATS,
    LNS8,
    LNS16,
    LNS8Config,
    LNS16Config,
    LNSConfig,
    canonicalize,
    code_of,
    describe,
    get_format,
    is_zero,
    log_of_code,
    negate,
    pack,
    sign_of,
    zero,
)
from .logadd import (
    CorrectionLUT,
    correction,
    default_lut_size,
    exact_mode,
    get_lut,
    get_mode,
    lut_mode,
    set_mode,
    use_mode,
)
from .ops import (
    OpResult,
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
    lns_sub_with_flags,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # formats
    "LNSConfig",
    "LNS16",
    "LNS8",
    "LNS16Config",
    "LNS8Config",
    "FORMATS",
    "get_format",
    "pack",
    "sign_of",
    "code_of",
    "is_zero",
    "zero",
    "negate",
    "canonicalize",
    "log_of_code",
    "describe",
    # conversion
    "EncodeResult",
    "encode_with_flags",
    "quantize_source",
    "to_fp32",
    "to_fp16",
    "fp32_to_lns",
    "fp16_to_lns",
    "lns_to_float",
    "fp32_to_lns16",
    "fp32_to_lns8",
    "fp16_to_lns16",
    "fp16_to_lns8",
    "lns16_to_float",
    "lns8_to_float",
    # log-add correction
    "exact_mode",
    "lut_mode",
    "correction",
    "CorrectionLUT",
    "get_lut",
    "default_lut_size",
    "set_mode",
    "get_mode",
    "use_mode",
    # arithmetic
    "OpResult",
    "lns_add",
    "lns_sub",
    "lns_mul",
    "lns_mac",
    "lns_dot",
    "lns_neg",
    "lns_abs",
    "lns_add_with_flags",
    "lns_sub_with_flags",
    "lns_mul_with_flags",
    "lns_mac_with_flags",
    # errors / metrics
    "LNSWarning",
    "LNSOverflowWarning",
    "LNSUnderflowWarning",
    "abs_error",
    "relative_error",
    "error_metric",
]
