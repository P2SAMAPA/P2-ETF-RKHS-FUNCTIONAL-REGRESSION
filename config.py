import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-rkhs-functional-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SMH", "SOXX", "XLB",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SMH", "SOXX", "XLB",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

MACRO_COLS_CORE     = ["VIX", "DXY", "T10Y2Y"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]

# ── Rolling windows (trading days) ────────────────────────────────────────────
WINDOWS = [63, 126, 252, 504]

# ── Functional data representation ───────────────────────────────────────────
# Each training sample is a path (function) of length SEQ_LEN trading days
# ending at time t, with target = forward return over PRED_HORIZON days.
SEQ_LEN      = 21    # length of each input path (function domain [0, SEQ_LEN])
PRED_HORIZON = 21    # forward return prediction target

# ── Kernel choice ─────────────────────────────────────────────────────────────
# "sobolev_w12" : W^{1,2} Sobolev kernel  k(f,g) = <f,g>_L2 + <f',g'>_L2
#                 Captures both level and derivative (momentum) information
# "l2"          : Pure L2 inner product  k(f,g) = <f,g>_L2
# "rbf_path"    : Gaussian RBF on path distance  k(f,g) = exp(-||f-g||^2 / 2sigma^2)
KERNEL = "sobolev_w12"
RBF_SIGMA = 0.01     # only used if KERNEL = "rbf_path"

# ── Operator-valued output ────────────────────────────────────────────────────
# The response is the forward RETURN PATH (function of horizon h):
#   y(h) = log_return over [t, t+h] for h in [1, PRED_HORIZON]
# This gives a full term structure of expected returns, not just a scalar.
# The score uses the integral of the predicted return path: E[integral y(h) dh]
OUTPUT_MODE = "path"    # "scalar" for mean return only, "path" for full horizon

# ── KRR regularisation ────────────────────────────────────────────────────────
KRR_LAMBDA = 1e-3

# ── Score construction ────────────────────────────────────────────────────────
# From the predicted return path yhat(h), h in [1, PRED_HORIZON]:
#   mean_score   : integral yhat(h) dh / PRED_HORIZON  (mean predicted return)
#   slope_score  : yhat(PRED_HORIZON) - yhat(1)         (term structure slope)
#   norm_score   : ||yhat||_L2                           (signal strength)
WEIGHT_MEAN  = 0.60
WEIGHT_SLOPE = 0.25
WEIGHT_NORM  = 0.15

# Minimum training samples
MIN_SAMPLES = 20

TOP_N = 3
