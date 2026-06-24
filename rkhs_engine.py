"""
rkhs_engine.py — RKHS Functional Regression Engine
====================================================

Theory
------
**Reproducing Kernel Hilbert Spaces (RKHS)**

An RKHS H is a Hilbert space of functions f: X → R where point evaluation
is a bounded linear functional. By the Riesz representation theorem, there
exists a unique kernel k: X × X → R such that:

    f(x) = <f, k(x,·)>_H    (reproducing property)

The kernel encodes the inner product structure of H:
    <k(x,·), k(x',·)>_H = k(x, x')

**Functional Data Analysis in RKHS**

When the input domain X is a function space (e.g. L²[0,T]), we work with
*functional* kernels — kernels defined between functions (paths) rather than
between vectors.

For return paths f, g: [0,T] → R, the **Sobolev W^{1,2} kernel** is:

    k(f, g) = <f, g>_{W^{1,2}} = ∫₀ᵀ f(t)g(t)dt + ∫₀ᵀ f'(t)g'(t)dt

This inner product penalises both the level (L² term) and the derivative
(H¹ term), making it sensitive to both the shape and the rate of change
of the return path.

**Operator-Valued Kernel Ridge Regression (Kadri et al. 2016)**

Standard KRR: scalar response y = f(x), fit f in RKHS.
    α = (K + λI)⁻¹ y,  ŷ(x) = Σᵢ αᵢ k(x, xᵢ)

Operator-valued KRR: the response is itself a FUNCTION y: [0,H] → R
(the term structure of forward returns over horizons h=1,...,H).

    Y ∈ R^{N × H}  (N training samples, H output horizons)
    α = (K + λI)⁻¹ Y   (solve N × N system for each of H outputs)
    Ŷ(x) = k_vec(x) @ α  (predicted return path, shape H)

Where k_vec(x) = [k(x, x₁), ..., k(x, xₙ)] is the kernel vector.

**Distinction from FPCR-MACRO (in suite):**
- FPCR-MACRO: functional PCA on macro signals → regression on PC scores
  (reduces dimensionality first, then fits a standard linear model)
- This engine: direct operator-valued regression in RKHS — no dimensionality
  reduction, no parametric form assumed, full infinite-dimensional path
  structure is used via the kernel

**Distinction from SKR (Signature Kernel, in suite):**
- SKR: signature kernel computed via Goursat PDE (captures non-linear path
  geometry via iterated integrals)
- RKHS-FR: Sobolev W^{1,2} kernel (captures smooth function structure via
  L² inner product + derivative inner product) — simpler but analytically
  tractable with explicit representer theorem

**Score Construction**

After fitting operator-valued KRR, the predicted return path for today's
input function f_today is Ŷ ∈ R^H, where Ŷ[h] is the predicted log return
over horizon h.

Three score components:
1. mean_score  : mean(Ŷ) — average predicted return across horizons
2. slope_score : Ŷ[H-1] - Ŷ[0] — term structure slope (rising = positive)
3. norm_score  : ||Ŷ||_L2 / sqrt(H) — normalised prediction strength

References
----------
- Kadri, H., Duflos, E., Preux, P., Canu, S., Rakotomamonjy, A. &
  Audiffren, J. (2016). Operator-valued kernels for learning from functional
  response data. JMLR, 17(20), 1–54.
- Micchelli, C.A. & Pontil, M. (2005). On learning vector-valued functions.
  Neural Computation, 17(1), 177–204.
- Ramsay, J.O. & Silverman, B.W. (2005). Functional Data Analysis (2nd ed.).
  Springer.
- Wahba, G. (1990). Spline Models for Observational Data. SIAM.
- Berlinet, A. & Thomas-Agnan, C. (2004). Reproducing Kernel Hilbert Spaces
  in Probability and Statistics. Springer.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

import config


# ── Functional kernels ────────────────────────────────────────────────────────

def _sobolev_w12_matrix(F: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    Sobolev W^{1,2} kernel matrix between two sets of paths.

    k(f, g) = (1/T) * [f @ g  +  f' @ g']

    F: (N1, T)  — N1 input paths of length T
    G: (N2, T)  — N2 input paths of length T
    Returns: (N1, N2) kernel matrix
    """
    T = F.shape[1]
    # Derivatives (first differences, prepend first value)
    dF = np.diff(F, axis=1, prepend=F[:, :1])   # (N1, T)
    dG = np.diff(G, axis=1, prepend=G[:, :1])   # (N2, T)

    l2_term  = (F @ G.T) / T
    h1_term  = (dF @ dG.T) / T
    return l2_term + h1_term


def _l2_matrix(F: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Pure L2 kernel: k(f,g) = (1/T) * integral f(t) g(t) dt"""
    T = F.shape[1]
    return (F @ G.T) / T


def _rbf_path_matrix(F: np.ndarray, G: np.ndarray,
                      sigma: float) -> np.ndarray:
    """
    Gaussian RBF on path L2 distance:
    k(f,g) = exp(-||f-g||^2 / (2*sigma^2*T))
    """
    T = F.shape[1]
    # Pairwise squared L2 distances
    F_sq = np.sum(F**2, axis=1, keepdims=True)       # (N1, 1)
    G_sq = np.sum(G**2, axis=1, keepdims=True).T     # (1, N2)
    FG   = F @ G.T                                    # (N1, N2)
    dist_sq = (F_sq + G_sq - 2*FG) / T
    return np.exp(-dist_sq / (2 * sigma**2))


def _kernel_matrix(F: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Dispatch to the configured kernel."""
    if config.KERNEL == "sobolev_w12":
        return _sobolev_w12_matrix(F, G)
    elif config.KERNEL == "l2":
        return _l2_matrix(F, G)
    elif config.KERNEL == "rbf_path":
        return _rbf_path_matrix(F, G, config.RBF_SIGMA)
    else:
        return _sobolev_w12_matrix(F, G)


# ── Dataset construction ──────────────────────────────────────────────────────

def _build_functional_dataset(
    log_ret:    np.ndarray,   # (T,) full log return history
    macro_norm: np.ndarray,   # (T, M) normalised macro signals
    window:     int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build functional training dataset.

    Input paths  X[i]: log_ret[t-SEQ_LEN:t]  — return path of length SEQ_LEN
    Output paths Y[i]: cumulative log_ret[t:t+h] for h=1..PRED_HORIZON

    Each input path is augmented with macro signals as additional channels,
    concatenated along the feature axis: X_aug[i] shape = (SEQ_LEN * (1+M),)
    treated as a flat function sample.

    Returns
    -------
    X : (N, SEQ_LEN*(1+M)) — flattened augmented input paths
    Y : (N, PRED_HORIZON)  — output return paths
    """
    T   = len(log_ret)
    M   = macro_norm.shape[1]
    L   = config.SEQ_LEN
    H   = config.PRED_HORIZON

    X_rows, Y_rows = [], []

    # Use only samples within the rolling window
    start = max(L, T - window)
    end   = T - H

    for t in range(start, end):
        # Input: return path + macro channels
        ret_path = log_ret[t-L:t]           # (L,)
        mac_path = macro_norm[t-L:t]        # (L, M)

        # Concatenate channels: [ret, mac_1, ..., mac_M] flattened
        aug = np.concatenate([ret_path, mac_path.ravel()])  # (L*(1+M),)

        if np.isnan(aug).any():
            continue

        # Output: cumulative return at each horizon 1..H
        cum_ret = np.array([
            log_ret[t:t+h].sum() for h in range(1, H+1)
        ])                                  # (H,)

        if np.isnan(cum_ret).any():
            continue

        X_rows.append(aug)
        Y_rows.append(cum_ret)

    if not X_rows:
        return np.empty((0, L*(1+M))), np.empty((0, H))

    return np.array(X_rows), np.array(Y_rows)


# ── Operator-valued KRR ───────────────────────────────────────────────────────

def _fit_ovkrr(
    X_train: np.ndarray,   # (N, D) — flattened functional inputs
    Y_train: np.ndarray,   # (N, H) — functional outputs
    lam:     float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit operator-valued KRR.

    Solves: (K + λI) α = Y_train
    where K[i,j] = k(X_train[i], X_train[j]) with X treated as functions.

    Returns
    -------
    alpha : (N, H) — dual coefficients
    X_train : (N, D) — stored for prediction
    """
    # Reshape to (N, L*(1+M)) → treat as N paths of length L*(1+M)
    # Actually keep flat for kernel computation
    K = _kernel_matrix(X_train, X_train)   # (N, N)
    N = len(K)
    A = K + lam * np.eye(N)
    try:
        alpha = np.linalg.solve(A, Y_train)   # (N, H)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(A, Y_train, rcond=None)[0]
    return alpha, X_train


def _predict_ovkrr(
    x_test:  np.ndarray,   # (D,) single test point
    X_train: np.ndarray,   # (N, D)
    alpha:   np.ndarray,   # (N, H)
) -> np.ndarray:
    """
    Predict return path for a single test input.
    Returns: (H,) predicted cumulative return path
    """
    # k(x_test, X_train): shape (1, N)
    k_vec = _kernel_matrix(x_test[None, :], X_train)   # (1, N)
    return (k_vec @ alpha).ravel()                       # (H,)


# ── Score components ──────────────────────────────────────────────────────────

def _score_from_path(y_hat: np.ndarray) -> float:
    """
    Compute composite score from predicted return path y_hat: (H,)
    """
    H = len(y_hat)
    if H == 0:
        return 0.0

    # 1. Mean predicted return (per-day average over horizon)
    mean_score = float(y_hat.mean())

    # 2. Term structure slope (rising → momentum, falling → mean-reversion)
    if H > 1:
        slope_score = float(y_hat[-1] - y_hat[0])
    else:
        slope_score = 0.0

    # 3. Normalised L2 norm of predicted path (signal strength)
    norm_score = float(np.sqrt(np.mean(y_hat**2)))

    return (
        config.WEIGHT_MEAN  * mean_score
        + config.WEIGHT_SLOPE * slope_score
        + config.WEIGHT_NORM  * norm_score
    )


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_rkhs_scores(
    prices:    pd.DataFrame,
    macro_df:  pd.DataFrame,
    tickers:   List[str],
    window:    int,
) -> pd.Series:
    """
    Fit RKHS functional regression per ETF and return cross-sectional z-scores.

    For each ETF over the rolling window:
      1. Build functional dataset: input paths X (return paths) and
         output paths Y (term structure of forward returns)
      2. Fit operator-valued KRR in the Sobolev W^{1,2} RKHS
      3. Predict today's return path: Ŷ(h) for h=1..PRED_HORIZON
      4. Score = weighted combination of mean(Ŷ), slope(Ŷ), ||Ŷ||

    Parameters
    ----------
    prices   : DataFrame of closing prices, DatetimeIndex
    macro_df : DataFrame of macro signal levels, DatetimeIndex
    tickers  : list of ETF tickers in this universe
    window   : lookback window in trading days

    Returns
    -------
    pd.Series indexed by ticker, values = composite RKHS z-score
    """
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return pd.Series(dtype=float)

    min_rows = window + config.PRED_HORIZON + config.SEQ_LEN + 5
    if len(prices) < min_rows:
        return pd.Series(dtype=float)

    # Align macro
    common    = prices.index.intersection(macro_df.index) if not macro_df.empty else prices.index
    prices_a  = prices.loc[common]
    macro_a   = macro_df.loc[common] if not macro_df.empty else pd.DataFrame(index=common)

    macro_vals = macro_a.values.astype(np.float64) if not macro_a.empty else np.zeros((len(common), 0))
    if macro_vals.shape[1] > 0:
        m_mu       = np.nanmean(macro_vals, axis=0, keepdims=True)
        m_std      = np.nanstd(macro_vals,  axis=0, keepdims=True) + 1e-8
        macro_norm = np.nan_to_num((macro_vals - m_mu) / m_std, 0.0)
    else:
        macro_norm = macro_vals

    raw_scores = {}

    for ticker in avail:
        price_series = prices_a[ticker].dropna()
        if len(price_series) < min_rows:
            continue

        log_ret = np.log(price_series / price_series.shift(1)).dropna().values
        mac     = macro_norm[-len(log_ret):]
        if len(mac) < len(log_ret):
            log_ret = log_ret[-len(mac):]

        # Build functional dataset
        X, Y = _build_functional_dataset(log_ret, mac, window)

        if len(X) < config.MIN_SAMPLES:
            print(f"    {ticker}: only {len(X)} samples, skipping")
            continue

        print(f"    Fitting RKHS for {ticker} "
              f"(N={len(X)}, D={X.shape[1]}, H={Y.shape[1]}, kernel={config.KERNEL})")

        # Fit operator-valued KRR
        try:
            alpha, X_train = _fit_ovkrr(X, Y, lam=config.KRR_LAMBDA)
        except Exception as e:
            print(f"    KRR failed {ticker}: {e}")
            continue

        # Build today's input path
        L = config.SEQ_LEN
        M = macro_norm.shape[1]
        today_ret = log_ret[-L:]
        today_mac = mac[-L:]
        x_today   = np.concatenate([today_ret, today_mac.ravel()])

        if np.isnan(x_today).any():
            continue

        # Predict return path
        try:
            y_hat = _predict_ovkrr(x_today, X_train, alpha)
        except Exception as e:
            print(f"    Prediction failed {ticker}: {e}")
            continue

        # Clip extremes
        y_hat = np.clip(y_hat, -0.5, 0.5)

        score = _score_from_path(y_hat)
        print(f"    {ticker}: mean={y_hat.mean():.5f}  "
              f"slope={float(y_hat[-1]-y_hat[0]):.5f}  score={score:.5f}")

        raw_scores[ticker] = score

    if not raw_scores:
        return pd.Series(dtype=float)

    scores = pd.Series(raw_scores)
    mu, std = scores.mean(), scores.std()
    if std < 1e-10:
        return pd.Series(0.0, index=scores.index)
    return (scores - mu) / std
