# ∫ P2-ETF-RKHS-FUNCTIONAL-REGRESSION

**RKHS Functional Regression Engine — Operator-Valued KRR (Kadri et al. 2016)**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine treats each ETF's return history as a **function** (path) in a
Reproducing Kernel Hilbert Space, then uses **operator-valued Kernel Ridge
Regression** to predict the full term structure of forward returns — not just
a scalar expected return, but the entire return curve over horizons h=1..21.

---

## Theory

### RKHS and Functional Kernels

An RKHS H is a Hilbert space of functions where the reproducing property holds:

```
f(x) = <f, k(x,·)>_H
```

For return paths f, g of length T, the **Sobolev W(1,2) kernel** is:

```
k(f, g) = (1/T) [integral f(t)g(t) dt  +  integral f'(t)g'(t) dt]
```

The two terms capture level similarity and derivative (momentum) similarity.

### Operator-Valued KRR (Kadri et al. 2016)

Standard KRR: scalar response. Operator-valued KRR: the response is a
**function** — the full return path over horizons h=1..PRED_HORIZON.

```
Y[i,h] = cumulative log return from t_i to t_i + h    (N x H matrix)
K[i,j] = k(X_i, X_j)                                  (N x N kernel matrix)
alpha   = (K + lambda*I)^{-1} Y                        (N x H coefficients)
Yhat(x) = k_vec(x) @ alpha                             (H predicted returns)
```

This gives a predicted return curve: Yhat[h] = expected cumulative return at
horizon h, for h = 1..21 days.

### Score Construction

```
score = 0.60 * mean(Yhat)  +  0.25 * (Yhat[H] - Yhat[1])  +  0.15 * norm(Yhat)
```

| Component | Meaning |
|-----------|---------|
| mean(Yhat) | Average predicted return — primary direction signal |
| Yhat[H] - Yhat[1] | Term structure slope — rising curve = momentum |
| norm(Yhat) | Signal strength — how much the model is predicting |

---

## Distinction from Other Suite Engines

| Engine | Method | Output |
|--------|--------|--------|
| FPCR-MACRO | Functional PCA then scalar regression | Scalar return |
| SKR | Signature kernel (iterated integrals) | Scalar return |
| QRF | Non-parametric quantile forest | Scalar quantiles |
| **RKHS-FR (this engine)** | **Sobolev kernel + operator-valued KRR** | **Full return path** |

The unique feature is the **operator-valued output**: a full predicted return
curve, not just a scalar. This gives the term structure of expectations.

---

## Universes & Windows

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

**Windows:** `63d · 126d · 252d · 504d`

---

## Repository Structure

```
P2-ETF-RKHS-FUNCTIONAL-REGRESSION/
├── config.py          # Universes, kernel choice, KRR lambda, score weights
├── data_manager.py    # HuggingFace loader → (prices, macro) DataFrames
├── rkhs_engine.py     # Core: Sobolev kernel, operator-valued KRR, scoring
├── trainer.py         # Orchestrator: load → fit → score → JSON → upload
├── push_results.py    # HfApi.upload_file wrapper
├── streamlit_app.py   # Two-tab Streamlit dashboard
├── us_calendar.py     # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml  # Single job (kernel matrix is O(N^2 L) — very fast)
```

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-RKHS-FUNCTIONAL-REGRESSION
cd P2-ETF-RKHS-FUNCTIONAL-REGRESSION
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-rkhs-functional-results`

---

## References

- Kadri, H., Duflos, E., Preux, P., Canu, S., Rakotomamonjy, A. &
  Audiffren, J. (2016). Operator-valued kernels for learning from functional
  response data. *JMLR*, 17(20), 1–54.
- Micchelli, C.A. & Pontil, M. (2005). On learning vector-valued functions.
  *Neural Computation*, 17(1), 177–204.
- Ramsay, J.O. & Silverman, B.W. (2005). *Functional Data Analysis* (2nd ed.).
  Springer.
- Wahba, G. (1990). *Spline Models for Observational Data*. SIAM.
