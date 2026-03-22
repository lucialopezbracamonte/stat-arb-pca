"""Mean-reversion signal: model idiosyncratic residuals as an Ornstein-Uhlenbeck process.

After the PCA factor model strips out systematic risk (src/factors.py), each
stock has a residual return series. Sum it into a cumulative "auxiliary
process" X_t. If X_t behaves like a mean-reverting OU process,

    dX_t = kappa (m - X_t) dt + sigma dW_t

then deviations from its long-run mean m are trading opportunities: short
when X_t is unusually high (expect reversion down), long when unusually low.

Estimation: the OU process's exact discrete-time solution is an AR(1),

    X_{t+dt} = a + b X_t + eps_t,   b = exp(-kappa dt)

so kappa, m, and the equilibrium (stationary) variance of X are recovered
from an OLS fit of X_t on X_{t-1} -- no numerical MLE needed.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def fit_ou_params(cum_resid: pd.DataFrame) -> pd.DataFrame:
    """Fit an AR(1) to each column's cumulative residual path.

    Returns a DataFrame indexed by ticker with columns:
    kappa (annualized mean-reversion speed), m (long-run mean),
    sigma_eq (equilibrium std dev of X), half_life (trading days),
    r_squared (AR(1) fit quality).
    """
    rows = {}
    for ticker in cum_resid.columns:
        x = cum_resid[ticker].values
        x_lag, x_now = x[:-1], x[1:]
        if len(x_lag) < 20 or np.std(x_lag) < 1e-10:
            rows[ticker] = dict(kappa=np.nan, m=np.nan, sigma_eq=np.nan,
                                 half_life=np.nan, r_squared=np.nan)
            continue
        X = np.column_stack([np.ones(len(x_lag)), x_lag])
        coef, res, *_ = np.linalg.lstsq(X, x_now, rcond=None)
        a, b = coef
        fitted = X @ coef
        ss_res = np.sum((x_now - fitted) ** 2)
        ss_tot = np.sum((x_now - x_now.mean()) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

        eps_var = ss_res / max(len(x_now) - 2, 1)
        if not (0 < b < 1):
            rows[ticker] = dict(kappa=np.nan, m=np.nan, sigma_eq=np.nan,
                                 half_life=np.nan, r_squared=r_sq)
            continue
        kappa = -np.log(b) * TRADING_DAYS
        m = a / (1 - b)
        sigma_eq_sq = eps_var / (1 - b ** 2)
        rows[ticker] = dict(
            kappa=kappa, m=m, sigma_eq=np.sqrt(max(sigma_eq_sq, 0)),
            half_life=np.log(2) / kappa * TRADING_DAYS, r_squared=r_sq,
        )
    return pd.DataFrame(rows).T


def s_scores(cum_resid: pd.DataFrame, ou_params: pd.DataFrame) -> pd.DataFrame:
    """(X_t - m) / sigma_eq for every stock, every day in the window."""
    m = ou_params["m"]
    sigma = ou_params["sigma_eq"]
    return (cum_resid - m) / sigma


def is_tradeable(ou_params: pd.DataFrame, max_half_life: float = 30.0,
                  min_r_squared: float = 0.0) -> pd.Series:
    """Only trade names whose mean reversion is fast enough to matter and
    well-estimated enough to trust -- otherwise the AR(1) fit is noise.
    """
    return (
        (ou_params["half_life"] <= max_half_life)
        & (ou_params["half_life"] > 0)
        & (ou_params["r_squared"] >= min_r_squared)
        & ou_params["kappa"].notna()
    )


def generate_positions(s_score_path: pd.DataFrame, tradeable: pd.Series,
                        s_open: float = 1.25, s_close: float = 0.5) -> pd.DataFrame:
    """State-machine trading rule (Avellaneda & Lee thresholds), applied
    day-by-day per stock so positions persist until an explicit close
    signal -- not just an instantaneous threshold crossing.

    +1 = long the residual (short overpriced factor exposure, long the stock
         after factor-hedging), -1 = short, 0 = flat.
    """
    positions = pd.DataFrame(0, index=s_score_path.index, columns=s_score_path.columns)
    state = pd.Series(0, index=s_score_path.columns)
    for t in s_score_path.index:
        s = s_score_path.loc[t]
        long_open = (s < -s_open) & tradeable
        short_open = (s > s_open) & tradeable
        close = s.abs() < s_close
        state = state.where(~long_open, 1)
        state = state.where(~short_open, -1)
        state = state.where(~close, 0)
        state = state.where(tradeable, 0)
        positions.loc[t] = state.fillna(0)
    return positions
