"""Statistical validation: is the backtested performance distinguishable from luck?

A single Sharpe ratio number is a point estimate with no error bar -- on
~15 years of daily data it's easy to eyeball a decent Sharpe that's really
sampling noise or an artifact of transaction-cost/turnover structure rather
than genuine predictive signal. Two independent checks:

1. Block bootstrap on the realized daily-return series -> confidence
   interval on the Sharpe ratio and a p-value for H0: mean daily return <= 0.
   Blocks (not iid resampling) because daily strategy returns are
   autocorrelated (positions are held for multiple days).

2. Signal-permutation test -> rerun the *entire* walk-forward backtest
   several times, each time randomly reassigning which tradeable stock
   gets which OU s-score before the trading rule is applied. This keeps
   turnover, transaction costs, hedge structure, and the number of active
   names identical, and destroys only the one thing the strategy claims
   to exploit: that a stock's *own* residual mean-reversion predicts its
   *own* future residual return. If real performance isn't better than
   this null, the PCA/OU machinery isn't adding anything a random
   market-neutral book wouldn't get from turnover/cost structure alone.
"""
import numpy as np
import pandas as pd

from src.backtest import BacktestConfig, run_backtest, performance_summary

TRADING_DAYS = 252


def _sharpe(returns: np.ndarray) -> float:
    vol = returns.std(ddof=1)
    return returns.mean() / vol * np.sqrt(TRADING_DAYS) if vol > 0 else np.nan


def block_bootstrap(daily_returns: pd.Series, n_boot: int = 5000, block_size: int = 20,
                     seed: int = 0) -> dict:
    """Moving-block bootstrap: resample contiguous blocks with replacement to
    preserve short-horizon autocorrelation, rebuild synthetic paths of the
    same length, and compute the Sharpe / mean of each.
    """
    rng = np.random.RandomState(seed)
    x = daily_returns.values
    n = len(x)
    n_blocks = int(np.ceil(n / block_size))
    starts_max = n - block_size

    boot_sharpes = np.empty(n_boot)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.randint(0, starts_max + 1, size=n_blocks)
        path = np.concatenate([x[s:s + block_size] for s in starts])[:n]
        boot_sharpes[b] = _sharpe(path)
        boot_means[b] = path.mean()

    sharpe_lo, sharpe_hi = np.percentile(boot_sharpes, [5, 95])
    p_value_mean_leq_0 = float(np.mean(boot_means <= 0))
    return dict(
        sharpe_point=_sharpe(x),
        sharpe_ci90=(sharpe_lo, sharpe_hi),
        boot_sharpes=boot_sharpes,
        p_value_mean_leq_0=p_value_mean_leq_0,
    )


def signal_permutation_test(returns: pd.DataFrame, cfg: BacktestConfig,
                             n_perm: int = 30, base_seed: int = 42) -> dict:
    """Compare the real strategy's Sharpe to a null built by shuffling which
    stock's residual s-score gets acted on, holding everything else fixed.
    """
    real_cfg = BacktestConfig(**{**cfg.__dict__, "permute_seed": None})
    real_result = run_backtest(returns, real_cfg)
    real_sharpe = performance_summary(real_result.daily_returns)["sharpe"]

    null_sharpes = np.empty(n_perm)
    for j in range(n_perm):
        perm_cfg = BacktestConfig(**{**cfg.__dict__, "permute_seed": base_seed + j})
        perm_result = run_backtest(returns, perm_cfg)
        null_sharpes[j] = performance_summary(perm_result.daily_returns)["sharpe"]

    p_value = float(np.mean(null_sharpes >= real_sharpe))
    return dict(
        real_sharpe=real_sharpe, null_sharpes=null_sharpes,
        null_mean=float(np.mean(null_sharpes)), null_std=float(np.std(null_sharpes)),
        p_value=p_value, real_result=real_result,
    )
