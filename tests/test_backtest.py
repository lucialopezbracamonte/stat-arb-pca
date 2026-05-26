"""The factor hedge is the most subtle piece of the backtest: holding stock i
should also short beta_ij dollars of eigenportfolio j, so the net book has
(approximately) zero exposure to every statistical factor.

In-sample, OLS guarantees Q @ B = I_k (up to intercept/mean terms), so the
dollar-holdings vector h built by _build_holdings must satisfy h @ B ~ 0.
If the hedge construction breaks, this test catches it.
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.factors import pca_eigenportfolios, factor_returns, residualize
from src.backtest import BacktestConfig, _build_holdings


def _synthetic_market(seed=3, n_days=300, n_assets=25, n_factors=3):
    rng = np.random.RandomState(seed)
    F = rng.normal(0, 0.01, (n_days, n_factors))
    B = rng.uniform(-1.0, 1.5, (n_assets, n_factors))
    idio = rng.normal(0, 0.004, (n_days, n_assets))
    R = F @ B.T + idio
    return pd.DataFrame(R, columns=[f"A{i}" for i in range(n_assets)])


def test_holdings_are_factor_neutral():
    returns = _synthetic_market()
    Q, _ = pca_eigenportfolios(returns, var_threshold=0.55, max_factors=6)
    F = factor_returns(returns, Q)
    betas, _ = residualize(returns, F)

    cfg = BacktestConfig()
    # force a mixed book: strong short signal on a few names, strong long on others
    s = pd.Series(0.0, index=returns.columns)
    s.iloc[:4] = 2.0
    s.iloc[4:8] = -2.0
    tradeable = pd.Series(True, index=returns.columns)
    prev = pd.Series(0.0, index=returns.columns)

    holdings, state = _build_holdings(prev, s, tradeable, betas, Q, cfg, returns.columns)
    assert (state != 0).sum() == 8

    B = betas.drop(columns="alpha").values          # n_assets x k
    exposure = holdings.values @ B                   # k-vector of factor exposures
    stock_leg_exposure = np.abs(
        (state.values * (cfg.target_gross / 8)) @ B
    )
    # hedged exposure should be tiny relative to the unhedged stock leg's
    assert np.all(np.abs(exposure) < 0.05 * np.maximum(stock_leg_exposure, 1e-6)), (
        exposure, stock_leg_exposure)


if __name__ == "__main__":
    test_holdings_are_factor_neutral()
    print("All backtest tests passed.")
