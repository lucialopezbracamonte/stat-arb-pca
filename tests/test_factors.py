"""Sanity-check the PCA factor model against synthetic data with a known,
planted factor structure -- if this doesn't recover the ground truth, nothing
built on top of it (residuals, OU fit, backtest) can be trusted either.
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.factors import pca_eigenportfolios, factor_returns, residualize


def test_pca_recovers_single_factor_structure():
    rng = np.random.RandomState(0)
    n_days, n_assets = 500, 20
    true_factor = rng.normal(0, 0.01, n_days)
    betas = rng.uniform(0.5, 1.5, n_assets)
    idio_vol = 0.002
    idio = rng.normal(0, idio_vol, (n_days, n_assets))
    R = true_factor[:, None] * betas[None, :] + idio
    returns = pd.DataFrame(R, columns=[f"A{i}" for i in range(n_assets)])

    Q, explained = pca_eigenportfolios(returns, var_threshold=0.5, max_factors=5)
    # A single dominant factor should explain the overwhelming majority of variance.
    assert explained[0] > 0.8, f"top factor only explains {explained[0]:.2f}"

    F = factor_returns(returns, Q)
    corr = np.corrcoef(F.iloc[:, 0], true_factor)[0, 1]
    assert abs(corr) > 0.9, f"recovered factor correlation with truth = {corr:.2f}"


def test_residualize_removes_common_factor():
    rng = np.random.RandomState(1)
    n_days, n_assets = 400, 15
    true_factor = rng.normal(0, 0.012, n_days)
    betas = rng.uniform(0.5, 1.5, n_assets)
    idio = rng.normal(0, 0.001, (n_days, n_assets))
    R = true_factor[:, None] * betas[None, :] + idio
    returns = pd.DataFrame(R, columns=[f"A{i}" for i in range(n_assets)])

    Q, _ = pca_eigenportfolios(returns, var_threshold=0.5, max_factors=3)
    F = factor_returns(returns, Q)
    _, resid = residualize(returns, F)

    # Residual variance should be far smaller than raw return variance --
    # the systematic factor has been regressed out.
    assert resid.var().mean() < 0.3 * returns.var().mean()


if __name__ == "__main__":
    test_pca_recovers_single_factor_structure()
    test_residualize_removes_common_factor()
    print("All factors tests passed.")
