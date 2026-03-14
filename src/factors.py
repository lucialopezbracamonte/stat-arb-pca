"""Statistical risk factors via PCA (Avellaneda & Lee, 2010 style).

Core idea: instead of assuming a fundamental factor model (industry, value,
momentum...), estimate the factors from returns alone. PCA on the standardized daily-return
covariance matrix of a broad cross-section of stocks recovers a small set
of orthogonal "eigenportfolios" that explain most of the systematic
co-movement -- empirically, the top eigenportfolio is almost always a
market-like factor, and the next several correspond to sector/style
tilts. What's left over per stock after regressing out these factors is
its idiosyncratic residual, which is the object we model as mean-reverting.
"""
import numpy as np
import pandas as pd


def standardize_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Z-score each column by its own in-window mean/std.

    PCA on raw returns would be dominated by whichever stocks happen to be
    most volatile in the window; standardizing puts every stock on equal
    footing so the eigenvectors reflect correlation structure, not variance.
    """
    return (returns - returns.mean()) / returns.std(ddof=1)


def pca_eigenportfolios(returns: pd.DataFrame, var_threshold: float = 0.55,
                         max_factors: int = 15) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit PCA on a trailing window of returns, return eigenportfolio weights.

    Q is (k factors x n assets): row j is the dollar-weight of each stock in
    eigenportfolio j, scaled so that eigenportfolio returns have unit-ish
    exposure to that stock's own volatility (Q_ji = v_ji / sigma_i), matching
    the construction in Avellaneda & Lee (2010) so the factor returns are in
    the same units as the underlying stock returns.

    Returns
    -------
    Q : DataFrame, index = factor id, columns = tickers
    explained_variance_ratio : array of length k
    """
    z = standardize_returns(returns)
    sigma = returns.std(ddof=1).values  # per-asset vol, for de-standardizing loadings

    corr = np.corrcoef(z.values.T)  # n_assets x n_assets, PCA on correlation matrix
    eigvals, eigvecs = np.linalg.eigh(corr)  # ascending order
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    explained = eigvals / eigvals.sum()
    cum = np.cumsum(explained)
    k = int(np.searchsorted(cum, var_threshold) + 1)
    k = max(1, min(k, max_factors, len(eigvals)))

    V = eigvecs[:, :k]  # n_assets x k
    Q = (V / sigma[:, None]).T  # k x n_assets, eigenportfolio weights
    Q = pd.DataFrame(Q, columns=returns.columns, index=[f"F{i+1}" for i in range(k)])
    return Q, explained[:k]


def factor_returns(returns: pd.DataFrame, Q: pd.DataFrame) -> pd.DataFrame:
    """Project asset returns onto eigenportfolios: F_t = returns_t @ Q.T."""
    return returns @ Q.T


def residualize(returns: pd.DataFrame, factors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OLS-regress each stock's returns on the factor returns; return betas + residuals.

    betas : DataFrame, index = ticker, columns = factor id (plus 'alpha')
    resid : DataFrame, same shape as `returns` -- idiosyncratic daily returns
    """
    X = np.column_stack([np.ones(len(factors)), factors.values])  # T x (k+1), intercept first
    Y = returns.values  # T x n_assets
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)  # (k+1) x n_assets
    fitted = X @ coef
    resid = Y - fitted

    betas = pd.DataFrame(
        coef.T, index=returns.columns, columns=["alpha"] + list(factors.columns),
    )
    resid = pd.DataFrame(resid, index=returns.index, columns=returns.columns)
    return betas, resid
