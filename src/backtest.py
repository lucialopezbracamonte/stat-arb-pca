"""Walk-forward backtest of the PCA statistical arbitrage strategy.

Every rebalance, the factor model and OU parameters are re-estimated on a
trailing window that ends the day *before* trading starts -- no parameter
is ever fit on data the strategy wouldn't have had in hand at decision
time. Holdings are held fixed between rebalances and marked to market
daily against realized returns.

Each stock position is factor-hedged: holding stock i also means shorting
beta_ij dollars of eigenportfolio j for every factor j it loaded on, so the
position genuinely isolates the idiosyncratic bet rather than smuggling in
directional/factor risk. That hedge is just matrix multiplication of the
beta and eigenportfolio-weight matrices.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.factors import pca_eigenportfolios, factor_returns, residualize
from src.signal import fit_ou_params, s_scores, is_tradeable, TRADING_DAYS


@dataclass
class BacktestConfig:
    lookback: int = 60             # trailing window for the OU fit (and, by default,
                                    # the PCA) -- 60 days is A&L (2010)'s OU window
    pca_lookback: int | None = None  # A&L estimate PCA on 252 days but the OU fit on
                                      # 60; set this to 252 to reproduce their split.
                                      # None = use `lookback` for both (tested: the
                                      # split performs comparably, see notebook)
    refit_every: int = 5           # weekly refit; daily refit was tested and more
                                    # than doubles transaction-cost drag for
                                    # negligible signal gain -- see notebook
    var_threshold: float = 0.55
    max_factors: int = 15
    max_half_life: float = 15.0
    min_r_squared: float = 0.0
    s_open: float = 1.25
    s_close: float = 0.5
    target_gross: float = 1.0      # gross dollar exposure of the idiosyncratic leg
    tc_bps: float = 5.0            # one-way transaction cost, in bps of notional traded
    permute_seed: int | None = None  # if set, shuffle s-score -> ticker assignment each
                                      # rebalance (null-hypothesis test, see src/stats.py)


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    holdings_history: pd.DataFrame
    turnover: pd.Series
    n_active_history: pd.Series
    diagnostics: list = field(default_factory=list)


def _build_holdings(prev_state: pd.Series, s: pd.Series, tradeable: pd.Series,
                     betas: pd.DataFrame, Q: pd.DataFrame, cfg: BacktestConfig,
                     columns: pd.Index) -> tuple[pd.Series, pd.Series]:
    """One state-machine step -> per-asset dollar holdings (stock leg + factor hedge)."""
    state = prev_state.reindex(s.index).fillna(0.0)
    long_open = (s < -cfg.s_open) & tradeable
    short_open = (s > cfg.s_open) & tradeable
    close = s.abs() < cfg.s_close
    state = state.where(~long_open, 1.0)
    state = state.where(~short_open, -1.0)
    state = state.where(~close, 0.0)
    state = state.where(tradeable, 0.0)

    n_active = int((state != 0).sum())
    if n_active == 0:
        return pd.Series(0.0, index=columns), state

    # Stock leg: equal signed dollar weight per active name. Hedge leg: for each
    # factor j, short (sum_i p_i * beta_ij) dollars of eigenportfolio j. Both legs
    # together are two matrix products:
    #   holdings = p - (p @ B) @ Q
    # where p is the stock-leg weight vector, B (n x k) the factor betas, and
    # Q (k x n) the eigenportfolio weights.
    stock_weight = state * (cfg.target_gross / n_active)
    B = betas.drop(columns="alpha")            # n_assets x k
    factor_dollars = stock_weight @ B          # k-vector: dollars of each eigenportfolio held
    holdings = stock_weight - factor_dollars @ Q
    return holdings.reindex(columns).fillna(0.0), state


def run_backtest(returns: pd.DataFrame, cfg: BacktestConfig = BacktestConfig()) -> BacktestResult:
    dates = returns.index
    daily_ret = pd.Series(0.0, index=dates)
    holdings_hist = pd.DataFrame(0.0, index=dates, columns=returns.columns)
    turnover = pd.Series(0.0, index=dates)
    n_active_hist = pd.Series(0, index=dates)
    diagnostics = []

    holdings = pd.Series(0.0, index=returns.columns)
    state = pd.Series(0.0, index=returns.columns)

    warmup = max(cfg.lookback, cfg.pca_lookback or 0)
    rebalance_idx = set(range(warmup, len(dates), cfg.refit_every))

    for i in range(warmup, len(dates)):
        today = dates[i]

        if i in rebalance_idx:
            window = returns.iloc[i - cfg.lookback:i]  # strictly before `today`
            pca_window = returns.iloc[i - (cfg.pca_lookback or cfg.lookback):i]
            Q, explained = pca_eigenportfolios(pca_window, cfg.var_threshold, cfg.max_factors)
            F = factor_returns(window, Q)
            betas, resid = residualize(window, F)
            cum_resid = resid.cumsum()
            ou_params = fit_ou_params(cum_resid)
            tradeable = is_tradeable(ou_params, cfg.max_half_life, cfg.min_r_squared)
            s_path = s_scores(cum_resid, ou_params)
            current_s = s_path.iloc[-1]

            if cfg.permute_seed is not None:
                rng = np.random.RandomState(cfg.permute_seed * 100003 + i)
                trade_names = current_s.index[tradeable]
                shuffled = rng.permutation(current_s.loc[trade_names].values)
                current_s = current_s.copy()
                current_s.loc[trade_names] = shuffled

            new_holdings, state = _build_holdings(state, current_s, tradeable, betas, Q, cfg, returns.columns)
            traded_notional = (new_holdings - holdings).abs().sum()
            cost = traded_notional * cfg.tc_bps / 10000.0
            holdings = new_holdings
            turnover.loc[today] = traded_notional
            n_active_hist.loc[today] = int((state != 0).sum())
            diagnostics.append(dict(
                date=today, n_factors=len(explained),
                explained_var=float(np.sum(explained)),
                n_tradeable=int(tradeable.sum()), n_active=int((state != 0).sum()),
            ))
        else:
            cost = 0.0

        gross_ret = float((holdings * returns.loc[today]).sum())
        daily_ret.loc[today] = gross_ret - cost
        holdings_hist.loc[today] = holdings

    valid = daily_ret.index[warmup:]
    return BacktestResult(
        daily_returns=daily_ret.loc[valid],
        holdings_history=holdings_hist.loc[valid],
        turnover=turnover.loc[valid],
        n_active_history=n_active_hist.loc[valid],
        diagnostics=diagnostics,
    )


def performance_summary(daily_returns: pd.Series) -> dict:
    ann_ret = daily_returns.mean() * TRADING_DAYS
    ann_vol = daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + daily_returns).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_dd = drawdown.min()
    hit_rate = (daily_returns > 0).mean()
    return dict(
        annualized_return=ann_ret, annualized_vol=ann_vol, sharpe=sharpe,
        max_drawdown=max_dd, hit_rate=hit_rate, n_days=len(daily_returns),
    )
