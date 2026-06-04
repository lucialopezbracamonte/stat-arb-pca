"""Robustness variants of the walk-forward backtest.

Every claim in the notebook of the form "X was tested and ..." is backed by a
run in this file -- the notebook renders results/robustness.json directly, so
the narrative can't drift from the numbers.

Takes ~8 minutes (the daily-refit variant alone is ~4 min).
"""
import json

from src.data import load_returns
from src.backtest import BacktestConfig, run_backtest, performance_summary

VARIANTS = {
    "default (60d window, weekly refit, 55% var threshold)": BacktestConfig(),
    "daily refit (paper's cadence)": BacktestConfig(refit_every=1),
    "fixed 5 factors (no variance threshold)": BacktestConfig(var_threshold=1.0, max_factors=5),
    "A&L window split (252d PCA / 60d OU)": BacktestConfig(pca_lookback=252),
    "variance threshold 45%": BacktestConfig(var_threshold=0.45),
    "variance threshold 65%": BacktestConfig(var_threshold=0.65),
}


def main():
    returns = load_returns()
    out = {}
    for name, cfg in VARIANTS.items():
        result = run_backtest(returns, cfg)
        s = performance_summary(result.daily_returns)
        rebal = result.turnover[result.turnover > 0]
        years = len(result.daily_returns) / 252
        out[name] = {
            "sharpe": round(float(s["sharpe"]), 3),
            "annualized_return": round(float(s["annualized_return"]), 4),
            "max_drawdown": round(float(s["max_drawdown"]), 3),
            "avg_turnover_per_rebalance": round(float(rebal.mean()), 3),
            "annualized_cost_drag": round(float(rebal.sum() * cfg.tc_bps / 10000 / years), 4),
        }
        print(name, "->", out[name])

    with open("results/robustness.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote results/robustness.json")


if __name__ == "__main__":
    main()
