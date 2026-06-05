"""End-to-end pipeline: load data, run the walk-forward backtest, validate
statistically, generate every plot and number used in the README / notebook.
"""
import json
import pickle

from src.data import load_returns
from src.backtest import BacktestConfig, run_backtest, performance_summary
from src.stats import block_bootstrap
from src.plots import (
    plot_equity_and_drawdown, plot_subperiod_sharpe,
    plot_factor_structure, plot_bootstrap_sharpe, plot_permutation_null,
)

SUBPERIODS = [
    ("2010", "2013", "2010-2013"),
    ("2014", "2017", "2014-2017"),
    ("2018", "2021", "2018-2021"),
    ("2022", "2026", "2022-2026"),
]


def main():
    returns = load_returns()
    cfg = BacktestConfig()
    result = run_backtest(returns, cfg)
    summary = performance_summary(result.daily_returns)
    print("Full-sample performance:", summary)

    subperiod_sharpes = {}
    for start, end, label in SUBPERIODS:
        sub = result.daily_returns.loc[start:end]
        if len(sub) > 50:
            subperiod_sharpes[label] = performance_summary(sub)["sharpe"]
    print("Sub-period Sharpes:", subperiod_sharpes)

    boot = block_bootstrap(result.daily_returns, n_boot=5000, block_size=20)
    print("Bootstrap Sharpe CI90:", boot["sharpe_ci90"], "p(mean<=0):", boot["p_value_mean_leq_0"])

    plot_equity_and_drawdown(result.daily_returns, "results/equity_drawdown.png")
    plot_subperiod_sharpe(subperiod_sharpes, "results/subperiod_sharpe.png")
    plot_factor_structure(result.diagnostics, "results/factor_structure.png")
    plot_bootstrap_sharpe(boot["boot_sharpes"], boot["sharpe_point"], boot["sharpe_ci90"],
                           "results/bootstrap_sharpe.png")

    with open("results/permutation_result.pkl", "rb") as f:
        perm = pickle.load(f)
    plot_permutation_null(perm["null_sharpes"], perm["real_sharpe"], "results/permutation_null.png")
    print("Permutation p-value:", perm["p_value"], "null mean/std:", perm["null_mean"], perm["null_std"])

    out = dict(
        summary={k: float(v) for k, v in summary.items()},
        subperiod_sharpes={k: float(v) for k, v in subperiod_sharpes.items()},
        bootstrap_sharpe_ci90=[float(x) for x in boot["sharpe_ci90"]],
        bootstrap_p_value_mean_leq_0=float(boot["p_value_mean_leq_0"]),
        permutation_real_sharpe=float(perm["real_sharpe"]),
        permutation_null_mean=float(perm["null_mean"]),
        permutation_null_std=float(perm["null_std"]),
        permutation_p_value=float(perm["p_value"]),
        avg_turnover_per_rebalance=float(result.turnover[result.turnover > 0].mean()),
        avg_active_positions=float(result.n_active_history[result.n_active_history > 0].mean()),
        n_rebalances=len(result.diagnostics),
        n_factors_min=int(min(d["n_factors"] for d in result.diagnostics)),
        n_factors_max=int(max(d["n_factors"] for d in result.diagnostics)),
        annualized_cost_drag=float(
            result.turnover.sum() * cfg.tc_bps / 10000 / (len(result.daily_returns) / 252)
        ),
    )
    with open("results/summary.json", "w") as f:
        json.dump(out, f, indent=2)

    result.daily_returns.to_csv("results/daily_returns.csv")
    print("Done. Results written to results/")


if __name__ == "__main__":
    main()
