"""Static result plots. Matplotlib, light-mode, fixed palette: blue = primary
series, red = negative/short, gray = neutral/drawdown, muted olive-gray for
axes/gridlines.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#2a78d6"
RED = "#e34948"
GREEN = "#0ca30c"
GRAY_FILL = "#f0efec"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_SECONDARY,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": GRID, "font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_equity_and_drawdown(daily_returns: pd.Series, path: str):
    cum = (1 + daily_returns).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(cum.index, cum.values, color=BLUE, linewidth=1.5)
    ax1.axhline(1.0, color=MUTED, linewidth=0.75, linestyle="--")
    ax1.set_ylabel("Cumulative return\n(growth of $1)")
    ax1.set_title("PCA statistical arbitrage: cumulative return and drawdown", loc="left")
    ax1.grid(axis="y", linewidth=0.5)

    ax2.fill_between(drawdown.index, drawdown.values, 0, color=RED, alpha=0.35, linewidth=0)
    ax2.plot(drawdown.index, drawdown.values, color=RED, linewidth=1.0)
    ax2.set_ylabel("Drawdown")
    ax2.grid(axis="y", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_subperiod_sharpe(subperiod_sharpes: dict, path: str):
    labels = list(subperiod_sharpes.keys())
    values = list(subperiod_sharpes.values())
    colors = [GREEN if v > 0 else RED for v in values]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.axhline(0, color=MUTED, linewidth=0.75)
    ax.set_ylabel("Annualized Sharpe")
    ax.set_title("Sharpe ratio decays across sub-periods", loc="left")
    ax.grid(axis="y", linewidth=0.5)
    for b, v in zip(bars, values):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                     textcoords="offset points", xytext=(0, 6 if v >= 0 else -14),
                     ha="center", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_factor_structure(diagnostics: list, path: str):
    df = pd.DataFrame(diagnostics).set_index("date")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(df.index, df["n_factors"], color=BLUE, linewidth=1.2)
    ax1.set_ylabel("# factors retained")
    ax1.set_title("Statistical risk-factor structure over time", loc="left")
    ax1.grid(axis="y", linewidth=0.5)

    ax2.plot(df.index, df["n_tradeable"], color=BLUE, linewidth=1.2, label="tradeable (fast mean-reversion)")
    ax2.plot(df.index, df["n_active"], color=RED, linewidth=1.2, label="active positions")
    ax2.set_ylabel("# stocks")
    ax2.legend(frameon=False, loc="center left", fontsize=9)
    ax2.grid(axis="y", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bootstrap_sharpe(boot_sharpes: np.ndarray, point: float, ci: tuple, path: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(boot_sharpes, bins=60, color=BLUE, alpha=0.75, edgecolor="none")
    ax.axvline(point, color=INK, linewidth=1.5, label=f"observed Sharpe = {point:.2f}")
    ax.axvline(0, color=MUTED, linewidth=1.0, linestyle="--", label="zero")
    ax.axvspan(ci[0], ci[1], color=GRAY_FILL, alpha=0.6, label="90% bootstrap CI")
    ax.set_xlabel("Sharpe ratio")
    ax.set_ylabel("Bootstrap draws")
    ax.set_title("Block-bootstrap distribution of the Sharpe ratio", loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_permutation_null(null_sharpes: np.ndarray, real_sharpe: float, path: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(null_sharpes, bins=20, color=MUTED, alpha=0.7, edgecolor="none",
            label="null: s-scores shuffled across tickers")
    ax.axvline(real_sharpe, color=RED, linewidth=1.75, label=f"actual strategy = {real_sharpe:.2f}")
    ax.set_xlabel("Sharpe ratio")
    ax.set_ylabel("Permutation runs")
    ax.set_title("Is the signal doing anything a random assignment wouldn't?", loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
