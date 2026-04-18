"""Simulate an exact OU process with known (kappa, m, sigma) and check that
fit_ou_params recovers parameters close to the ground truth -- and that the
trading state machine actually produces the open/close transitions it should.
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signal import fit_ou_params, s_scores, generate_positions

TRADING_DAYS = 252


def simulate_ou(kappa, m, sigma, n_days, dt=1 / TRADING_DAYS, seed=0):
    rng = np.random.RandomState(seed)
    x = np.zeros(n_days)
    x[0] = m
    b = np.exp(-kappa * dt)
    eps_std = sigma * np.sqrt(1 - b ** 2)
    for t in range(1, n_days):
        x[t] = m + b * (x[t - 1] - m) + rng.normal(0, eps_std)
    return x


def test_fit_ou_params_recovers_known_parameters():
    kappa_true, m_true, sigma_true = 15.0, 0.0, 0.02
    x = simulate_ou(kappa_true, m_true, sigma_true, n_days=1000, seed=42)
    df = pd.DataFrame({"SIM": x})

    params = fit_ou_params(df)
    kappa_hat = params.loc["SIM", "kappa"]
    m_hat = params.loc["SIM", "m"]
    sigma_hat = params.loc["SIM", "sigma_eq"]

    assert abs(kappa_hat - kappa_true) / kappa_true < 0.25, f"kappa {kappa_hat} vs {kappa_true}"
    assert abs(m_hat - m_true) < 0.01, f"m {m_hat} vs {m_true}"
    assert abs(sigma_hat - sigma_true) / sigma_true < 0.25, f"sigma {sigma_hat} vs {sigma_true}"


def test_state_machine_opens_and_closes():
    # Manually construct an s-score path that should trigger short-open,
    # then close, then long-open, then close.
    path = pd.Series([0.0, 2.0, 2.0, 0.1, -2.0, -2.0, 0.1])
    s_df = pd.DataFrame({"A": path})
    tradeable = pd.Series({"A": True})

    positions = generate_positions(s_df, tradeable, s_open=1.25, s_close=0.5)
    expected = [0, -1, -1, 0, 1, 1, 0]
    assert list(positions["A"]) == expected, list(positions["A"])


if __name__ == "__main__":
    test_fit_ou_params_recovers_known_parameters()
    test_state_machine_opens_and_closes()
    print("All signal tests passed.")
