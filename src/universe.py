"""Trading universe: liquid large-caps spanning all 11 GICS sectors.

Picked for long, clean price history (no post-2015 IPOs) so the backtest
window isn't survivorship-biased toward recent listings. This is NOT the
full S&P 500 -- a curated cross-sector subset keeps the PCA covariance
matrix well-conditioned (n_obs >> n_assets) and keeps runtime fast.
"""

UNIVERSE = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "ORCL": "Technology",
    "IBM": "Technology", "CSCO": "Technology", "INTC": "Technology",
    "TXN": "Technology", "QCOM": "Technology", "ADBE": "Technology",
    "CRM": "Technology",
    # Communication Services
    "GOOGL": "Communication Services", "META": "Communication Services",
    "DIS": "Communication Services", "CMCSA": "Communication Services",
    "VZ": "Communication Services", "T": "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "TJX": "Consumer Discretionary", "BKNG": "Consumer Discretionary",
    # Consumer Staples
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "WMT": "Consumer Staples", "COST": "Consumer Staples", "CL": "Consumer Staples",
    "KMB": "Consumer Staples", "MO": "Consumer Staples",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "C": "Financials",
    "AXP": "Financials", "BLK": "Financials", "SPGI": "Financials",
    "USB": "Financials",
    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "MRK": "Healthcare", "ABT": "Healthcare", "TMO": "Healthcare",
    "BMY": "Healthcare", "AMGN": "Healthcare", "MDT": "Healthcare",
    "CVS": "Healthcare",
    # Industrials
    "HON": "Industrials", "UPS": "Industrials", "CAT": "Industrials",
    "BA": "Industrials", "GE": "Industrials", "MMM": "Industrials",
    "LMT": "Industrials", "RTX": "Industrials", "DE": "Industrials",
    "UNP": "Industrials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy",
    # Materials
    "LIN": "Materials", "APD": "Materials", "ECL": "Materials",
    "NEM": "Materials", "DD": "Materials",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "D": "Utilities", "AEP": "Utilities",
    # Real Estate
    "PLD": "Real Estate", "AMT": "Real Estate", "SPG": "Real Estate",
    "EQIX": "Real Estate",
}

TICKERS = sorted(UNIVERSE.keys())
