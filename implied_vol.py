"""
Options Pricing & Implied Volatility Tool
=========================================

Prices European options with Black-Scholes, then inverts the formula against
live market prices to extract implied volatility - the market's forecast of
future movement, backed out of what people are actually paying.

Plotting IV across strikes produces the volatility smile: the systematic gap
between what Black-Scholes assumes (one constant volatility) and what the
market actually charges (more for crash protection).

Usage:  python implied_vol.py [TICKER]
Output: charts/smile.png, charts/surface.png, charts/validation.png
"""

import os
import sys
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm

warnings.filterwarnings("ignore")

TICKER = "SPY"
RISK_FREE = 0.043          # ~3-month T-bill; see note in README
N_EXPIRIES = 6
MIN_OPEN_INTEREST = 10     # liquidity filter - illiquid quotes are noise
MIN_PRICE = 0.10           # below this, the 1c tick dominates the price
MAX_REL_SPREAD = 0.80      # reject quotes wider than 80% of their mid

CHART_DIR = "charts"
INK, ACCENT, ACCENT_2 = "#1a1a1a", "#0b5d8a", "#b5451c"
GRID = "#d9d9d9"


# ----------------------------------------------------------------------------
# 1. Black-Scholes
# ----------------------------------------------------------------------------

def d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def black_scholes(S, K, T, r, sigma, kind="call"):
    """
    Price a European option.

    S     spot price of the underlying
    K     strike price
    T     time to expiry, in years
    r     risk-free rate (annualized, decimal)
    sigma volatility (annualized, decimal)
    """
    d1, d2 = d1_d2(S, K, T, r, sigma)
    if kind == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def vega(S, K, T, r, sigma):
    """Sensitivity of option price to volatility. Collapses toward zero far OTM."""
    d1, _ = d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def delta(S, K, T, r, sigma, kind="call"):
    """Sensitivity to the underlying. Used here to measure moneyness."""
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1


# ----------------------------------------------------------------------------
# 2. Inverting to implied volatility
# ----------------------------------------------------------------------------

def implied_vol(price, S, K, T, r, kind="call"):
    """
    Solve for the sigma that makes Black-Scholes reproduce the observed price.

    Uses Brent's method rather than Newton-Raphson. Newton divides by vega on
    every step, and vega collapses to ~0 for deep out-of-the-money options -
    precisely the strikes that make the smile interesting. Brent brackets the
    root between bounds and cannot diverge.
    """
    # Arbitrage bounds: a call is worth at least its discounted intrinsic value
    # and never more than the spot. Prices outside this have no valid IV.
    intrinsic = max(S - K * np.exp(-r * T), 0) if kind == "call" \
        else max(K * np.exp(-r * T) - S, 0)
    if price < intrinsic or price <= 0:
        return np.nan

    objective = lambda sig: black_scholes(S, K, T, r, sig, kind) - price
    try:
        return brentq(objective, 1e-6, 5.0, xtol=1e-8, maxiter=200)
    except ValueError:
        return np.nan


# ----------------------------------------------------------------------------
# 3. Market data
# ----------------------------------------------------------------------------

def fetch_chains(ticker=TICKER, n_expiries=N_EXPIRIES):
    """Pull live option chains and compute implied vol for every liquid strike."""
    tk = yf.Ticker(ticker)
    spot = float(tk.fast_info["lastPrice"])
    today = pd.Timestamp.now().normalize()

    # Skip the nearest expiries: with days to go, T is tiny and IV is unstable
    expiries = [e for e in tk.options
                if (pd.Timestamp(e) - today).days >= 7][:n_expiries]

    rows = []
    for expiry in expiries:
        T = (pd.Timestamp(expiry) - today).days / 365.0
        chain = tk.option_chain(expiry)

        for kind, df in [("call", chain.calls), ("put", chain.puts)]:
            df = df.copy()
            # Quality filters. Each one removes a specific kind of garbage:
            #  - bid/ask > 0        : contract has a live two-sided market
            #  - open interest      : someone actually holds these; not a dead strike
            #  - mid >= $0.10       : below this the 1c tick is a large share of the
            #                         price, so IV is dominated by rounding, not views
            #  - relative spread    : a quote 80%+ wide has no meaningful mid
            df = df[(df.bid > 0) & (df.ask > 0) &
                    (df.openInterest >= MIN_OPEN_INTEREST)]
            if df.empty:
                continue

            # Mid price, not lastPrice - the last trade may be hours stale
            mid = (df.bid + df.ask) / 2
            rel_spread = (df.ask - df.bid) / mid
            keep = (mid >= MIN_PRICE) & (rel_spread <= MAX_REL_SPREAD)
            df, mid = df[keep], mid[keep]
            if df.empty:
                continue

            for K, price, yahoo_iv in zip(df.strike, mid, df.impliedVolatility):
                iv = implied_vol(price, spot, K, T, RISK_FREE, kind)
                if np.isnan(iv) or iv < 0.01 or iv > 3.0:
                    continue
                rows.append({
                    "expiry": expiry, "T": T, "kind": kind, "strike": K,
                    "mid": price, "iv": iv, "yahoo_iv": yahoo_iv,
                    "moneyness": K / spot,
                    "delta": delta(spot, K, T, RISK_FREE, iv, kind),
                })

    return spot, pd.DataFrame(rows)


def otm_only(df, spot):
    """
    Keep out-of-the-money options only.

    Standard practice: OTM contracts are more liquid and carry more of their
    value as time value, so their prices are more informative about volatility.
    Puts below spot, calls above - which also avoids double-counting each strike.
    """
    return df[((df.kind == "put") & (df.strike <= spot)) |
              ((df.kind == "call") & (df.strike > spot))]


# ----------------------------------------------------------------------------
# 4. Charts
# ----------------------------------------------------------------------------

def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)


def plot_smile(df, spot, ticker, path):
    """The headline chart: IV against strike for the nearest liquid expiry."""
    expiry = sorted(df.expiry.unique())[0]
    sub = otm_only(df[df.expiry == expiry], spot).sort_values("strike")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for kind, color, label in [("put", ACCENT_2, "OTM puts (downside)"),
                               ("call", ACCENT, "OTM calls (upside)")]:
        s = sub[sub.kind == kind]
        ax.plot(s.strike, 100 * s.iv, "o-", color=color, ms=4,
                lw=1.4, label=label, alpha=0.9)

    ax.axvline(spot, color=INK, ls="--", lw=1, alpha=0.7)
    ax.annotate(f"spot {spot:.0f}", xy=(spot, ax.get_ylim()[1]),
                xytext=(4, -12), textcoords="offset points",
                fontsize=8.5, color=INK)

    atm = sub.iloc[(sub.strike - spot).abs().argsort()[:1]]
    if not atm.empty:
        ax.axhline(100 * atm.iv.iloc[0], color=INK, ls=":", lw=0.9, alpha=0.5)

    days = int(round(sub["T"].iloc[0] * 365))
    ax.set_xlabel("Strike ($)", fontsize=9.5, color=INK)
    ax.set_ylabel("Implied volatility (%)", fontsize=9.5, color=INK)
    ax.set_title(f"{ticker} Volatility Smile - {expiry} ({days} days)",
                 fontsize=13, color=INK, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)

    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_surface(df, spot, ticker, path):
    """Smile across multiple expiries - the term structure of the skew."""
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    expiries = sorted(df.expiry.unique())
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(expiries)))

    for color, expiry in zip(cmap, expiries):
        sub = otm_only(df[df.expiry == expiry], spot).sort_values("moneyness")
        if len(sub) < 5:
            continue
        days = int(round(sub["T"].iloc[0] * 365))
        ax.plot(sub.moneyness, 100 * sub.iv, "-", color=color,
                lw=1.5, label=f"{days}d", alpha=0.9)

    ax.axvline(1.0, color=INK, ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Moneyness (strike / spot)", fontsize=9.5, color=INK)
    ax.set_ylabel("Implied volatility (%)", fontsize=9.5, color=INK)
    ax.set_title(f"{ticker} Volatility Surface - Skew by Expiry",
                 fontsize=13, color=INK, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, title="Days to expiry",
              title_fontsize=8.5)
    _style(ax)

    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_validation(df, path):
    """Sanity check: own solver against the data vendor's published IV."""
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.scatter(100 * df.yahoo_iv, 100 * df.iv, s=11, color=ACCENT,
               alpha=0.45, edgecolors="none")

    lo = 0
    hi = max(100 * df.iv.max(), 100 * df.yahoo_iv.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color=ACCENT_2, ls="--", lw=1.1, label="45° line")

    corr = df[["iv", "yahoo_iv"]].corr().iloc[0, 1]
    ax.set_xlabel("Vendor implied volatility (%)", fontsize=9.5, color=INK)
    ax.set_ylabel("This model's implied volatility (%)", fontsize=9.5, color=INK)
    ax.set_title(f"Solver Validation  (r = {corr:.4f}, n = {len(df):,})",
                 fontsize=12, color=INK, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)

    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. Skew metrics
# ----------------------------------------------------------------------------

def skew_table(df, spot):
    """
    Quantify the smile per expiry.

    25-delta skew is the desk-standard measure: IV of the 25-delta put minus
    IV of the 25-delta call. Positive means downside protection costs more
    than equivalent upside - the persistent signature of equity index options.
    """
    rows = []
    for expiry in sorted(df.expiry.unique()):
        sub = df[df.expiry == expiry]
        puts = sub[sub.kind == "put"]
        calls = sub[sub.kind == "call"]
        if puts.empty or calls.empty:
            continue

        p25 = puts.iloc[(puts.delta + 0.25).abs().argsort()[:1]]
        c25 = calls.iloc[(calls.delta - 0.25).abs().argsort()[:1]]
        atm = sub.iloc[(sub.strike - spot).abs().argsort()[:1]]

        rows.append({
            "expiry": expiry,
            "days": int(round(sub["T"].iloc[0] * 365)),
            "atm_iv": 100 * atm.iv.iloc[0],
            "put_25d_iv": 100 * p25.iv.iloc[0],
            "call_25d_iv": 100 * c25.iv.iloc[0],
            "skew_25d": 100 * (p25.iv.iloc[0] - c25.iv.iloc[0]),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 6. Run
# ----------------------------------------------------------------------------

def main():
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else TICKER
    os.makedirs(CHART_DIR, exist_ok=True)

    print(f"Fetching {ticker} option chains...")
    spot, df = fetch_chains(ticker)
    print(f"  Spot: ${spot:,.2f}")
    print(f"  {len(df):,} contracts priced across {df.expiry.nunique()} expiries\n")

    corr = df[["iv", "yahoo_iv"]].corr().iloc[0, 1]
    print(f"Validation vs. vendor IV: r = {corr:.4f}\n")

    skew = skew_table(df, spot)
    print("Skew by expiry (IV in %):")
    print(skew.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
    print()

    plot_smile(df, spot, ticker, f"{CHART_DIR}/smile.png")
    plot_surface(df, spot, ticker, f"{CHART_DIR}/surface.png")
    plot_validation(df, f"{CHART_DIR}/validation.png")

    df.to_csv("implied_vols.csv", index=False)
    print(f"Wrote {CHART_DIR}/ and implied_vols.csv")
    return spot, df, skew


if __name__ == "__main__":
    main()
