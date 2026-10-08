"""
Supertrend Wick-Touch Scanner (Binance) — single-file Streamlit app.

Signal (Supertrend 10,3 by default):
  * UPSIDE   : trend is bullish and the candle's LOW wick touches the GREEN line from above
  * DOWNSIDE : trend is bearish and the candle's HIGH wick touches the RED line from below
  (a candle that closes through the line is a trend flip, not a wick touch -> ignored)

Priority tiers (per coin + timeframe, counted inside the current Supertrend run):
  1. FIRST touch      : first wick touch since the trend started        -> top priority
  2. RE-TOUCH         : touched before, but >= N candles formed since   -> still fresh
  3. REPEAT           : touched again within N candles of the last one  -> hidden by default
N ("gap candles") is set separately for every timeframe in the sidebar (default 5).

Alerts fire as soon as the touch is seen on the live (forming) candle.
Needs: streamlit>=1.37, pandas, numpy, requests   (no ccxt, no pandas_ta)
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Supertrend Wick-Touch Scanner", page_icon="🎯", layout="wide")

IST = ZoneInfo("Asia/Kolkata")

# ----------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------
# data-api.binance.vision is Binance's public market-data mirror; it is not geo-blocked
# for US-hosted servers (e.g. Streamlit Community Cloud) the way api.binance.com is.
SPOT_HOSTS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
]
FUT_HOSTS = ["https://fapi.binance.com"]

TF_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "1d": 1440}
STABLES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDE", "AEUR", "EUR", "USD1", "XUSD", "RLUSD"}

TIER_FIRST, TIER_RETOUCH, TIER_REPEAT = 0, 1, 2
TIER_LABEL = {
    TIER_FIRST: "🟢 FIRST touch",
    TIER_RETOUCH: "🟡 RE-TOUCH",
    TIER_REPEAT: "⚪ REPEAT",
}


# ----------------------------------------------------------------------------------
# Binance REST helpers
# ----------------------------------------------------------------------------------
class BinanceError(Exception):
    pass


class GeoBlocked(BinanceError):
    pass


class RateLimited(BinanceError):
    pass


_tls = threading.local()


def _session():
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "supertrend-wick-scanner/1.0"})
        _tls.s = s
    return s


def _get(market, path, params=None):
    hosts = SPOT_HOSTS if market == "Spot" else FUT_HOSTS
    prefix = "/api/v3" if market == "Spot" else "/fapi/v1"
    err = None
    for host in hosts:
        try:
            r = _session().get(host + prefix + path, params=params, timeout=10)
        except requests.RequestException as e:
            err = BinanceError(f"network: {type(e).__name__}")
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (418, 429):
            raise RateLimited(f"HTTP {r.status_code} rate limit")
        if r.status_code in (451, 403):
            err = GeoBlocked(f"HTTP {r.status_code} (server location blocked by Binance)")
            continue
        err = BinanceError(f"HTTP {r.status_code}: {r.text[:100]}")
    raise err or BinanceError("no host reachable")


@st.cache_data(ttl=120, show_spinner=False)
def load_tickers(market):
    """{symbol: (24h quote volume in USDT, 24h change %)} for every USDT pair."""
    data = _get(market, "/ticker/24hr")
    out = {}
    for t in data:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        base = sym[:-4]
        if not base or base.upper() in STABLES:
            continue
        try:
            out[sym] = (float(t["quoteVolume"]), float(t["priceChangePercent"]))
        except (KeyError, ValueError):
            continue
    return out


def build_universe(market, top_n, min_vol_m, custom_text):
    tickers = load_tickers(market)
    if custom_text.strip():
        syms = []
        for raw in custom_text.replace("\n", ",").replace(" ", ",").split(","):
            s = raw.strip().upper()
            if not s:
                continue
            if not s.endswith("USDT"):
                s += "USDT"
            if s in tickers and s not in syms:
                syms.append(s)
        return [(s, tickers[s][0]) for s in syms]
    rows = [(s, v[0]) for s, v in tickers.items() if v[0] >= min_vol_m * 1e6]
    rows.sort(key=lambda x: -x[1])
    return rows[:top_n]


def fetch_klines(market, symbol, tf, limit=300):
    raw = _get(market, "/klines", {"symbol": symbol, "interval": tf, "limit": limit})
    if not raw:
        return np.empty((0, 5)), False
    now_ms = time.time() * 1000
    forming = float(raw[-1][6]) > now_ms
    arr = np.array([[r[0], r[1], r[2], r[3], r[4]] for r in raw], dtype=float)  # t,o,h,l,c
    return arr, forming


# ----------------------------------------------------------------------------------
# Supertrend + wick-touch logic (same maths as TradingView's ta.supertrend:
# Wilder/RMA ATR, ratcheting bands)
# ----------------------------------------------------------------------------------
def supertrend(h, l, c, period, mult):
    n = len(c)
    st = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)  # +1 bullish (green line), -1 bearish (red line), 0 = warm-up
    if n <= period + 1:
        return st, trend

    prev_c = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    tr[0] = h[0] - l[0]

    atr = np.full(n, np.nan)
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    hl2 = (h + l) / 2.0
    s = period - 1
    fu = fl = 0.0
    direction = 1
    for i in range(s, n):
        bu = hl2[i] + mult * atr[i]
        bl = hl2[i] - mult * atr[i]
        if i == s:
            fu, fl = bu, bl
            direction = 1 if c[i] >= hl2[i] else -1
        else:
            nfu = bu if (bu < fu or c[i - 1] > fu) else fu
            nfl = bl if (bl > fl or c[i - 1] < fl) else fl
            fu, fl = nfu, nfl
            if direction == -1 and c[i] > fu:
                direction = 1
            elif direction == 1 and c[i] < fl:
                direction = -1
        trend[i] = direction
        st[i] = fl if direction == 1 else fu
    return st, trend
# ---- end of part 1 ----
