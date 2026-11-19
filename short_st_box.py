"""
Kishore - ST SHORT-ONLY BOX
Supertrend(10,3) wick-touch SHORT trades: live PnL + backtest RR 1..10

Use inside your existing Supertrend Wick-Touch Scanner (app.py):
    from short_st_box import render_short_box
    render_short_box()          # put this BEFORE your auto-refresh / sleep / st.rerun part

Or run it alone:  streamlit run short_st_box.py
requirements: streamlit>=1.37, pandas, numpy, requests
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
HOSTS = [
    "https://data-api.binance.vision",   # Spot Vision = primary
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api-gcp.binance.com",
]
ALL_TFS = ["5m", "15m", "30m", "1h", "4h"]
STABLES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDE", "EUR", "AEUR",
           "XUSD", "USD1", "UST", "PAXG", "WBTC", "BFUSD", "RLUSD", "EURI"}
ATR_LEN, MULT = 10, 3.0
RR_LIST = list(range(1, 11))
INF = 10 ** 9
REFRESH_SEC = 60
IST = "Asia/Kolkata"

# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
_CACHE = {}
_LOCK = threading.Lock()


def _get(path, params, timeout=10):
    last = None
    for host in HOSTS:
        try:
            r = requests.get(host + path, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = f"{host} -> {r.status_code}"
        except Exception as e:  # noqa
            last = f"{host} -> {e}"
    raise RuntimeError(last)


def _fetch(symbol, tf, n):
    rows, end = [], None
    while len(rows) < n:
        lim = min(1000, n - len(rows))
        params = {"symbol": symbol, "interval": tf, "limit": lim}
        if end:
            params["endTime"] = end
        data = _get("/api/v3/klines", params)
        if not data:
            break
        rows = data + rows
        end = data[0][0] - 1
        if len(data) < lim:
            break
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["t", "o", "h", "l", "c", "v"]
    df = df.astype({"t": "int64", "o": float, "h": float, "l": float, "c": float, "v": float})
    return df.drop_duplicates("t").reset_index(drop=True)


def klines(symbol, tf, n, ttl):
    key = (symbol, tf, n)
    now = time.time()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    df = _fetch(symbol, tf, n)
    with _LOCK:
        _CACHE[key] = (now, df)
        if len(_CACHE) > 500:
            for k, _ in sorted(_CACHE.items(), key=lambda kv: kv[1][0])[:100]:
                _CACHE.pop(k, None)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def top_symbols(n):
    data = _get("/api/v3/ticker/24hr", {})
    df = pd.DataFrame(data)
    df = df[df["symbol"].str.endswith("USDT")].copy()
    df["base"] = df["symbol"].str[:-4]
    df = df[~df["base"].isin(STABLES)]
    df["q"] = df["quoteVolume"].astype(float)
    return df.sort_values("q", ascending=False)["symbol"].head(n).tolist()


# ----------------------------------------------------------------------------
# SUPERTREND (10,3)  - same maths as TradingView ta.supertrend
# ----------------------------------------------------------------------------
def supertrend(h, l, c, length=ATR_LEN, mult=MULT):
    n = len(c)
    st = np.full(n, np.nan)
    trend = np.ones(n, dtype=int)
    if n <= length + 1:
        return st, trend
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    pc = c[:-1]
    tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - pc), np.abs(l[1:] - pc)])
    atr = np.full(n, np.nan)
    atr[length - 1] = tr[:length].mean()
    for i in range(length, n):
        atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length
    hl2 = (h + l) / 2
    up, dn = hl2 - mult * atr, hl2 + mult * atr
    fu, fd = up.copy(), dn.copy()
    s = length - 1
    for i in range(s + 1, n):
        fu[i] = max(up[i], fu[i - 1]) if c[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if c[i - 1] < fd[i - 1] else dn[i]
        if trend[i - 1] == -1 and c[i] > fd[i - 1]:
            trend[i] = 1
        elif trend[i - 1] == 1 and c[i] < fu[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    st = np.where(trend == 1, fu, fd)
    st[: s + 1] = np.nan
    return st, trend


# ----------------------------------------------------------------------------
# SHORT RULES
# ----------------------------------------------------------------------------
def _prep(h, l, start, entry, sl):
    """First SL-hit candle and first TP-hit candle (for every RR) after entry."""
    n = len(h)
    risk = sl - entry
    if start >= n:
        return INF, [INF] * len(RR_LIST)
    idx = np.flatnonzero(h[start:] >= sl)
    sl_i = start + int(idx[0]) if idx.size else INF
    neg_cummin = -np.minimum.accumulate(l[start:])
    tp_i = []
    for rr in RR_LIST:
        tp = entry - rr * risk
        k = int(np.searchsorted(neg_cummin, -tp, side="left"))
        tp_i.append(start + k if k < neg_cummin.size else INF)
    return sl_i, tp_i


def _trade(kind, entry_i, entry, top, p, h, l):
    sl = top * (1 + p["buf"] / 100)
    risk = sl - entry
    if risk <= 0 or risk / entry * 100 < p["min_risk"]:
        return None
    sl_i, tp_i = _prep(h, l, entry_i + 1, entry, sl)
    return {"kind": kind, "entry_i": entry_i, "entry": entry, "sl": sl,
            "risk": risk, "sl_i": sl_i, "tp_i": tp_i, "re": None}


def _reentry(main, o, h, l, c, st, tr, p, n):
    """After SL hit: if within N candles price closes back below the SL price
    and below the ST line (bearish), short again. SL = highest high since the SL hit."""
    k = main["sl_i"]
    if k >= INF:
        return None
    for m in range(k + 1, min(k + p["re_win"], n - 1) + 1):
        if tr[m] == -1 and c[m] < st[m] and c[m] < main["sl"]:
            top = h[k: m + 1].max()
            t = _trade("RE-ENTRY", m, c[m], top, p, h, l)
            if t is not None:
                return t
    return None


def find_short_trades(df, p):
    """df must contain CLOSED candles only."""
    o, h, l, c = (df[x].to_numpy(float) for x in ("o", "h", "l", "c"))
    n = len(df)
    st, tr = supertrend(h, l, c)
    if n < 30 or np.isnan(st).all():
        return [], st, tr
    tol = p["tol"] / 100
    valid = ~np.isnan(st)
    # wick reached the red line, candle closed under it
    touch = valid & (tr == -1) & (h >= st * (1 - tol)) & (c < st)
    body_below = valid & (np.maximum(o, c) < st)
    red = c < o

    cands, used = [], set()
    # RULE 1: touch candle (body fully under line) + next candle closes under line and is RED
    for i in np.flatnonzero(touch & body_below):
        j = i + 1
        if j >= n:
            continue
        if tr[j] == -1 and c[j] < st[j] and red[j]:
            cands.append(("ST WICK", j, c[j], max(h[i], h[j])))
            used.add(j)

    # RULE 3: double top rejected at the SAME ST line (price must not go above)
    if p["dt_on"]:
        tl = list(np.flatnonzero(touch))
        dtol = p["dt_tol"] / 100
        for a, i in enumerate(tl):
            pp = None
            for b in range(a - 1, -1, -1):
                q = tl[b]
                if i - q > p["dt_lb"]:
                    break
                if i - q >= 2:
                    pp = q
                    break
            j = i + 1
            if pp is None or j >= n or j in used:
                continue
            top = max(h[pp], h[i])
            if abs(h[i] - h[pp]) / h[pp] > dtol:
                continue
            if not (np.all(tr[pp: i + 1] == -1) and np.all(c[pp + 1: i] < st[pp + 1: i])):
                continue
            if h[pp + 1: i].size and h[pp + 1: i].max() > top * (1 + dtol):
                continue
            if tr[j] == -1 and c[j] < st[j] and red[j] and h[j] <= top * (1 + dtol):
                cands.append(("DOUBLE TOP", j, c[j], max(top, h[j])))
                used.add(j)

    cands.sort(key=lambda x: x[1])
    trades = []
    for kind, j, entry, top in cands:
        t = _trade(kind, j, entry, top, p, h, l)
        if t is None:
            continue
        if p["re_on"]:
            t["re"] = _reentry(t, o, h, l, c, st, tr, p, n)
        trades.append(t)
    return trades, st, tr


def _outcome(t, k):
    s, tp = t["sl_i"], t["tp_i"][k]
    if s >= INF and tp >= INF:
        return "OPEN", INF
    if s <= tp:                      # SL and TP in same candle -> SL (conservative)
        return "LOSS", s
    return "WIN", tp


def _rec(t, res, ex, rr):
    entry, risk = t["entry"], t["risk"]
    exit_p = entry - rr * risk if res == "WIN" else (t["sl"] if res == "LOSS" else np.nan)
    return {"kind": t["kind"], "entry_i": t["entry_i"], "entry": entry, "sl": t["sl"],
            "tp": entry - rr * risk, "risk": risk, "exit_i": ex if ex < INF else -1,
            "result": res, "exit": exit_p, "rr": rr}


def simulate(trades, rr):
    """One position at a time. Re-entry only after the first trade was stopped out."""
    k, out, busy = rr - 1, [], -1
    for t in trades:
        if t["entry_i"] <= busy:
            continue
        res, ex = _outcome(t, k)
        out.append(_rec(t, res, ex, rr))
        if res == "OPEN":
            break
        busy = ex
        if res == "LOSS" and t["re"] is not None:
            r = t["re"]
            res2, ex2 = _outcome(r, k)
            out.append(_rec(r, res2, ex2, rr))
            if res2 == "OPEN":
                break
            busy = ex2
    return out


def net_r(rec, price, fee):
    """Short PnL in R (fees = % per side)."""
    return (rec["entry"] - price) / rec["risk"] - fee / 100 * (rec["entry"] + price) / rec["risk"]


# ----------------------------------------------------------------------------
# BACKTEST
# ----------------------------------------------------------------------------
def backtest_one(symbol, tf, candles, p, fee):
    full = klines(symbol, tf, candles, ttl=900)
    df = full.iloc[:-1].reset_index(drop=True)       # closed candles only
    trades, _, _ = find_short_trades(df, p)
    if not trades:
        return []
    times = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(IST)
    rows = []
    for rr in RR_LIST:
        for r in simulate(trades, rr):
            if r["result"] == "OPEN":
                continue
            rows.append({
                "Coin": symbol[:-4], "TF": tf, "Setup": r["kind"], "RR": rr,
                "Entry time": times.iloc[r["entry_i"]], "Entry": r["entry"], "SL": r["sl"], "TP": r["tp"],
                "Exit time": times.iloc[r["exit_i"]], "Exit": r["exit"], "Result": r["result"],
                "R": net_r(r, r["exit"], fee),
            })
    return rows


def summarize(tr, capital, risk_pct):
    rows = []
    for rr, g in tr.groupby("RR"):
        w, l = (g["Result"] == "WIN").sum(), (g["Result"] == "LOSS").sum()
        gp, gl = g.loc[g["R"] > 0, "R"].sum(), -g.loc[g["R"] < 0, "R"].sum()
        rows.append({
            "RR": f"1:{rr}", "Trades": len(g), "Wins": int(w), "Losses": int(l),
            "Win %": round(w / max(w + l, 1) * 100, 1), "Net R": round(g["R"].sum(), 2),
            "Profit factor": round(gp / gl, 2) if gl > 0 else np.inf,
            "PnL": round(g["R"].sum() * capital * risk_pct / 100, 0),
        })
    return pd.DataFrame(rows)


def group_table(g, by, capital, risk_pct):
    rows = []
    for key, x in g.groupby(by):
        w, l = (x["Result"] == "WIN").sum(), (x["Result"] == "LOSS").sum()
        rows.append({by: key, "Trades": len(x), "Wins": int(w), "Losses": int(l),
                     "Win %": round(w / max(w + l, 1) * 100, 1), "Net R": round(x["R"].sum(), 2),
                     "PnL": round(x["R"].sum() * capital * risk_pct / 100, 0)})
    return pd.DataFrame(rows).sort_values("Net R", ascending=False)


# ----------------------------------------------------------------------------
# UI HELPERS
# ----------------------------------------------------------------------------
def _fp(x):
    return f"{x:.6g}" if pd.notna(x) else "-"


def _show(d, styler=None):
    obj = styler if styler is not None else d
    try:
        st.dataframe(obj, width="stretch", hide_index=True)
    except Exception:  # older streamlit
        st.dataframe(obj, use_container_width=True, hide_index=True)


def _color(v):
    if isinstance(v, (int, float)) and pd.notna(v):
        return "color:#26a69a" if v > 0 else ("color:#ef5350" if v < 0 else "")
    return ""


def _style(d, cols):
    s = d.style
    f = s.map if hasattr(s, "map") else s.applymap
    return f(_color, subset=cols)


# ----------------------------------------------------------------------------
# LIVE
# ----------------------------------------------------------------------------
def _live_work(job):
    symbol, tf, p, rr, max_age, fee = job
    try:
        full = klines(symbol, tf, 500, ttl=20)
        df = full.iloc[:-1].reset_index(drop=True)
        n = len(df)
        trades, _, _ = find_short_trades(df, p)
        recs = simulate(trades, rr)
        times = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(IST)
        hf, lf, pf = full["h"].iloc[-1], full["l"].iloc[-1], full["c"].iloc[-1]
        out = []
        for r in recs:
            if r["entry_i"] < n - max_age:
                continue
            if r["result"] == "OPEN":
                if hf >= r["sl"]:
                    status, px = "SL HIT", r["sl"]
                elif lf <= r["tp"]:
                    status, px = "TP HIT", r["tp"]
                else:
                    status, px = "OPEN", pf
            else:
                status, px = ("TP HIT" if r["result"] == "WIN" else "SL HIT"), r["exit"]
            out.append({"symbol": symbol, "tf": tf, "r": r, "status": status, "px": px,
                        "price_now": pf, "age": n - 1 - r["entry_i"],
                        "time": times.iloc[r["entry_i"]]})
        return out, None
    except Exception as e:  # noqa
        return [], f"{symbol} {tf}: {e}"


def _live_body(n_coins, tfs, rr, max_age, show_closed, p, capital, risk_pct, fee):
    try:
        symbols = top_symbols(n_coins)
    except Exception as e:  # noqa
        st.error(f"Binance symbols fetch failed: {e}")
        return
    jobs = [(s, tf, p, rr, max_age, fee) for s in symbols for tf in tfs]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_live_work, jobs))
    items = [x for res, _ in results for x in res]
    errors = [e for _, e in results if e]

    rows = []
    for it in items:
        r = it["r"]
        if it["status"] != "OPEN" and not show_closed:
            continue
        rv = net_r(r, it["px"], fee)
        rows.append({
            "": "NEW" if it["age"] == 0 and it["status"] == "OPEN" else "",
            "Coin": it["symbol"][:-4], "TF": it["tf"], "Setup": r["kind"],
            "Entry time": it["time"].strftime("%d %b %H:%M"), "Age": it["age"],
            "Entry": _fp(r["entry"]), "SL": _fp(r["sl"]), f"TP (1:{rr})": _fp(r["tp"]),
            "Price": _fp(it["price_now"]), "PnL %": (r["entry"] - it["px"]) / r["entry"] * 100,
            "PnL R": rv, "PnL": rv * capital * risk_pct / 100, "Status": it["status"],
            "_i": -it["age"],
        })
    d = pd.DataFrame(rows)
    open_n = int((d["Status"] == "OPEN").sum()) if len(d) else 0
    new_n = int((d[""] == "NEW").sum()) if len(d) else 0
    live_pnl = float(d.loc[d["Status"] == "OPEN", "PnL"].sum()) if len(d) else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coins x TF", len(jobs))
    c2.metric("Open shorts", open_n)
    c3.metric("NEW", new_n)
    c4.metric("Live PnL", f"{live_pnl:+,.0f}")
    st.caption(f"Last scan (IST): {pd.Timestamp.now(tz=IST).strftime('%H:%M:%S')} "
               f"- auto refresh every {REFRESH_SEC}s - only closed candles make signals")

    if d.empty:
        st.info("No short trade qualified right now. Rules are strict - waiting for ST wick touch + "
                "red confirmation candle.")
    else:
        d = d.sort_values("_i", ascending=False).drop(columns="_i")
        _show(d, _style(d, ["PnL %", "PnL R", "PnL"]).format(
            {"PnL %": "{:+.2f}", "PnL R": "{:+.2f}", "PnL": "{:+,.0f}"}))
    if errors:
        with st.expander(f"{len(errors)} fetch errors"):
            st.write("\n".join(errors[:20]))


_live = st.fragment(run_every=REFRESH_SEC)(_live_body) if hasattr(st, "fragment") else _live_body

RULES_MD = """
**Rule 1 - ST wick short.** Supertrend(10,3) is RED (line above price). A candle's **body closes below the line**
and only its **wick touches/crosses** the line. Next candle must also **close below the line and be RED**.
Entry = close of that red candle. **SL = highest wick** (of the touch candle / confirm candle) - price crossing
that wick means SL hit.

**Rule 2 - Re-entry after SL.** If SL gets hit, watch the next **10 candles** (same timeframe). If a candle
**closes back below the SL price and below the ST line** -> short again. New SL = highest high since the SL hit.
One re-entry per signal.

**Rule 3 - Double top.** Two wick touches of the **same ST line** (highs within tolerance), no close above the
line in between, and price must **reject** (not break up). Then the red confirmation candle closing below the line
-> short.

Backtest: RR 1:1 to 1:10, one position at a time per coin/TF. SL and TP in the same candle = SL (conservative).
"""


# ----------------------------------------------------------------------------
# MAIN BOX
# ----------------------------------------------------------------------------
def render_short_box():
    with st.container(border=True):
        st.markdown("### \U0001F53B ST Short-Only Box")
        st.caption("Binance Spot - Supertrend(10,3) - SHORT trades only")
        with st.expander("\U0001F4DC Rules"):
            st.markdown(RULES_MD)
        with st.expander("\u2699\uFE0F Settings"):
            a, b = st.columns(2)
            n_coins = a.slider("Coins (top by volume)", 10, 100, 56, 2, key="ssb_n")
            tfs = b.multiselect("Timeframes", ALL_TFS, default=["15m", "1h"], key="ssb_tf")
            a, b = st.columns(2)
            live_rr = a.selectbox("Live trade RR (TP)", RR_LIST, index=2, format_func=lambda x: f"1:{x}", key="ssb_rr")
            max_age = b.number_input("Show trades up to N candles old", 1, 200, 20, key="ssb_age")
            show_closed = st.checkbox("Also show closed (SL/TP hit) live trades", False, key="ssb_closed")
            a, b, c = st.columns(3)
            capital = a.number_input("Capital", 1000.0, 1e9, 100000.0, 1000.0, key="ssb_cap")
            risk_pct = b.number_input("Risk per trade %", 0.1, 10.0, 1.0, 0.1, key="ssb_risk")
            fee = c.number_input("Fee % per side", 0.0, 1.0, 0.05, 0.01, key="ssb_fee")
            a, b, c = st.columns(3)
            re_on = a.checkbox("Re-entry rule", True, key="ssb_re")
            re_win = b.number_input("Re-entry window (candles)", 1, 50, 10, key="ssb_rew")
            dt_on = c.checkbox("Double-top rule", True, key="ssb_dt")
            a, b, c = st.columns(3)
            dt_lb = a.number_input("Double-top lookback", 5, 100, 30, key="ssb_dtlb")
            dt_tol = b.number_input("Double-top tolerance %", 0.05, 2.0, 0.30, 0.05, key="ssb_dttol")
            tol = c.number_input("Wick touch tolerance %", 0.0, 1.0, 0.0, 0.01, key="ssb_tol")
            a, b = st.columns(2)
            buf = a.number_input("SL buffer above wick %", 0.0, 1.0, 0.0, 0.01, key="ssb_buf")
            min_risk = b.number_input("Min risk % (skip tiny SL)", 0.0, 2.0, 0.05, 0.01, key="ssb_minr")
        p = {"tol": tol, "buf": buf, "min_risk": min_risk, "re_on": re_on, "re_win": int(re_win),
             "dt_on": dt_on, "dt_lb": int(dt_lb), "dt_tol": dt_tol}
        if not tfs:
            st.warning("Select at least one timeframe in Settings.")
            return

        tab_live, tab_bt = st.tabs(["\U0001F534 Live Short Trades", "\U0001F4CA Backtest RR 1-10"])

        with tab_live:
            _live(n_coins, tuple(tfs), live_rr, int(max_age), show_closed, p, capital, risk_pct, fee)

        with tab_bt:
            a, b = st.columns(2)
            candles = a.selectbox("Candles per coin/TF", [1000, 2000, 3000, 5000], index=1, key="ssb_c")
            go = b.button("\u25B6 Run backtest", key="ssb_go")
            if go:
                try:
                    symbols = top_symbols(n_coins)
                except Exception as e:  # noqa
                    st.error(f"Binance symbols fetch failed: {e}")
                    symbols = []
                jobs = [(s, tf) for s in symbols for tf in tfs]
                bar, rows, errs = st.progress(0.0, "Backtesting..."), [], []
                with ThreadPoolExecutor(max_workers=8) as ex:
                    futs = [ex.submit(backtest_one, s, tf, candles, p, fee) for s, tf in jobs]
                    for k, f in enumerate(futs):
                        try:
                            rows += f.result()
                        except Exception as e:  # noqa
                            errs.append(f"{jobs[k][0]} {jobs[k][1]}: {e}")
                        bar.progress((k + 1) / len(futs), f"{k + 1}/{len(futs)}")
                bar.empty()
                st.session_state["ssb_bt"] = pd.DataFrame(rows)
                st.session_state["ssb_bt_errs"] = errs
            tr = st.session_state.get("ssb_bt")
            if tr is None:
                st.info("Press Run backtest.")
            elif tr.empty:
                st.warning("No trades found with these settings.")
            else:
                summ = summarize(tr, capital, risk_pct)
                best = summ.sort_values("Net R", ascending=False).iloc[0]
                st.success(f"Best RR by Net R: **{best['RR']}** - {int(best['Trades'])} trades, "
                           f"win {best['Win %']}%, Net {best['Net R']}R, PnL {best['PnL']:+,.0f}")
                _show(summ, _style(summ, ["Net R", "PnL"]))
                sel = st.selectbox("Details for RR", RR_LIST, index=int(best["RR"].split(":")[1]) - 1,
                                   format_func=lambda x: f"1:{x}", key="ssb_sel")
                g = tr[tr["RR"] == sel].sort_values("Exit time")
                st.markdown("**Equity curve (cumulative R)**")
                st.line_chart(g["R"].cumsum().reset_index(drop=True))
                t1, t2, t3, t4 = st.tabs(["By setup", "By coin", "By TF", "All trades"])
                with t1:
                    _show(group_table(g, "Setup", capital, risk_pct))
                with t2:
                    _show(group_table(g, "Coin", capital, risk_pct))
                with t3:
                    _show(group_table(g, "TF", capital, risk_pct))
                with t4:
                    d = g.copy()
                    d["Entry time"] = d["Entry time"].dt.strftime("%d %b %y %H:%M")
                    d["Exit time"] = d["Exit time"].dt.strftime("%d %b %y %H:%M")
                    for col in ("Entry", "SL", "TP", "Exit"):
                        d[col] = d[col].map(_fp)
                    d["R"] = d["R"].round(2)
                    _show(d.drop(columns="RR").iloc[::-1], _style(d.drop(columns="RR").iloc[::-1], ["R"]))
            errs = st.session_state.get("ssb_bt_errs") or []
            if errs:
                with st.expander(f"{len(errs)} fetch errors"):
                    st.write("\n".join(errs[:20]))


if __name__ == "__main__":
    st.set_page_config(page_title="ST Short Box", page_icon="\U0001F53B", layout="wide")
    render_short_box()
