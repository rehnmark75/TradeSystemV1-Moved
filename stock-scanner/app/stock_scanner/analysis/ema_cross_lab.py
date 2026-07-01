"""
EMA 9/21/50 Cross — Optimization Lab
====================================
Heavy-precompute experimentation harness for pushing the ema_cross_9_21_50
scanner toward higher profit factor while keeping tradable frequency.

It computes, ONCE:
  - per-signal daily features (EMA stack, RSI, ADX, ATR%, rel-vol, distance,
    multi-horizon momentum, relative strength vs a synthetic market proxy)
  - a synthetic MARKET proxy + regime/breadth series (no index ETF in the data):
      market_ret  = equal-weight mean daily return across the liquid universe
      mkt_index   = cumprod(1+market_ret); mkt_above_ema50 = index > its EMA50
      breadth     = fraction of universe with close > own EMA50 (daily)
  - per-signal forward 1h path (high/low/close) truncated to the timeout, the
    next-session-open entry, and MFE/MAE over that window.

Then ARBITRARY entry filters and exit policies are evaluated cheaply from the
cached paths — no re-fetch. Entry fill + the conservative SL-first walk match
the validated daytrade_edge_sim convention.

OOS split: TRAIN < 2026-04-01, OOS >= 2026-04-01. A config that only works
in-sample is a FAIL.

Usage (inside stock-scanner container):
    docker exec stock-scanner python -m stock_scanner.analysis.ema_cross_lab --top-liquid 500 --dump-mfe
    docker exec stock-scanner python -m stock_scanner.analysis.ema_cross_lab --top-liquid 500 --experiments
The cache is built once per process; --experiments runs the registered configs.
"""

from __future__ import annotations

import sys
import argparse
import warnings
from typing import Optional, Callable

import numpy as np
import pandas as pd

from stock_scanner.analysis.daytrade_edge_sim import get_conn, fetch_df, bars_after
from stock_scanner.strategies.ema_cross_9_21_50 import EmaCross92150Strategy

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

TIMEOUT_TRADING_DAYS = 10
BARS_PER_DAY = 6
N_TIMEOUT = TIMEOUT_TRADING_DAYS * BARS_PER_DAY
NOTIONAL = 500.0
TRAIN_END = pd.Timestamp("2026-04-01")
TRADING_DAYS_IN_WINDOW = None  # set after load, for trades/day


# ---------------------------------------------------------------------------
# Feature computation (all from daily candles -> full-window, no leakage)
# ---------------------------------------------------------------------------
def add_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.reset_index(drop=True).copy()
    c = pd.to_numeric(g["close"], errors="coerce")
    h = pd.to_numeric(g["high"], errors="coerce")
    low = pd.to_numeric(g["low"], errors="coerce")
    v = pd.to_numeric(g["volume"], errors="coerce")

    g["ema_9"] = c.ewm(span=9, adjust=False).mean()
    g["ema_21"] = c.ewm(span=21, adjust=False).mean()
    g["ema_50"] = c.ewm(span=50, adjust=False).mean()
    g["ema_200"] = c.ewm(span=200, adjust=False).mean()
    g["_above9"] = c > g["ema_9"]
    g["_prev_above9"] = g["_above9"].shift(1)

    # RSI(14)
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    g["rsi"] = 100 - (100 / (1 + rs))

    # ATR(14) + ATR%
    pc = c.shift(1)
    tr = pd.concat([h - low, (h - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=27, min_periods=14, adjust=False).mean()
    g["atr"] = atr
    g["atr_pct"] = atr / c * 100

    # ADX(14)
    up = h.diff()
    dn = -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    atr_s = atr.replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    g["adx"] = dx.ewm(span=14, adjust=False).mean()

    # volume / rel-vol
    g["rel_vol"] = v / v.rolling(20).mean().replace(0, np.nan)

    # distance / momentum
    g["dist_ema9_pct"] = (c - g["ema_9"]) / g["ema_9"] * 100
    g["dist_ema21_pct"] = (c - g["ema_21"]) / g["ema_21"] * 100
    g["ret_5d"] = c.pct_change(5) * 100
    g["ret_20d"] = c.pct_change(20) * 100
    g["ret_60d"] = c.pct_change(60) * 100

    # cross-bar conviction: close position in the day's range + green
    rng = (h - low).replace(0, np.nan)
    g["range_pos"] = (c - low) / rng
    g["green"] = c > pd.to_numeric(g["open"], errors="coerce")

    # pullback depth: how far prior 1-3 bars dipped below EMA9, in ATR units
    below9 = (g["ema_9"] - low) / atr.replace(0, np.nan)   # +ve = dipped below 9
    g["pullback_atr"] = below9.shift(1).rolling(3).max()   # deepest dip in prior 3 bars

    # EMA stack alignment + slope
    g["stack_aligned"] = (g["ema_9"] >= g["ema_21"]) & (g["ema_21"] >= g["ema_50"])
    g["ema50_slope"] = g["ema_50"].pct_change(5) * 100
    g["above_200"] = c > g["ema_200"]

    # EMA50 fresh-cross age
    cross_up = (c > g["ema_50"]) & (c.shift(1) <= g["ema_50"].shift(1))
    idx = pd.Series(np.where(cross_up.values, np.arange(len(g)), np.nan), index=g.index)
    g["age50"] = pd.Series(np.arange(len(g)), index=g.index) - idx.ffill()

    g["daily_ret"] = c.pct_change()
    return g


def build_market_regime(daily: pd.DataFrame) -> pd.DataFrame:
    """Synthetic market proxy + breadth, indexed by date."""
    # equal-weight mean daily return per date
    mkt = daily.groupby("date")["daily_ret"].mean().rename("market_ret").to_frame()
    mkt["mkt_index"] = (1 + mkt["market_ret"].fillna(0)).cumprod()
    mkt["mkt_ema50"] = mkt["mkt_index"].ewm(span=50, adjust=False).mean()
    mkt["mkt_above_ema50"] = mkt["mkt_index"] > mkt["mkt_ema50"]
    mkt["mkt_ema20"] = mkt["mkt_index"].ewm(span=20, adjust=False).mean()
    mkt["mkt_above_ema20"] = mkt["mkt_index"] > mkt["mkt_ema20"]
    # breadth: fraction above own ema50
    breadth = daily.groupby("date").apply(
        lambda x: (pd.to_numeric(x["close"], errors="coerce") > x["ema_50"]).mean()
    ).rename("breadth")
    mkt = mkt.join(breadth)
    mkt["breadth_ema10"] = mkt["breadth"].ewm(span=10, adjust=False).mean()
    mkt["mkt_ret_20d"] = mkt["mkt_index"].pct_change(20) * 100
    return mkt


# ---------------------------------------------------------------------------
# Exit engine — generic walk over a forward path
# ---------------------------------------------------------------------------
def walk_exit(path: np.ndarray, entry: float, atr_pct: float, cfg: dict) -> dict:
    """path: (n,3) array of [high, low, close]. Returns outcome + pnl_pct.

    cfg keys (all optional except sl):
      sl            fixed stop %% below entry (e.g. 2.5)
      tp            fixed take-profit %% (None = no hard TP)
      sl_atr        stop = sl_atr * atr_pct  (overrides sl if set)
      tp_atr        tp   = tp_atr * atr_pct
      be_arm        arm breakeven once high >= entry*(1+be_arm/100); stop->entry
      trail_arm     start trailing once high >= entry*(1+trail_arm/100)
      trail_give    trailing stop = peak*(1 - trail_give/100)  (pct giveback)
      trail_atr     trailing stop = peak - trail_atr*atr_pct%  (atr giveback)
      tp1, tp1_frac partial: take tp1_frac at +tp1%, runner trails by trail_give/atr
      green_by_bar, green_by_pct  time-stop: exit at close of that bar if pnl<pct
    """
    n = min(len(path), N_TIMEOUT)
    if n == 0:
        return {"outcome": "no_data", "pnl_pct": 0.0, "bars": 0}

    a = float(atr_pct) if atr_pct and not np.isnan(atr_pct) else 2.0
    sl_pct = cfg.get("sl_atr", None)
    sl_pct = sl_pct * a if sl_pct is not None else cfg.get("sl", 2.5)
    tp_pct = cfg.get("tp_atr", None)
    tp_pct = tp_pct * a if tp_pct is not None else cfg.get("tp", None)

    sl = entry * (1 - sl_pct / 100)
    tp = entry * (1 + tp_pct / 100) if tp_pct is not None else None
    active_sl = sl
    peak = entry
    realized = 0.0           # locked partial pnl%
    remaining = 1.0          # fraction still open
    tp1 = cfg.get("tp1")
    tp1_frac = cfg.get("tp1_frac", 0.0)
    tp1_done = False

    be_arm = cfg.get("be_arm")
    trail_arm = cfg.get("trail_arm")
    trail_give = cfg.get("trail_give")
    trail_atr = cfg.get("trail_atr")
    gbb = cfg.get("green_by_bar")
    gbp = cfg.get("green_by_pct", 0.0)

    def close_all(price, i, tag):
        pnl = realized + remaining * ((price - entry) / entry * 100)
        return {"outcome": tag, "pnl_pct": pnl, "bars": i + 1}

    for i in range(n):
        high, low, close = float(path[i, 0]), float(path[i, 1]), float(path[i, 2])
        peak = max(peak, high)

        # ---- stop adjustments BEFORE testing (conservative: SL checked first) ----
        if be_arm is not None and high >= entry * (1 + be_arm / 100):
            active_sl = max(active_sl, entry)
        if trail_arm is not None and peak >= entry * (1 + trail_arm / 100):
            if trail_give is not None:
                active_sl = max(active_sl, peak * (1 - trail_give / 100))
            elif trail_atr is not None:
                active_sl = max(active_sl, peak * (1 - trail_atr * a / 100))

        # ---- SL first (conservative) ----
        if low <= active_sl:
            return close_all(active_sl, i, "stop")
        # ---- partial TP ----
        if tp1 is not None and not tp1_done and high >= entry * (1 + tp1 / 100):
            realized += tp1_frac * (tp1)
            remaining -= tp1_frac
            tp1_done = True
            # after partial, arm trailing on the runner if configured
            if trail_arm is None and (trail_give is not None or trail_atr is not None):
                trail_arm = tp1  # start trailing from here
        # ---- hard TP ----
        if tp is not None and high >= tp and remaining > 0:
            return close_all(tp, i, "win")
        # ---- time stop ----
        if gbb is not None and i + 1 >= gbb:
            cur = (close - entry) / entry * 100
            if cur < gbp:
                return close_all(close, i, "timestop")

    # timeout exit at last close
    last_close = float(path[n - 1, 2])
    return close_all(last_close, n - 1, "timeout")


# ---------------------------------------------------------------------------
# Build the cache (signals + features + market regime + forward paths + MFE/MAE)
# ---------------------------------------------------------------------------
class Lab:
    def __init__(self, conn, top_liquid: Optional[int]):
        self.conn = conn
        self.top_liquid = top_liquid
        self.signals: pd.DataFrame = pd.DataFrame()
        self.paths: list = []        # aligned with signals.index: (n,3) float arrays
        self.entries: list = []
        self.n_days = 1

    def build(self):
        daily = fetch_df(self.conn, """
            SELECT ticker, timestamp, open, high, low, close, volume
            FROM stock_candles_synthesized WHERE timeframe='1d'
            ORDER BY ticker, timestamp
        """)
        daily["timestamp"] = pd.to_datetime(daily["timestamp"])
        daily["date"] = daily["timestamp"].dt.normalize()

        if self.top_liquid:
            d = daily.copy()
            d["dv"] = pd.to_numeric(d["close"], errors="coerce") * pd.to_numeric(d["volume"], errors="coerce")
            keep = set(d.groupby("ticker")["dv"].median().sort_values(ascending=False)
                       .head(self.top_liquid).index)
            daily = daily[daily["ticker"].isin(keep)]
            print(f"  liquidity filter: {len(keep)} tickers")

        # per-ticker features
        feats = []
        for tkr, g in daily.groupby("ticker", sort=True):
            if len(g) < 60:
                continue
            feats.append(add_features(g).assign(ticker=tkr))
        daily = pd.concat(feats, ignore_index=True)
        self.n_days = daily["date"].nunique()

        # market regime
        mkt = build_market_regime(daily)
        daily = daily.merge(mkt, left_on="date", right_index=True, how="left")

        # relative strength vs market (20d)
        daily["rs_20d"] = daily["ret_20d"] - daily["mkt_ret_20d"]

        # signals = entry trigger bars
        strat = EmaCross92150Strategy()
        c = pd.to_numeric(daily["close"], errors="coerce")
        cond = (
            (c > daily["ema_50"]) & (c > daily["ema_21"]) &
            daily["_above9"] & (~daily["_prev_above9"].fillna(False)) &
            (c >= 3.0) &
            daily[["ema_9", "ema_21", "ema_50"]].notna().all(axis=1)
        )
        sig = daily[cond].copy()
        print(f"  {len(sig)} raw signals across {sig['ticker'].nunique()} tickers")

        # forward 1h paths
        tickers = sorted(sig["ticker"].unique().tolist())
        candles = fetch_df(self.conn, """
            SELECT ticker, timestamp, open, high, low, close
            FROM stock_candles WHERE timeframe='1h' AND ticker = ANY(%s)
            ORDER BY ticker, timestamp
        """, (tickers,))
        candles["timestamp"] = pd.to_datetime(candles["timestamp"])
        by_t = {str(t): g.reset_index(drop=True) for t, g in candles.groupby("ticker")}

        rows, paths, entries = [], [], []
        for _, s in sig.iterrows():
            bars, entry = bars_after(by_t, s["ticker"], s["timestamp"])
            if bars.empty or entry <= 0:
                continue
            bars = bars.iloc[:N_TIMEOUT]
            arr = bars[["high", "low", "close"]].to_numpy(dtype=float)
            # MFE / MAE
            mfe = (arr[:, 0].max() - entry) / entry * 100
            mae = (arr[:, 1].min() - entry) / entry * 100
            r = s.to_dict()
            r["entry"] = entry
            r["mfe"] = mfe
            r["mae"] = mae
            rows.append(r)
            paths.append(arr)
            entries.append(entry)

        self.signals = pd.DataFrame(rows).reset_index(drop=True)
        self.paths = paths
        self.entries = entries
        self.signals["oos"] = self.signals["timestamp"] >= TRAIN_END
        print(f"  {len(self.signals)} trades with forward paths; {self.n_days} trading days")

    # ---- evaluation -------------------------------------------------------
    def evaluate(self, mask: pd.Series, cfg: dict) -> pd.DataFrame:
        idxs = np.where(mask.to_numpy())[0]
        out = []
        for i in idxs:
            res = walk_exit(self.paths[i], self.entries[i],
                            self.signals.iloc[i]["atr_pct"], cfg)
            if res["outcome"] == "no_data":
                continue
            out.append((i, res["pnl_pct"], res["outcome"]))
        if not out:
            return pd.DataFrame()
        df = pd.DataFrame(out, columns=["i", "pnl_pct", "outcome"])
        df["pnl_usd"] = df["pnl_pct"] / 100 * NOTIONAL
        df["oos"] = self.signals.iloc[df["i"].values]["oos"].values
        df["ts"] = self.signals.iloc[df["i"].values]["timestamp"].values
        return df


def metrics(df: pd.DataFrame, n_days: int) -> dict:
    if df is None or df.empty:
        return {"n": 0}
    wins = df[df["pnl_usd"] > 0]["pnl_usd"].sum()
    losses = -df[df["pnl_usd"] < 0]["pnl_usd"].sum()
    pf = wins / losses if losses > 0 else float("inf")
    return {
        "n": len(df), "pf": pf, "wr": (df["pnl_usd"] > 0).mean() * 100,
        "exp": df["pnl_usd"].mean(), "tot": df["pnl_usd"].sum(),
        "tpd": len(df) / max(1, n_days),
    }


def line(label, m):
    if m["n"] == 0:
        return f"  {label:<34} n=0"
    pf = "inf" if m["pf"] == float("inf") else f"{m['pf']:.2f}"
    return (f"  {label:<34} n={m['n']:<6} PF={pf:<5} WR={m['wr']:.1f}%  "
            f"exp=${m['exp']:+.2f}  tot=${m['tot']:+.0f}  {m['tpd']:.2f}/day")


def show(lab: Lab, title: str, mask: pd.Series, cfg: dict):
    df = lab.evaluate(mask, cfg)
    print(f"\n{title}  cfg={cfg}")
    print(line("ALL", metrics(df, lab.n_days)))
    if df is not None and not df.empty:
        tr_days = lab.signals[~lab.signals["oos"]]["timestamp"].dt.normalize().nunique()
        oo_days = lab.signals[lab.signals["oos"]]["timestamp"].dt.normalize().nunique()
        print(line("TRAIN", metrics(df[~df["oos"]], tr_days)))
        print(line("OOS", metrics(df[df["oos"]], oo_days)))
    return df


def dump_mfe(lab: Lab):
    s = lab.signals
    print("\n=== MFE / MAE distribution (over 10-day window, %) ===")
    for col in ["mfe", "mae"]:
        q = s[col].quantile([.1, .25, .5, .75, .9]).round(2)
        print(f"  {col}: mean={s[col].mean():.2f}  p10={q.iloc[0]} p25={q.iloc[1]} "
              f"p50={q.iloc[2]} p75={q.iloc[3]} p90={q.iloc[4]}")
    # how often MFE reaches various targets
    print("\n  P(MFE >= X%) :", {f"{x}": round((s['mfe'] >= x).mean(), 3)
                                  for x in [2, 3, 4, 5, 7, 10]})
    print("  P(MAE <= -X%):", {f"{x}": round((s['mae'] <= -x).mean(), 3)
                                for x in [1, 2, 2.5, 3, 5]})
    # MFE before MAE? proxy: trades green at all
    print("  median hold-to-MFE not tracked here; see exit sims.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-liquid", type=int, default=500)
    ap.add_argument("--dump-mfe", action="store_true")
    ap.add_argument("--experiments", action="store_true")
    ap.add_argument("--round2", action="store_true")
    ap.add_argument("--round3", action="store_true")
    ap.add_argument("--round4", action="store_true")
    ap.add_argument("--round5", action="store_true")
    args = ap.parse_args()

    conn = get_conn()
    try:
        lab = Lab(conn, args.top_liquid)
        print("Building cache ...")
        lab.build()
        if args.dump_mfe:
            dump_mfe(lab)
        if args.experiments:
            from stock_scanner.analysis import ema_cross_experiments as ex
            ex.run(lab, show, metrics, line)
        if args.round2:
            from stock_scanner.analysis import ema_cross_experiments as ex
            ex.run2(lab, show, metrics, line)
        if args.round3:
            from stock_scanner.analysis import ema_cross_experiments as ex
            ex.run3(lab, show, metrics, line)
        if args.round4:
            from stock_scanner.analysis import ema_cross_experiments as ex
            ex.run4(lab, show, metrics, line)
        if args.round5:
            from stock_scanner.analysis import ema_cross_experiments as ex
            ex.run5(lab, show, metrics, line)
    finally:
        conn.close()
