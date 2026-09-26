"""
One-time / repeatable backfill + backtest script for the PAXGUSD ORB monitor.

Reads the already-downloaded full-history 1-minute CSV (DAT_DELTA_PAXGUSD_M1.csv)
and does two things with the SAME strategy code the live monitor uses (imported
straight from delta_orb_action.py, never duplicated - so the backtest and the
live monitor can never silently drift into two different implementations):

  1. SEEDS the live dashboard's price chart: writes price_and_equity.csv with
     one row per historical bar so the "PAXGUSD price" chart has real history
     to show immediately. These historical rows carry a flat 0% return /
     unchanged balance - the live monitor hadn't started trading yet, so
     nothing is fabricated here. Also seeds state.json so the live monitor
     picks up cleanly from the last bar in this file, with no gap and no
     reprocessing.

  2. BACKTESTS the whole file: replays every bar through the exact same
     process_bar() logic as a "what would this strategy have done"
     simulation, and writes backtest_trades.csv / backtest_alerts.csv /
     backtest_price_and_equity.csv covering the whole span. The dashboard's
     "Strategy backtest" section lets you pick any start/end date+time and
     filters these down to that range.

Run this once after downloading history, or again any time you extend the
downloaded CSV further back / forward and want the backtest to cover it too.
It is NOT part of the GitHub Actions workflow - run it locally and commit the
files it writes.
"""
import bisect
import csv
import os
import sys
import datetime as dt

import pandas as pd

import delta_orb_action as doa

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "../DAT_DELTA_PAXGUSD_M1.csv"

# The price/equity line is chart backdrop, not used for any signal logic, so
# it's thinned to 1-in-N rows to keep the committed CSV (and each dashboard
# refresh's download) small. Trade/alert markers are unaffected - they stay
# at full, exact precision; only the plain background line is coarser.
CONTEXT_SAMPLE_EVERY = 15  # ~15-minute resolution instead of 1-minute


def load_bars(path):
    df = pd.read_csv(path)
    df["ts"] = df["time"].apply(
        lambda s: int(dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
                      .replace(tzinfo=dt.timezone.utc).timestamp()))
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    return df


def _epoch_seconds(idx):
    """Robust tz-aware DatetimeIndex -> int64 epoch seconds, independent of
    pandas' internal datetime64 storage resolution. IMPORTANT: a raw
    `idx.view('int64') // 10**9` (the previous version of this function)
    silently assumes nanosecond resolution, which was true in old pandas but
    is WRONG under pandas >=2.x/3.x, which stores datetime64 in microseconds
    by default - that bug made every ATR lookup here resolve to (effectively)
    the same wrong bucket, corrupting every backtested stop/target. Found and
    fixed via the chat while validating the 24/7 redesign - see that
    conversation for the before/after numbers."""
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return ((idx - epoch) // pd.Timedelta(seconds=1)).to_numpy()


def build_atr_lookup(df):
    """Reproduce atr14_m15()'s exact math (ATR14 over 15m true range, using
    only 15m bars fully closed before the lookup time) from the in-memory
    1-min frame instead of a live network call."""
    idx = pd.to_datetime(df["ts"], unit="s", utc=True)
    s = df.set_index(idx)[["open", "high", "low", "close"]]
    agg = s.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    agg["pc"] = agg["close"].shift(1)
    tr = pd.concat([
        agg["high"] - agg["low"],
        (agg["high"] - agg["pc"]).abs(),
        (agg["low"] - agg["pc"]).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()

    bucket_end_ts = _epoch_seconds(agg.index) + 15 * 60
    pairs = [(int(t), a) for t, a in zip(bucket_end_ts.tolist(), atr14.tolist()) if a == a]
    times = [t for t, _ in pairs]

    def atr_fn(bar_time):
        cutoff = bar_time - 15 * 60  # mirror atr14_m15: only bars closed before this
        i = bisect.bisect_right(times, cutoff) - 1
        return pairs[i][1] if i >= 0 else None

    return atr_fn


def build_ema_lookup(df, period=None):
    """Reproduce ema100_h1()'s exact math (EMA of close on 1h bars) from the
    in-memory 1-min frame instead of a live network call. Mirrors
    build_atr_lookup()'s bisect-lookup pattern exactly."""
    period = period or doa.EMA_PERIOD
    idx = pd.to_datetime(df["ts"], unit="s", utc=True)
    s = df.set_index(idx)["close"].resample("1h").last().dropna()
    ema = s.ewm(span=period, adjust=False).mean()
    bucket_end_ts = _epoch_seconds(ema.index) + 3600
    pairs = list(zip(bucket_end_ts.tolist(), ema.tolist()))
    times = [t for t, _ in pairs]

    def ema_fn(bar_time):
        i = bisect.bisect_right(times, bar_time) - 1
        return pairs[i][1] if i >= 0 else None

    return ema_fn


def fresh_state(last_bar_time=None):
    s = dict(last_bar_time=last_bar_time, open_trade=None,
             tiers={str(int(b)): {**doa.default_tier(), "balance": b, "peak": b} for b in doa.BALANCES},
             alerts_total=0, alerts_followed=0, alerts_not_followed=0,
             trades_total=0, trades_wins=0, trades_losses=0, sum_R=0.0,
             trades_with_alert=0, wins_with_alert=0, trades_without_alert=0, wins_without_alert=0)
    s.update(doa.default_day_state(None))
    s.update(doa.default_session_state(None))
    return s


def run():
    print(f"Loading {CSV_PATH} ...")
    df = load_bars(CSV_PATH)
    if df.empty:
        print("No rows loaded - check the CSV path.")
        return
    print(f"Loaded {len(df)} bars: {doa.utc_iso(int(df['ts'].iloc[0]))} .. {doa.utc_iso(int(df['ts'].iloc[-1]))}")

    # ---------- 1) seed price_and_equity.csv (live chart context) + state.json ----------
    doa.PRICE_LOG = "price_and_equity.csv"
    doa.STATE_FILE = "state.json"
    for f in (doa.PRICE_LOG,):
        if os.path.exists(f):
            os.remove(f)
    doa.ensure_csv(doa.PRICE_LOG, doa.PRICE_HEADER)
    seed_df = df.iloc[::CONTEXT_SAMPLE_EVERY]
    if seed_df.index[-1] != df.index[-1]:
        seed_df = pd.concat([seed_df, df.iloc[[-1]]])  # always keep the true last bar
    with open(doa.PRICE_LOG, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for row in seed_df.itertuples():
            w.writerow([doa.utc_iso(int(row.ts)), row.close, 0.0] + [b for b in doa.BALANCES])

    last_ts = int(df["ts"].iloc[-1])
    doa.save_state(fresh_state(last_bar_time=last_ts))
    print(f"Seeded {doa.PRICE_LOG} ({len(seed_df)} of {len(df)} bars, ~{CONTEXT_SAMPLE_EVERY}min resolution, "
          f"flat/no-trade context) and {doa.STATE_FILE} "
          f"(last_bar_time={last_ts} = {doa.utc_iso(last_ts)}) - "
          f"the live monitor will continue forward from here at full 1-min resolution.")

    # ---------- 2) full historical backtest, same process_bar() logic ----------
    doa.TRADES_LOG = "backtest_trades.csv"
    doa.ALERTS_LOG = "backtest_alerts.csv"
    doa.PRICE_LOG = "backtest_price_and_equity.csv"
    doa.LIVE_LOG = "backtest_log.txt"
    doa.SUMMARY_FILE = "BACKTEST_SUMMARY.md"
    doa.ATR_FN = build_atr_lookup(df)
    doa.EMA_FN = build_ema_lookup(df)

    for f in (doa.TRADES_LOG, doa.ALERTS_LOG, doa.PRICE_LOG, doa.LIVE_LOG):
        if os.path.exists(f):
            os.remove(f)

    # process_bar()/log_line() call append_csv/ensure_csv on every single bar
    # (price_and_equity row) or event - fine for the live monitor's handful of
    # calls per run, but 300k+ individual file open/close cycles for a full
    # backtest is slow, especially over a mounted/synced folder. Buffer them
    # in memory here and flush once at the end - same output, same order.
    _buffers = {}

    def _buffered_ensure_csv(path, header):
        _buffers.setdefault(path, {"header": header, "rows": []})

    def _buffered_append_csv(path, header, row):
        _buffers.setdefault(path, {"header": header, "rows": []})["rows"].append(row)

    def _buffered_log_line(msg):
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        _buffers.setdefault(doa.LIVE_LOG, {"header": None, "rows": []})["rows"].append([f"[{ts}] {msg}"])

    doa.ensure_csv = _buffered_ensure_csv
    doa.append_csv = _buffered_append_csv
    doa.log_line = _buffered_log_line

    bt_state = fresh_state()
    t0 = dt.datetime.now()
    n = len(df)
    for i, row in enumerate(df.itertuples()):
        bar = dict(time=int(row.ts), open=row.open, high=row.high, low=row.low, close=row.close)
        doa.process_bar(bt_state, bar)
        bt_state["last_bar_time"] = bar["time"]
        if i and i % 50000 == 0:
            print(f"  ...{i}/{n} bars ({(dt.datetime.now()-t0).total_seconds():.0f}s elapsed)")

    print(f"Replayed {n} bars in {(dt.datetime.now()-t0).total_seconds():.1f}s. Writing output files...")

    # thin the backtest price/equity backdrop the same way (trades/alerts keep full precision)
    price_buf = _buffers.get(doa.PRICE_LOG)
    if price_buf and price_buf["rows"]:
        thinned = price_buf["rows"][::CONTEXT_SAMPLE_EVERY]
        if thinned[-1] is not price_buf["rows"][-1]:
            thinned.append(price_buf["rows"][-1])
        price_buf["rows"] = thinned

    for path, buf in _buffers.items():
        with open(path, "w" if buf["header"] else "a", newline="", encoding="utf-8") as fh:
            if buf["header"] is not None:
                w = csv.writer(fh)
                w.writerow(buf["header"])
                w.writerows(buf["rows"])
            else:
                fh.write("\n".join(r[0] for r in buf["rows"]) + ("\n" if buf["rows"] else ""))

    doa.write_summary(bt_state)
    print(f"\nBacktest done over the full {len(df)}-bar history:")
    print(f"  Trades: {bt_state['trades_total']}  Wins: {bt_state['trades_wins']}  "
          f"Losses: {bt_state['trades_losses']}  Sum R: {bt_state['sum_R']:.2f}")
    for b in doa.BALANCES:
        t = bt_state["tiers"][str(int(b))]
        print(f"  ${b:,.0f} -> ${t['balance']:,.2f} ({(t['balance']/b-1)*100:+.2f}%), max DD {t['max_dd_pct']:.2f}%")
    print(f"\nWrote: {doa.TRADES_LOG}, {doa.ALERTS_LOG}, {doa.PRICE_LOG}, {doa.LIVE_LOG}, {doa.SUMMARY_FILE}")


if __name__ == "__main__":
    run()
