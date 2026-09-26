"""
STATELESS PAXGUSD ORB monitor, designed to run as a GitHub Actions job on a
schedule (see .github/workflows/live-monitor.yml). No API key needed - all
data used here is Delta Exchange's public market data.

*** Settings below are the COMEX-gold-validated ORB parameters, carried over *
*** as a provisional starting point. They have NOT yet been separately     *
*** validated on PAXGUSD's own history - update them once that backtest is *
*** done.                                                                  *

WHY "STATELESS": every GitHub Actions run starts a brand-new, empty machine
with no memory of the last run. So this script:
  1. loads state.json (committed in the repo) to see where it left off
  2. re-fetches every 1-minute PAXGUSD candle since the last bar it processed,
     from Delta's server (the authoritative, unchanging record) - all day,
     every day, not just during the trading session
  3. replays only the candles it hasn't processed yet through the EXACT same
     bar-by-bar logic as the original backtest/live-monitor script
  4. saves its new position back to state.json
This makes the result identical to what a continuously-running version would
have computed for the same closed candles - see the chat explanation for why.
The one real difference is notification timing (checked every few minutes
here, not instantly), not the correctness of what gets logged.

Fetching is intentionally NOT restricted to the ORB session window - it runs
around the clock (see the workflow's cron) so the price/equity chart stays
"live" all day, even though process_bar() itself only opens ranges/trades/
alerts during the actual session (RANGE_START_H..TRADE_END_H, NY time) - that
part of the logic is unchanged.

This file doubles as the backtest engine: run_backtest.py imports it, points
ATR_FN and the *_LOG file constants at a local OHLC lookup and backtest_*.csv
filenames, and replays a full historical CSV bar-by-bar through the exact
same process_bar() used live - so the backtest and the live monitor can never
silently drift apart into two different implementations of "the strategy."

FILES this writes/updates in the repo (the workflow commits them every run):
  state.json       - internal bookkeeping, not meant to be read by a person
  live_log.txt     - human-readable running narrative (open this to just read)
  alerts_log.csv   - every early-warning alert: FIRED, then RESOLVED with
                     whether a real breakout followed
  trades_log.csv   - every paper trade, with entry/exit/result/R, AND the
                     dollar effect on three simulated account sizes
  price_and_equity.csv - one row per processed 1-min bar: PAXGUSD close +
                     running % return (same for every balance tier, since
                     they're all risking the same 1% - only the dollar
                     amounts differ) + each tier's dollar balance. This is
                     what a dashboard chart should plot.
  SUMMARY.md       - always-current headline stats, rewritten every run

All timestamps written to the CSVs are UTC (ISO 8601, e.g. 2026-09-26T12:34:00Z)
- unambiguous, so any viewer (dashboard, spreadsheet) can convert to whatever
local time zone it needs (New York market time, IST, etc.) itself.
"""

import csv
import json
import os
import time
import datetime as dt
from collections import deque

import requests

# ============================== settings ===================================
# v2 (24/7 multi-session redesign) - see chat for the full research writeup.
# PAXGUSD trades around the clock as a crypto perpetual, so instead of one
# NY-session-anchored opening range per day, the "opening range" concept now
# repeats every SESSION_HOURS, all day, every day: each block gets its own
# RANGE_MINUTES range, its own breakout watch, independent of the others.
# Params below (SESSION_HOURS=4, RANGE_MINUTES=15, ATR_STOP_MULT=0.75, RR=2.0)
# were chosen by a train/test split over the full downloaded history (not
# just curve-fit to the whole sample) - see run_backtest.py's summary and the
# chat writeup for the honest out-of-sample numbers, including where this
# still falls short of a 50-60% win rate target.
BASE_URL = "https://cdn.india.deltaex.org"
SYMBOL = "PAXGUSD"

SESSION_HOURS = 4          # a fresh opening-range "session" every 4 UTC hours (6/day)
RANGE_MINUTES = 15         # minutes at the start of each session used to set the range

ATR_STOP_MULT = 0.75
RR = 2.0                   # 1:2 risk-reward, as requested
MAX_RANGE_ATR_MULT = 2.0   # skip a session if its opening range is too wide vs ATR (noisy/no edge)
ALERT_ATR_FRAC = 0.20
MOMENTUM_BARS = 3          # require this many consecutive rising/falling closes to confirm
MAX_TRADES_PER_DAY = 3     # hard cap, resets at UTC midnight

BALANCES = [100.0, 1000.0, 10000.0]   # the three simulated account sizes
RISK_PERCENT = 1.0                    # fixed % risk per trade, same as TEST_fixed.py

STATE_FILE = "state.json"
LIVE_LOG = "live_log.txt"
ALERTS_LOG = "alerts_log.csv"
TRADES_LOG = "trades_log.csv"
PRICE_LOG = "price_and_equity.csv"
SUMMARY_FILE = "SUMMARY.md"

TRADES_HEADER = (
    ["entry_time_utc", "side", "entry", "sl", "tp", "exit_time_utc", "exit", "result", "R", "had_alert"]
    + [f"{int(b)}_before" for b in BALANCES]
    + [f"{int(b)}_pnl" for b in BALANCES]
    + [f"{int(b)}_after" for b in BALANCES]
)
ALERTS_HEADER = ["event", "alert_id", "time_utc", "side", "price", "range_high", "range_low", "atr", "outcome"]
PRICE_HEADER = ["time_utc", "close", "pct_return"] + [f"{int(b)}_balance" for b in BALANCES]


def utc_iso(ts):
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------ small helpers -------------------------------
def log_line(msg):
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LIVE_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ensure_csv(path, header):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def append_csv(path, header, row):
    ensure_csv(path, header)
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def fetch_candles(session, resolution, start, end):
    r = session.get(f"{BASE_URL}/v2/history/candles",
                     params=dict(symbol=SYMBOL, resolution=resolution, start=start, end=end),
                     timeout=20)
    r.raise_for_status()
    data = r.json()
    rows = data.get("result", []) if data.get("success") else []
    return sorted(rows, key=lambda r: r["time"])


def atr14_m15(session, before_ts):
    rows = fetch_candles(session, "15m", before_ts - 15 * 60 * 30, before_ts - 15 * 60)
    if len(rows) < 15:
        return None
    trs = []
    for i in range(1, len(rows)):
        h, l, pc = rows[i]["high"], rows[i]["low"], rows[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs[-14:]) / 14


def _live_atr_fn(bar_time):
    return atr14_m15(SESSION, bar_time)


# process_bar() calls ATR_FN(bar_time), never atr14_m15 directly, so a backtest
# script can point this at a local, already-loaded OHLC lookup instead of a
# live network call, without touching a single line of the strategy logic.
ATR_FN = _live_atr_fn


# ------------------------------ trend filter ---------------------------------
# Backtesting (see chat) showed a raw breakout entry is NOT robust out-of-
# sample: it looks great on whichever slice of history it was tuned on and
# falls apart on the next one. Requiring the breakout to agree with a slower,
# higher-timeframe trend (1h EMA-100) was the one filter that held up in BOTH
# the train and the held-out test windows - see EMA_PERIOD below.
EMA_PERIOD = 100
EMA_LOOKBACK_HOURS = 24 * 30  # 30 days of 1h candles - plenty for a 100-period EMA to converge


def ema_of_closes(closes, period):
    if len(closes) < period:
        return None
    ema = closes[0]
    k = 2 / (period + 1)
    for px in closes[1:]:
        ema = px * k + ema * (1 - k)
    return ema


def ema100_h1(session, before_ts):
    rows = fetch_candles(session, "1h", before_ts - EMA_LOOKBACK_HOURS * 3600, before_ts)
    return ema_of_closes([r["close"] for r in rows], EMA_PERIOD)


def _live_ema_fn(bar_time):
    return ema100_h1(SESSION, bar_time)


# same dependency-injection pattern as ATR_FN, for the same reason
EMA_FN = _live_ema_fn


def utc_day_key(ts_utc):
    return dt.datetime.fromtimestamp(ts_utc, tz=dt.timezone.utc).date().isoformat()


def default_tier():
    return dict(balance=None, peak=None, max_dd_pct=0.0, trades=0, wins=0, losses=0)


def default_day_state(day_key):
    """Resets once per UTC calendar day - just the trade counter."""
    return dict(day=day_key, trades_today=0)


def default_session_state(session_key):
    """Resets every SESSION_HOURS (a fresh opening range each time), independent
    of the once-a-day trade counter above. NOTE: open_trade is NOT part of this -
    a position can run past a session boundary until it hits SL/TP, so it lives
    at the top level of state instead (see load_state/process_bar)."""
    return dict(session=session_key, range_high=None, range_low=None, day_atr=None, session_ema=None,
                range_ok=False, alerted_up=False, alerted_dn=False, closes_before=[])


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    s = dict(last_bar_time=None, open_trade=None,
              tiers={str(int(b)): {**default_tier(), "balance": b, "peak": b} for b in BALANCES},
              alerts_total=0, alerts_followed=0, alerts_not_followed=0,
              trades_total=0, trades_wins=0, trades_losses=0, sum_R=0.0,
              trades_with_alert=0, wins_with_alert=0, trades_without_alert=0, wins_without_alert=0)
    s.update(default_day_state(None))
    s.update(default_session_state(None))
    return s


def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)


# ------------------------------- main engine --------------------------------
def close_trade(state, ot, result, exit_px, exit_iso):
    risk = abs(ot["entry"] - ot["sl"])
    R = ((exit_px - ot["entry"]) / risk if ot["side"] == 1 else (ot["entry"] - exit_px) / risk)
    state["trades_total"] += 1
    state["sum_R"] += R
    if R > 0:
        state["trades_wins"] += 1
    else:
        state["trades_losses"] += 1
    if ot["had_alert"]:
        state["trades_with_alert"] += 1
        if R > 0:
            state["wins_with_alert"] += 1
    else:
        state["trades_without_alert"] += 1
        if R > 0:
            state["wins_without_alert"] += 1

    befores, pnls, afters = [], [], []
    for b in BALANCES:
        key = str(int(b))
        tier = state["tiers"][key]
        before = tier["balance"]
        risk_dollars = before * (RISK_PERCENT / 100.0)
        pnl = R * risk_dollars
        after = before + pnl
        tier["balance"] = after
        tier["peak"] = max(tier["peak"], after)
        dd = (after - tier["peak"]) / tier["peak"] * 100.0 if tier["peak"] > 0 else 0.0
        tier["max_dd_pct"] = min(tier["max_dd_pct"], dd)
        tier["trades"] += 1
        if R > 0:
            tier["wins"] += 1
        else:
            tier["losses"] += 1
        befores.append(round(before, 2))
        pnls.append(round(pnl, 2))
        afters.append(round(after, 2))

    log_line(f"PAPER TRADE CLOSED  {('BUY' if ot['side']==1 else 'SELL')} "
             f"entry {ot['entry']:.2f} -> exit {exit_px:.2f}  result={result}  R={R:.2f}  "
             f"(had_alert={ot['had_alert']})")
    append_csv(TRADES_LOG, TRADES_HEADER,
               [ot["entry_time"], "BUY" if ot["side"] == 1 else "SELL", ot["entry"], ot["sl"],
                ot["tp"], exit_iso, exit_px, result, round(R, 3), ot["had_alert"]]
               + befores + pnls + afters)


def _finalize_unresolved_alerts(state, as_of_iso, last_price):
    """Called right before a session resets: any alert that FIRED this session
    but never saw a real breakout gets logged RESOLVED/NOT_FOLLOWED, so the
    dashboard's indicator-accuracy stat doesn't leave it hanging forever."""
    for flag_key, side_key, label in (("alerted_up", "UP", "BUY"), ("alerted_dn", "DN", "SELL")):
        if state.get(flag_key):
            aid = f"{state['session']}-{side_key}"
            state["alerts_not_followed"] += 1
            append_csv(ALERTS_LOG, ALERTS_HEADER,
                       ["RESOLVED", aid, as_of_iso, label, last_price,
                        state["range_high"], state["range_low"],
                        state["day_atr"], "NOT_FOLLOWED"])


def process_bar(state, bar):
    ts = bar["time"]
    day_key = utc_day_key(ts)
    if state.get("day") != day_key:
        state.update(default_day_state(day_key))

    session_key = ts // (SESSION_HOURS * 3600)
    o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]

    # --- manage an open paper trade first (can run past a session boundary -
    # there's no NY-style "flatten before close": PAXGUSD trades 24/7) ---
    ot = state["open_trade"]
    if ot is not None:
        hit_sl = (h >= ot["sl"]) if ot["side"] == -1 else (l <= ot["sl"])
        hit_tp = (l <= ot["tp"]) if ot["side"] == -1 else (h >= ot["tp"])
        if hit_sl or hit_tp:
            result, exit_px = ("SL", ot["sl"]) if hit_sl else ("TP", ot["tp"])
            close_trade(state, ot, result, exit_px, utc_iso(ts))
            state["open_trade"] = None

    if session_key != state.get("session"):
        if state.get("session") is not None:
            _finalize_unresolved_alerts(state, utc_iso(ts), c)
        state.update(default_session_state(session_key))

    session_start = session_key * SESSION_HOURS * 3600
    secs_into_session = ts - session_start

    # --- opening range build (first RANGE_MINUTES of each session block) ---
    # NOTE: falls through to the price/equity row at the bottom either way -
    # only the alert/signal detection below is skipped during this window.
    in_range_window = secs_into_session < RANGE_MINUTES * 60
    if in_range_window:
        state["range_high"] = h if state["range_high"] is None else max(state["range_high"], h)
        state["range_low"] = l if state["range_low"] is None else min(state["range_low"], l)
    else:
        if not state["range_ok"] and state["range_high"] is not None and state["day_atr"] is None:
            a = ATR_FN(ts)
            state["day_atr"] = a
            state["session_ema"] = EMA_FN(ts)  # computed once per session, held for its duration
            if a:
                rng_w = state["range_high"] - state["range_low"]
                state["range_ok"] = rng_w <= MAX_RANGE_ATR_MULT * a
                verdict = "TRADEABLE" if state["range_ok"] else "SKIPPED (range too wide vs ATR)"
                log_line(f"Session {session_key} range set: {state['range_low']:.2f}-{state['range_high']:.2f} "
                         f"(width {rng_w:.2f}, ATR15 {a:.2f}, EMA100(1h) {state['session_ema']}) -> {verdict}")

        closes_before = deque(state["closes_before"], maxlen=MOMENTUM_BARS)

        # --- alerts + real signal --- (up to MAX_TRADES_PER_DAY fills/day, no
        # extra per-session trade limit - the daily cap and "flat to re-enter"
        # are the only gates, matching what was actually validated - see chat)
        if state["range_ok"] and state["trades_today"] < MAX_TRADES_PER_DAY and state["open_trade"] is None:
            rh, rl, a = state["range_high"], state["range_low"], state["day_atr"]
            alert_zone = ALERT_ATR_FRAC * a

            if len(closes_before) == MOMENTUM_BARS:
                prior = list(closes_before)
                rising = all(prior[k] > prior[k - 1] for k in range(1, len(prior)))
                falling = all(prior[k] < prior[k - 1] for k in range(1, len(prior)))

                if not state["alerted_up"] and c < rh and (rh - h) <= alert_zone and rising:
                    state["alerted_up"] = True
                    aid = f"{session_key}-UP"
                    log_line(f"\U0001F514 ALERT FIRED - WATCH BUY  approaching {rh:.2f}  [alert_id={aid}]")
                    state["alerts_total"] += 1
                    append_csv(ALERTS_LOG, ALERTS_HEADER,
                               ["FIRED", aid, utc_iso(ts), "BUY", c, rh, rl, a, ""])

                if not state["alerted_dn"] and c > rl and (l - rl) <= alert_zone and falling:
                    state["alerted_dn"] = True
                    aid = f"{session_key}-DN"
                    log_line(f"\U0001F514 ALERT FIRED - WATCH SELL  approaching {rl:.2f}  [alert_id={aid}]")
                    state["alerts_total"] += 1
                    append_csv(ALERTS_LOG, ALERTS_HEADER,
                               ["FIRED", aid, utc_iso(ts), "SELL", c, rh, rl, a, ""])

            # momentum-confirmed breakout: require MOMENTUM_BARS consecutive
            # closes in the breakout's direction, not just a single spike
            mom_up_ok = len(closes_before) == MOMENTUM_BARS and all(
                closes_before[k] > closes_before[k - 1] for k in range(1, len(closes_before)))
            mom_dn_ok = len(closes_before) == MOMENTUM_BARS and all(
                closes_before[k] < closes_before[k - 1] for k in range(1, len(closes_before)))

            # trend filter: only take the breakout if it agrees with the slower
            # 1h EMA-100 - this is what separated "works out-of-sample" from
            # "looked great on the tuning window" in the chat's validation
            ema = state["session_ema"]
            trend_up_ok = ema is None or c > ema
            trend_dn_ok = ema is None or c < ema

            side = 1 if (c > rh and mom_up_ok and trend_up_ok) else (
                -1 if (c < rl and mom_dn_ok and trend_dn_ok) else 0)
            if side != 0:
                had_alert = state["alerted_up"] if side == 1 else state["alerted_dn"]
                aid = f"{session_key}-{'UP' if side == 1 else 'DN'}"
                if had_alert:
                    state["alerts_followed"] += 1
                    append_csv(ALERTS_LOG, ALERTS_HEADER,
                               ["RESOLVED", aid, utc_iso(ts), "BUY" if side == 1 else "SELL",
                                c, rh, rl, a, "BREAKOUT_FOLLOWED"])
                    # this alert is now settled - don't finalize it again as
                    # NOT_FOLLOWED at session end, and let the same side re-arm
                    state["alerted_up" if side == 1 else "alerted_dn"] = False
                sl = c - ATR_STOP_MULT * a if side == 1 else c + ATR_STOP_MULT * a
                tp = c + RR * ATR_STOP_MULT * a if side == 1 else c - RR * ATR_STOP_MULT * a
                log_line(f">>> {'BUY' if side==1 else 'SELL'} SIGNAL  entry~{c:.2f}  SL {sl:.2f}  TP {tp:.2f}  "
                         f"(had_alert={had_alert}, trade {state['trades_today']+1}/{MAX_TRADES_PER_DAY} today)")
                state["trades_today"] += 1
                state["open_trade"] = dict(side=side, entry=c, sl=sl, tp=tp,
                                            entry_time=utc_iso(ts), had_alert=had_alert)

        closes_before.append(c)
        state["closes_before"] = list(closes_before)

    # --- price + equity history, one row per processed bar (for the dashboard chart) ---
    base_bal0 = BALANCES[0]
    base_tier = state["tiers"][str(int(base_bal0))]
    pct_return = (base_tier["balance"] / base_bal0 - 1) * 100
    append_csv(PRICE_LOG, PRICE_HEADER,
               [utc_iso(bar["time"]), c, round(pct_return, 4)]
               + [round(state["tiers"][str(int(b))]["balance"], 2) for b in BALANCES])


def write_summary(state):
    lines = []
    now_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append("# PAXGUSD ORB - live forward-test summary\n")
    lines.append(f"Last updated: {now_utc}\n")
    lines.append("**Provisional settings copied from COMEX-gold backtest - not yet separately "
                  "validated on PAXGUSD. Paper trading only, no real money involved. "
                  "Checked on a schedule (see workflow) - notification lag applies.**\n")

    at, af, anf = state["alerts_total"], state["alerts_followed"], state["alerts_not_followed"]
    resolved = af + anf
    lines.append("## Early-warning indicator accuracy\n")
    lines.append(f"- Alerts fired: {at}\n")
    lines.append(f"- Resolved so far: {resolved} (followed by real breakout: {af}, not followed: {anf})\n")
    if resolved:
        lines.append(f"- Follow-through rate: {af/resolved*100:.1f}%\n")

    tt, tw, tl, sr = state["trades_total"], state["trades_wins"], state["trades_losses"], state["sum_R"]
    lines.append("\n## Trades (paper)\n")
    lines.append(f"- Total: {tt}  |  Wins: {tw}  |  Losses: {tl}\n")
    if tt:
        lines.append(f"- Win rate: {tw/tt*100:.1f}%\n")
        lines.append(f"- Total R: {sr:.2f}  |  Avg R/trade: {sr/tt:.3f}\n")
    twa, wwa, twoa, wwoa = (state["trades_with_alert"], state["wins_with_alert"],
                            state["trades_without_alert"], state["wins_without_alert"])
    if twa:
        lines.append(f"- Trades WITH a prior alert: {twa}, win rate {wwa/twa*100:.1f}%\n")
    if twoa:
        lines.append(f"- Trades WITHOUT a prior alert: {twoa}, win rate {wwoa/twoa*100:.1f}%\n")

    lines.append("\n## Simulated account balances (1% risk per trade, compounding)\n")
    lines.append("| Starting balance | Current balance | Return | Max drawdown | Trades | Win rate |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for b in BALANCES:
        tier = state["tiers"][str(int(b))]
        ret_pct = (tier["balance"] / b - 1) * 100
        wr = (tier["wins"] / tier["trades"] * 100) if tier["trades"] else 0.0
        lines.append(f"| ${b:,.0f} | ${tier['balance']:,.2f} | {ret_pct:+.2f}% | "
                      f"{tier['max_dd_pct']:.2f}% | {tier['trades']} | {wr:.1f}% |\n")

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)


SESSION = requests.Session()


def main():
    state = load_state()
    now = int(time.time())

    # Continuous, all-day fetching: pick up right where the last run left off,
    # regardless of what time of day (or NY session) it currently is. The
    # workflow's cron now fires every 5 minutes around the clock, so this
    # normally only needs to cover the last few minutes; the 6-hour fallback
    # only matters on a genuinely first-ever run with no state.json yet
    # (normally state.json is seeded with real history before this ever runs -
    # see the chat/README for the one-time backfill step).
    last_bar_time = state.get("last_bar_time")
    fetch_from = (last_bar_time + 60) if last_bar_time else (now - 6 * 3600)

    rows = fetch_candles(SESSION, "1m", fetch_from, now)
    new_rows = [r for r in rows
                if (last_bar_time is None or r["time"] > last_bar_time) and r["time"] + 60 <= now]

    if not new_rows:
        log_line("No new closed 1-min bars since last run - nothing to do.")
    else:
        for bar in new_rows:
            process_bar(state, bar)
            state["last_bar_time"] = bar["time"]

    write_summary(state)
    save_state(state)


if __name__ == "__main__":
    main()
