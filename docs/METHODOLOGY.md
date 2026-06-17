# Methodology — How I Tested for an Edge (and What Survived)

This is the honest record of a crypto trading investigation: what I built, how I
tried to disprove it, what failed, and the one thing that held up. The goal of
publishing it is not to sell a strategy — it's to show the *method*, because the
method is the transferable part.

> All figures below come from historical data and paper simulation. They are not
> a live track record. See [`../DISCLAIMER.md`](../DISCLAIMER.md).

---

## 0. The premise

I built an AI-driven directional bot: compute deterministic features (RSI, ATR,
order-book imbalance, regime) → ask an LLM for a direction, stop, and targets →
pass it through a multi-gate risk engine → execute (paper or MEXC). It ran 24/7,
logged everything, and *looked* sophisticated.

The mistake most people make is to stop there — to watch a paper equity curve
drift up and conclude they have something. I tried to do the opposite: assume it
has **no** edge until the data forces me to admit otherwise.

A useful unit throughout: **R**, the profit/loss of a trade expressed in
multiples of the risk taken (distance to stop). An edge has to show a positive
*expected* R **after costs**, out-of-sample. Everything below is measured in R.

---

## 1. Disproof #1 — the mechanical signal dies to costs

First I stripped the LLM out entirely and measured the raw mechanical signal the
features implied, over historical candles.

| Measurement | Expected R per trade |
|---|---|
| Signal, gross (no costs) | **+0.035R** |
| Fees only | −0.091R |
| Fees + realistic slippage | **−0.243R** |

The signal was *barely* positive gross — and the slippage (−0.187R) alone was
larger than the fees. After modeling real execution, a tiny positive edge became
a clear negative one. This is the single most common way retail strategies lie
to themselves: they backtest on mid-price with no slippage.

Tooling: `sentinel/backtest/` (engine, metrics) and `tests/test_costs.py`.

## 2. Disproof #2 — the LLM doesn't rescue it

Maybe the LLM adds something the mechanical signal doesn't. To test that fairly,
I built a **gated replay**: take real historical setups, ask the model for a
decision, and apply the *same* live risk-engine geometry gates, so the replay
matches what the bot would actually have traded.

| Model | Gated replay (n) | Expected R |
|---|---|---|
| Cerebras (Llama) | 39 | **−0.309R** |
| Gemini | 7 | **−0.363R** |

Both negative. The LLM did not beat break-even. It produced confident,
well-structured trade plans — and lost money net of costs, just like the
mechanical signal. A plausible-sounding narrative is not an edge.

To be explicit about rigor: these samples are **small** (n=39 and n=7), so on
their own they're suggestive, not conclusive. But they don't stand alone — the
*underlying* mechanical signal was already negative after costs (§1), so the LLM
would have to add enough alpha to overcome both the signal and the costs, and it
showed no sign of doing so. The honest claim is "no evidence of an edge," not a
tight statistical bound.

Tooling: `sentinel/backtest/llm_replay.py` (note the `--no-gates` and `--dump`
flags) and `sentinel/backtest/sweep.py` (offline R:R-floor sweep over the dumped
tape).

## 3. Disproof #3 — pairs/stat-arb is overfit

If directional doesn't work, maybe market-neutral statistical arbitrage does. I
built a pairs backtester with proper cointegration testing (ADF) and a
walk-forward out-of-sample split.

Result: **every pair that looked profitable in-sample reversed out-of-sample.**
The in-sample "edge" was the optimizer fitting noise. Walk forward, and it
inverts. This is the textbook signature of overfitting, and it's why an
in-sample backtest is worth almost nothing without an OOS split.

(An early version of the backtester even had a subtle bug — computing PnL from a
*time-varying* hedge ratio rather than fixing the ratio at entry — which
produced absurd numbers and was a good reminder to be suspicious of results that
are too large in either direction.)

## 4. Disproof #4 — cross-exchange funding is already arbitraged

Funding rates differ between exchanges. Could you capture the difference (long
the cheap-funding venue, short the expensive one)? I measured the MEXC-vs-Binance
funding differential on majors.

Result: **~0.4%/yr.** It's already arbitraged away by people faster and
better-capitalized than me. The gross spread that exists is smaller than the
costs of capturing it. No edge.

---

## 5. What survived — static delta-neutral funding carry

The one thing that held up is the simplest. On a perpetual future, longs and
shorts exchange a **funding** payment every 8 hours. When an asset is
*structurally hard to short* — privacy coins, tokens with constrained borrow,
hyped low-float names — perp longs persistently outweigh shorts, and **longs pay
shorts** to hold the position, cycle after cycle.

You can harvest that, market-neutral, by holding:

```
  long  spot   (notional N)
  short perp   (notional N)   ← collects funding every 8h
```

The spot and perp legs cancel on price (delta-neutral), so your PnL is
essentially **funding minus costs**. You are not predicting anything. You are
collecting a structural premium that exists because other people want leverage
on something that's hard to short.

### 5.1 The key insight: consistency beats headline rate

The naive move is to chase the highest funding rate. That's wrong, because a
high rate that flips sign forces you to unwind (and pay costs) repeatedly. What
actually matters for a *static hold* is **consistency** — how reliably funding
stays on one side.

> A name paying **+13%/yr that is 100% one-sided** beats a name paying
> **+40%/yr that is only 60% one-sided**, because the second one's negative
> stretches force costly exits that eat the headline.

So the scanner ranks by carry weighted by the **square** of consistency, not by
raw rate (`sentinel/carry/scanner.py::carry_score`). Names that pass also have to
clear a **spot-liquidity filter** — a great funding rate on a token with a
$100k spot market can't actually be hedged delta-neutral.

### 5.2 Hysteresis → near-zero turnover

The exit rule isn't "leave the moment funding dips." It's a **hysteresis** rule:
unwind only after *N consecutive* negative settlements (default 3). This avoids
churning a working position on a single blip. In a 400-day simulation on XMR,
this produced only ~6 round-trips total — turnover is almost nothing, which is
exactly why costs don't eat the edge here the way they ate the directional
signal. (`sentinel/carry/simulator.py`.)

### 5.3 Breadth — the one free lunch

Grinold's Fundamental Law: `IR = IC × √breadth`. If you harvest the same modest
edge across *N uncorrelated* names, portfolio risk-adjusted return scales with
√N. The question is whether funding across alts is actually uncorrelated.

I measured it: average pairwise funding correlation across the basket was
**~0.03** — essentially uncorrelated — giving an **effective breadth of ~9.3 out
of 12** names. So diversifying across the basket is a real, measurable
improvement, not a diversification illusion. Sizing is **risk-parity**
(inverse-vol, with an iterative per-name cap) in
`sentinel/carry/executor.py::risk_parity_notionals`.

### 5.4 What it pays, honestly

| Metric | Value (backtest / paper) |
|---|---|
| Net yield, after costs (e.g. XMR, 400d) | **~15–20%/yr** |
| Worst drawdown | **< 1%** |
| Round-trips (XMR, 400d) | ~6 (very low turnover) |
| Market exposure | ~0 (delta-neutral) |
| **Capacity ceiling** | **~$100k order-of-magnitude** (per-name, liquidity-dependent — see §5.6) |

A market-neutral ~15% with sub-1% drawdown is genuinely good. The catch is
**capacity**: the spot markets on these names are thin, so the strategy tops out
around $100k before slippage starts eating the edge. This is a small-capital
strategy, structurally. It cannot be scaled into an institutional product — which
is, ironically, *why* the edge still exists.

### 5.5 Guarding against survivorship bias — walk-forward validation

The obvious objection to a consistency-screened basket is that it's chosen *with
hindsight*: rank names by how one-sided their funding was over the whole history,
then "backtest" on that same history — of course it looks good. That's
survivorship/look-ahead bias, and it's the first thing a skeptic should attack.

`sentinel/carry/walkforward.py` answers it directly by rolling a train→test
window across the funding history:

- the basket is selected using **only the training window** (past data);
- realised funding is then measured on the **following test window** the
  selector never saw (genuinely out-of-sample);
- per-window OOS yields are **bootstrapped** into a 95% confidence interval and
  compared against a **random-selection baseline** over the same windows.

The logic is the falsifiable part. If the screen only worked in hindsight, the
out-of-sample yield collapses toward the random baseline and the CI straddles
zero. If the edge is real, the OOS mean stays positive, its CI excludes zero, and
it beats random selection. Run it yourself (figures will reflect *current* data,
not a fixed snapshot — the point is that the **method** survives, not that a
specific number reproduces):

```bash
# scan the liquid universe:
python -m sentinel.carry.walkforward --scan --train 600 --test 120 --top-n 6 -v

# or a fixed set:
python -m sentinel.carry.walkforward --symbols XMR_USDT,EVAA_USDT,VELVET_USDT --train 600 --test 120
```

The no-look-ahead property is **unit-tested** (`tests/test_carry_walkforward.py`):
a synthetic name that only turns positive *in the test window* is never selected,
and the bootstrap CI is deterministic for a fixed seed. This is the difference
between "it backtested well" and "it survived a procedure designed to break it."

### 5.6 Capacity — a curve, not a number

"Capacity ~$100k" is shorthand; the honest object is a **curve**.
`sentinel/carry/capacity.py` builds net-yield-vs-notional for a name from its
measured gross carry and current spot volume, using a standard *square-root*
market-impact model (slippage ≈ k·√(notional / ADV)) plus a participation cap.
It reports the notional where the edge stops being worth it — binding on
whichever trips first, the **yield floor** or the **volume footprint** — plus a
deterministic sensitivity band over the impact coefficient.

```bash
python -m sentinel.carry.capacity --symbol XMR_USDT -v
```

The model is assumption-driven (it has no order-book depth — only daily volume),
so it's a sanity check, not a guarantee. And it's an honest one: under a
conservative impact coefficient it can put practical capacity *well below* $100k
for a thin name. That's the point — capacity is **per-name, liquidity-dependent,
and smaller than the headline yield tempts you to believe**. The deepest limit
(your own size compressing the funding you harvest) needs open-interest /
elasticity data this tool doesn't yet collect, and is called out as out of scope.

### 5.7 Is the screen just noise-mining? A permutation test

Scanning ~150–200 perps and keeping the most-consistent handful is a
**multiple-testing** problem: with enough names, some look "consistent" by pure
chance. Walk-forward already guards against this (chance-consistency doesn't
persist into the next window), but `walkforward.permutation_test` makes it
explicit. It compares the real consistency-selected OOS yield against a **null
distribution** built from many *random* basket selections over the same universe
and windows, and returns an empirical p-value `P(random ≥ real)`.

```bash
python -m sentinel.carry.walkforward --scan --train 600 --test 120 --permutations 1000 -v
```

On synthetic data it does the honest thing both ways: with a genuine edge it
returns p ≈ 0.003 (the screen clearly beats chance); when every name is
identical (no real selection advantage) it returns **p ≈ 1.0 and refuses to
reject** — exactly what a test you can trust must do. Scope: this isolates the
value of the *selection rule* vs. chance from the same universe; it does **not**
correct for survivorship in the universe itself (only currently-listed names),
which would need point-in-time listing data the scanner doesn't collect.

---

## 6. Honest limitations

Most of these came from deliberately red-teaming the strategy (see §5.5/§5.7 for
the parts that *are* defended in code; the rest are disclosed here because they
need live data or operational discipline rather than a backtest tweak).

- **Paper, not live-proven.** Everything here is backtest + paper simulation.
  Simulated fills are optimistic by nature. A real edge needs a long *live*
  track record, which this does not yet have.
- **Margin / liquidation mechanics are not simulated.** The backtest treats the
  short perp as a clean hedge and never models maintenance margin, liquidation,
  or auto-deleveraging (ADL). A short perp is liquidated by the price *rallying*
  (which the long spot leg offsets) — **not** by negative funding, which is only
  a cash bleed. So this is *not* an inherent edge-killer, but it is a real gap:
  the mitigation is operational — run the perp leg **unleveraged / cross-margined
  / same-venue** so a violent move can't liquidate it before the spot gain is
  realised. The carry does not need leverage; using it reintroduces ruin risk.
- **Capacity-limited.** ~$100k order-of-magnitude, per name (§5.6). Adding
  capital or crowding the trade *reduces* the edge. It does not scale. The
  capacity model also ignores **funding compression from your own size** (your
  short pushes the funding you harvest toward zero) — the deepest limit, and one
  it can't yet measure without open-interest data.
- **Breadth may fail in a crisis.** The measured ~0.03 average pairwise funding
  correlation is likely a calm-regime figure. In a market-wide deleveraging,
  funding across alts can move together (correlations → 1), so the diversification
  benefit shrinks exactly when you need it. Treat the breadth claim as a
  benign-regime estimate, not a guarantee.
- **Single-regime survivorship.** ~400 days ≈ one regime. "100% consistent" is
  history, not a structural law — except where there's an independent structural
  reason (e.g. XMR's short constraint), which is why the *reason* matters more
  than the *rate*.
- **Cost realism.** The sim uses flat fees and a static slippage assumption.
  Real tiered maker/taker fees, settlement-timing edge cases, short-size/position
  caps on thin perps, and spot-perp basis widening in stress all erode returns
  and aren't fully captured.
- **Execution risk on thin alts.** Halts, delistings, and gaps are tail risks
  paper trading doesn't capture.

## 7. The transferable lessons

1. **Try to disprove your own strategy.** Assume no edge until the data forces
   you to concede one. Most "edges" don't survive this.
2. **Model real costs — especially slippage.** A mid-price backtest with no
   slippage is fiction. Costs turned a +0.035R signal into −0.243R.
3. **Out-of-sample or it didn't happen.** In-sample pairs results inverted OOS.
   The optimizer will always find a pattern in noise.
4. **A confident narrative is not an edge.** The LLM produced beautiful,
   reasoned trade plans and still lost net of costs.
5. **The surviving edge is usually modest, structural, and market-neutral** —
   and it's small precisely because it can't be scaled.
6. **At small capital, savings rate beats return.** 15% of a small stake is a
   small number. The strategy is a good home for capital you already have, not a
   way to manufacture it.

---

*The most valuable output of this whole project wasn't a money printer. It was a
repeatable method for telling the difference between a real edge and a
comfortable story — and the discipline to keep the worthless 90% honest.*
