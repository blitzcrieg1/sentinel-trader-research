"""
Backtest CLI.

Run::

    python -m sentinel.backtest
    python -m sentinel.backtest --data-dir data/historical --symbols BTC/USDT,ETH/USDT
    python -m sentinel.backtest --rr 2.0 --sl-atr 1.5 --no-macd-confirm
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sentinel.backtest.engine import BacktestConfig, run_backtest
from sentinel.backtest.loader import available_symbols
from sentinel.backtest.metrics import compute_metrics
from sentinel.backtest.strategy import StrategyParams


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sentinel deterministic-proxy backtest")
    parser.add_argument("--data-dir", default="data/historical", help="Historical OHLCV directory")
    parser.add_argument("--symbols", default="", help="Comma-separated; default = all in manifest")
    parser.add_argument("--rr", type=float, default=None, help="Reward:risk ratio (default = min_rr_ratio)")
    parser.add_argument("--sl-atr", type=float, default=1.0, help="SL distance as × ATR(14)")
    parser.add_argument("--max-hold", type=int, default=96, help="Max bars to hold (15m each)")
    parser.add_argument("--no-macd-confirm", action="store_true", help="Drop the 1h MACD confirmation filter")
    parser.add_argument("--per-symbol", action="store_true", help="Also print a per-symbol breakdown")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s — %(message)s",
    )

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"error: data dir not found: {data_dir}", file=sys.stderr)
        return 2

    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else available_symbols(data_dir)
    )
    if not symbols:
        print("error: no symbols found", file=sys.stderr)
        return 2

    from sentinel.config import get_settings
    settings = get_settings()
    cfg = BacktestConfig.from_settings(settings)
    if args.rr is not None:
        cfg = BacktestConfig(
            rr_ratio=args.rr, sl_atr_mult=args.sl_atr, min_sl_pct=cfg.min_sl_pct,
            max_sl_pct=cfg.max_sl_pct, max_hold_bars=args.max_hold, fee_pct=cfg.fee_pct,
            slippage_pct=cfg.slippage_pct, risk_per_trade_pct=cfg.risk_per_trade_pct,
        )
    else:
        cfg = BacktestConfig(
            rr_ratio=cfg.rr_ratio, sl_atr_mult=args.sl_atr, min_sl_pct=cfg.min_sl_pct,
            max_sl_pct=cfg.max_sl_pct, max_hold_bars=args.max_hold, fee_pct=cfg.fee_pct,
            slippage_pct=cfg.slippage_pct, risk_per_trade_pct=cfg.risk_per_trade_pct,
        )

    params = StrategyParams(require_macd_confirm=not args.no_macd_confirm)

    print(f"Backtesting {len(symbols)} symbol(s): {', '.join(symbols)}")
    print(
        f"Config: RR={cfg.rr_ratio}  SL={cfg.sl_atr_mult}xATR  "
        f"max_hold={cfg.max_hold_bars}bars  fee={cfg.fee_pct}%  "
        f"slip={cfg.slippage_pct}%  macd_confirm={params.require_macd_confirm}"
    )
    print()

    trades = run_backtest(data_dir, symbols, cfg, params, settings)
    report = compute_metrics(trades, cfg.risk_per_trade_pct)
    print(report.format_text())

    if args.per_symbol:
        print("\nBy symbol:")
        by_sym: dict[str, list] = {}
        for t in trades:
            by_sym.setdefault(t.symbol, []).append(t)
        for sym in symbols:
            sym_trades = by_sym.get(sym, [])
            if not sym_trades:
                print(f"  {sym:<14} no trades")
                continue
            r = compute_metrics(sym_trades, cfg.risk_per_trade_pct)
            print(
                f"  {sym:<14} n={r.n_trades:<4} win={r.win_rate * 100:4.0f}%  "
                f"exp={r.expectancy_r:+.3f}R  PF={r.profit_factor:.2f}  "
                f"ret={r.total_return_pct:+.1f}%"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
