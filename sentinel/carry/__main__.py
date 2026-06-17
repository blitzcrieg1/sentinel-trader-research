"""
Funding-carry scanner CLI.

    python -m sentinel.carry                 # scan, show curated basket
    python -m sentinel.carry --min-vol 5e6 --top 8
    python -m sentinel.carry --all           # show full ranked table, not just basket
"""

from __future__ import annotations

import argparse
import sys

from sentinel.carry.scanner import carry_score, curate_basket, scan


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rank MEXC perps by harvestable funding carry")
    p.add_argument("--min-vol", type=float, default=3_000_000.0, help="Min 24h quote volume (USDT)")
    p.add_argument("--min-consistency", type=float, default=85.0, help="Min funding consistency %")
    p.add_argument("--min-carry", type=float, default=8.0, help="Min annualised carry %")
    p.add_argument("--min-spot-vol", type=float, default=2_000_000.0,
                   help="Min 24h spot volume (USDT) for the hedge leg; 0 disables")
    p.add_argument("--top", type=int, default=6, help="Basket size")
    p.add_argument("--all", action="store_true", help="Show full ranked table")
    args = p.parse_args(argv)

    print(f"Scanning MEXC perps (>{args.min_vol/1e6:.0f}M/24h vol)…", file=sys.stderr)
    stats = scan(min_vol_usdt=args.min_vol)
    if not stats:
        print("No symbols with sufficient funding history.", file=sys.stderr)
        return 1

    spot_floor = args.min_spot_vol if args.min_spot_vol > 0 else None
    rows = stats if args.all else curate_basket(
        stats, min_consistency=args.min_consistency, min_carry_yr=args.min_carry,
        min_spot_vol_usdt=spot_floor, top_n=args.top,
    )
    title = "FULL RANKING" if args.all else "CURATED CARRY BASKET"
    print(f"\n=== {title} ===")
    print(f"{'symbol':18}{'n':>5}{'mean/8h%':>10}{'%pos':>6}{'carry/yr%':>11}"
          f"{'consist%':>10}{'spotVol$M':>11}{'score':>8}")
    print("-" * 80)
    for s in rows:
        sv = f"{s.spot_vol_usdt/1e6:9.1f}" if s.spot_vol_usdt is not None else "      n/a"
        print(f"{s.symbol:18}{s.n:5}{s.mean_8h*100:10.4f}{s.pct_positive:6.0f}"
              f"{s.static_carry_yr:11.1f}{s.consistency:10.0f}{sv}{carry_score(s):8.1f}")
    if not args.all:
        print(f"\n{len(rows)} names selected for long-spot / short-perp static carry "
              f"(consistency ≥ {args.min_consistency:.0f}%, carry ≥ {args.min_carry:.0f}%/yr, "
              f"spot ≥ ${spot_floor/1e6:.0f}M/24h).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
