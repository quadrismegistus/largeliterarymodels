#!/usr/bin/env python3
"""price_run.py — thin shim over `largeliterarymodels.costs` / `litmod price`.

Originally a standalone implementation adopted from malign-logits; it is now
a delegation layer, because a second implementation of the same table had
already diverged from the module within one PR (no dated-row logic, no
cache-floor gate, reasoning flags from the table instead of measured
behaviour). One implementation, two entry points.

  python scripts/price_run.py --fresh 517547 --cached 18389760 --output 657056
  python scripts/price_run.py --selftest      # the $1.8511-vs-$1.86 invoice gate

`litmod price` is the first-class interface; this script survives so callers
pointed at the old path keep working.
"""
import argparse
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

from largeliterarymodels import costs  # noqa: E402
from largeliterarymodels.cli.main import build_parser  # noqa: E402


def selftest():
    est = costs.price("gpt-4o-mini", fresh=517_547, cached=18_389_760,
                      output=657_056)
    ok = abs(est["usd"] - 1.8511) < 0.0005
    print("SELFTEST  Registration P gpt-4o-mini arm")
    print("  predicted $%.4f   billed $1.86   known-good $1.8511   %s"
          % (est["usd"], "PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv[1:]:
        return selftest()
    # Everything else is exactly `litmod price`.
    args = build_parser().parse_args(["price"] + sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
