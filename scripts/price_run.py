#!/usr/bin/env python3
"""price_run.py — price a measured LLM workload against any model, any provider.

Reads `data/model_pricing.json` (multi-provider; see its `_what`). Written to be
moved into `largeliterarymodels` as-is: it imports nothing from this repo, and
`PRICING_FILE` is the only path it knows.

WHY IT EXISTS. `largeliterarymodels/costs.py` reads `anthropic_pricing.json`,
which is Anthropic-only and was last updated 2026-04-27 — it tops out at
claude-sonnet-4-6 and can price none of Registration P's three arms.

  # price ONE measured arm across every model
  python scripts/price_run.py --fresh 517547 --cached 18389760 --output 657056

  # ... within one provider, and add batch pricing
  python scripts/price_run.py --fresh 213784 --cached 30682380 \\
      --output 1220318 --provider anthropic --batch

  # what one specific model would cost
  python scripts/price_run.py --fresh 1000 --cached 0 --output 500 \\
      --model claude-sonnet-5

VALIDATION. Registration P's gpt-4o-mini arm: predicted $1.8511 from
(517,547 fresh / 18,389,760 cached / 657,056 output); billed $1.86. 0.05%.
`--selftest` re-runs that check.

**A PRICE IS NOT A QUOTE.** Pricing a workload measured on model A against
model B is a counterfactual: output volume and tokenization both differ, and
across Anthropic's 4.7 tokenizer boundary the same text is ~30% more tokens.
Reasoning models bill hidden reasoning tokens as output, so a non-reasoning
workload priced against one is a FLOOR. Both are flagged in the output.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PRICING_FILE = os.path.join(os.path.dirname(HERE), "largeliterarymodels",
                            "model_pricing.json")
M = 1_000_000


def load(path=PRICING_FILE):
    with open(path) as fh:
        return json.load(fh)


def models(p, provider=None):
    """-> [(provider, name, rates)] over the requested providers."""
    out = []
    for prov in ("anthropic", "openai", "deepseek", "google"):
        if provider and prov != provider:
            continue
        for name, r in p.get(prov, {}).items():
            out.append((prov, name, r))
    return out


def resolve(p, name):
    name = p.get("aliases", {}).get(name, name)
    for prov in ("anthropic", "openai", "deepseek", "google"):
        if name in p.get(prov, {}):
            return prov, name, p[prov][name]
    raise SystemExit("unknown model: %s" % name)


def price(rates, fresh, cached, output, cache_write=0, ttl="5m", batch=0.0):
    """USD for one workload. `cached is None` in the table means NO cache tier,
    so cached tokens bill at the full input rate — the single largest driver of
    cost on -pro models, and easy to miss."""
    cr = rates["cached"]
    if cr is None:
        cr = rates["input"]
    wr = rates.get("cache_write_1h" if ttl == "1h" else "cache_write_5m")
    if wr is None:
        wr = rates["input"]
    disc = 1.0 - batch
    return (fresh * rates["input"] * disc
            + cached * cr * disc
            + cache_write * wr * disc
            + output * rates["output"] * disc) / M


def selftest(p):
    got = price(resolve(p, "gpt-4o-mini")[2], 517_547, 18_389_760, 657_056)
    ok = abs(got - 1.8511) < 0.0005
    print("SELFTEST  Registration P gpt-4o-mini arm")
    print("  predicted $%.4f   billed $1.86   known-good $1.8511   %s"
          % (got, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", type=int, default=0, help="uncached input tokens")
    ap.add_argument("--cached", type=int, default=0, help="cache-read input tokens")
    ap.add_argument("--output", type=int, default=0)
    ap.add_argument("--cache-write", type=int, default=0)
    ap.add_argument("--ttl", choices=("5m", "1h"), default="5m")
    ap.add_argument("--provider", choices=("anthropic", "openai", "deepseek", "google"))
    ap.add_argument("--model", help="price one model instead of the table")
    ap.add_argument("--batch", action="store_true", help="apply the batch discount")
    ap.add_argument("--times", type=int, default=1, help="multiply (e.g. 3 coder arms)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    p = load()
    if a.selftest:
        return selftest(p)

    disc = p["batch_discount"] if a.batch else {}
    n = a.times

    if a.model:
        prov, name, r = resolve(p, a.model)
        b = (disc.get(prov) or 0.0) if a.batch else 0.0
        c = price(r, a.fresh, a.cached, a.output, a.cache_write, a.ttl, b)
        print("%s / %s%s: $%.4f%s"
              % (prov, name, "  [batch]" if b else "", c * n,
                 "  x%d = $%.4f" % (n, c * n) if n > 1 else ""))
        return 0

    rows = []
    for prov, name, r in models(p, a.provider):
        b = (disc.get(prov) or 0.0) if a.batch else 0.0
        rows.append((price(r, a.fresh, a.cached, a.output, a.cache_write,
                           a.ttl, b) * n, prov, name, r))
    rows.sort()

    print("workload: %s fresh + %s cached input, %s output%s%s"
          % (f"{a.fresh:,}", f"{a.cached:,}", f"{a.output:,}",
             ", %s cache-write" % f"{a.cache_write:,}" if a.cache_write else "",
             "  x%d" % n if n > 1 else ""))
    if a.batch:
        print("BATCH pricing applied where the provider offers it.")
    print()
    print("  %-11s %-30s %9s %9s %9s %11s"
          % ("provider", "model", "input", "cached", "output", "COST"))
    print("  " + "-" * 84)
    for c, prov, name, r in rows:
        cache = "%9.3f" % r["cached"] if r["cached"] is not None else "     none"
        print("  %-11s %-30s %9.2f %s %9.2f %11.2f%s"
              % (prov, name, r["input"], cache, r["output"], c,
                 " *" if r.get("reasoning") else ""))
    print()
    print("  * REASONING model — hidden reasoning tokens bill as output, so a")
    print("    non-reasoning workload priced here is a FLOOR, not an estimate.")
    print("  `none` in the cached column = NO cache tier; those tokens bill at")
    print("    the full input rate, which dominates cost on high-cache workloads.")
    for k, v in p["caveats"].items():
        print("  [%s] %s" % (k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
