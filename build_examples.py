"""Attach spans to Propp's OWN function sequences, for use as few-shot examples.

The examples must teach Propp's judgement, not the annotator's. So the
function sequence is fixed — verbatim from his printed scheme — and the only
thing inferred is WHERE in the text each function sits. That is a
constrained inference, and it is checkable: every span is verified against
the source, and one that cannot be located is dropped rather than kept.
"""
import json, os, re, sys
from pydantic import BaseModel, Field

sys.path.insert(0, "data/propp")
from propp_eval import parse_scheme
from largeliterarymodels.llm import LLM
from largeliterarymodels.tasks.classify_propp import (
    PROPP_FUNCTIONS, prepare_tale_text)

NAME_BY_SYMBOL = {f['symbol']: f['name'] for f in PROPP_FUNCTIONS}
TITLE_BY_SYMBOL = {f['symbol']: f['title'] for f in PROPP_FUNCTIONS}
DESC_BY_SYMBOL = {f['symbol']: f['description'] for f in PROPP_FUNCTIONS}


unwrap = prepare_tale_text


class LocatedFunction(BaseModel):
    ordinal: int = Field(description="1-based position in the given sequence.")
    symbol: str = Field(description="The symbol from the given sequence, copied exactly.")
    span: str = Field(description="The shortest VERBATIM quote from the tale that "
                                  "instantiates this function, copied character for "
                                  "character. Never paraphrase.")
    note: str = Field(description="One short clause on why this span, or empty.")


def locate_for_tale(llm, text, moves):
    seq = []
    for mi, mv in enumerate(moves, start=1):
        for sym in mv:
            seq.append((mi, sym))
    listing = "\n".join(
        f"{i}. move {mi}  {sym}  ({TITLE_BY_SYMBOL.get(sym, '?')}): "
        f"{DESC_BY_SYMBOL.get(sym, '')[:150]}"
        for i, (mi, sym) in enumerate(seq, start=1))
    system = (
        "Vladimir Propp analysed the tale below and recorded the exact "
        "sequence of plot functions it contains. That sequence is GIVEN and "
        "is not to be revised: do not add functions, drop functions, or "
        "reorder them. Your only job is to find, for each function in the "
        "list, the shortest verbatim stretch of the tale that instantiates "
        "it.\n\n"
        "Quote character for character. Do not paraphrase, modernise "
        "spelling, or normalise punctuation — every span is checked against "
        "the source.\n\n"
        "Return exactly one entry per numbered item, in the same order.\n\n"
        "PROPP'S SEQUENCE FOR THIS TALE:\n" + listing)
    out = llm.extract(text, list[LocatedFunction], system_prompt=system,
                      max_tokens=8192)
    return seq, out


def verify(text, seq, located):
    """Keep spans locatable in the text, and canonicalise them to the source.

    Matching mirrors locate_spans: exact first, then whitespace-normalised —
    a span crossing a paragraph break is quoted by the model with a single
    space and is a real quotation, not a paraphrase. Whatever matches, the
    span STORED is the actual source substring, so the example carries a
    genuinely verbatim quote rather than the model's rendering of one.
    """
    from largeliterarymodels.tasks.classify_propp import _normalise_ws
    norm, index_map = _normalise_ws(text)
    kept, dropped = [], []
    for (mi, sym), item in zip(seq, located):
        span = (item.span or "").strip()
        if not span:
            dropped.append((mi, sym, "<empty>")); continue
        start = text.find(span)
        if start != -1:
            end = start + len(span)
        else:
            span_norm, _ = _normalise_ws(span)
            j = norm.find(span_norm)
            if j == -1:
                dropped.append((mi, sym, span[:60])); continue
            start = index_map[j]
            end = index_map[min(j + len(span_norm) - 1, len(index_map) - 1)] + 1
        kept.append({"move": mi, "symbol": sym,
                     "function": NAME_BY_SYMBOL[sym],
                     "span": text[start:end],
                     "note": (item.note or "").strip()})
    return kept, dropped


def main(numbers):
    d = json.load(open("data/propp/propp_tales.json"))
    by_num = {str(t["afanasyev_number"]): t for t in d["tales"]}
    llm = LLM("deepseek/deepseek-v4-pro", temperature=0.0)
    out = {}
    for num in numbers:
        t = by_num[num]
        path = os.path.join("data/propp/tale_texts",
                            os.path.basename(t["text_available"]["local_file"]))
        text = unwrap(open(path, encoding="utf-8").read())
        moves, flat, _ = parse_scheme(t["propp_scheme"])
        seq, located = locate_for_tale(llm, text, moves)
        kept, dropped = verify(text, seq, located)
        print(f"#{num} {t['english_title']}: {len(flat)} functions, "
              f"{len(kept)} located, {len(dropped)} dropped")
        for mi, sym, sp in dropped:
            print(f"    DROPPED move {mi} {sym}: {sp!r}")
        out[num] = {"afanasyev_number": num,
                    "english_title": t["english_title"],
                    "propp_scheme": t["propp_scheme"],
                    "n_moves": len(moves), "text": text, "functions": kept}
    json.dump(out, open("data/propp/examples_located.json", "w"),
              ensure_ascii=False, indent=1)
    print("\nwrote data/propp/examples_located.json")


if __name__ == "__main__":
    main(sys.argv[1:])
