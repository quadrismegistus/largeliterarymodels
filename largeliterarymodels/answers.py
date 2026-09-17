"""Typed answers — what a Survey gets back, and how not to throw it away.

The whole reason to administer typed questions rather than a schema prompt
is that the answers are CONTINUOUS: a Noul is a probability, a Score is a
probability-weighted position that can land between levels, a Choice carries
a distribution over its options. Measured on a literary instrument
(2026-09-17, 583 passages, agreement with a Sonnet reference): thresholding
each answer at 0.5 and averaging booleans gave a half-century contrast of
-0.111 with a 95% CI of [-0.276, +0.050]; averaging the probability itself
gave -0.129, CI [-0.179, -0.081] — the same 583 passages, a CI more than
three times tighter, and a discontinuity that the booleans could not
resolve at all. The quantization was the whole loss.

So this module has no `bool` property and no implicit threshold. Getting a
hard label requires naming a cut point, because on the same corpus the
right cut varied from 0.24 to 0.59 across fields while AUC stayed above
0.86 throughout — the model's ordering was good and its absolute placement
was not, and a silent 0.5 would have read as an accuracy failure rather
than a calibration one.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping


class AnswerError(ValueError):
    """The response did not answer what was asked."""


@dataclass(frozen=True)
class Noul:
    """P(the question is true). One number; there is no separate confidence.

    A value near 0.5 means yes and no are near-equally likely. It does NOT
    mean "medium" — a degree wants a Score.
    """
    qid: str
    p: float

    def at(self, threshold):
        """Hard label at an explicit cut. No default: see the module docstring."""
        return self.p >= threshold

    @property
    def value(self):
        return self.p


@dataclass(frozen=True)
class Choice:
    """The selected option, with the distribution that produced it.

    `confidence` summarises how concentrated the distribution is. It is not
    a probability that the workflow is correct and not permission to act,
    and several acceptable alternatives can spread it without anything being
    wrong.
    """
    qid: str
    choice: str
    confidence: float | None = None
    probabilities: Mapping[str, float] = field(default_factory=dict)

    @property
    def value(self):
        return self.choice

    def p(self, option):
        """Probability mass on one option, 0.0 if the model gave it none."""
        return float(self.probabilities.get(option, 0.0))

    def margin(self):
        """Top probability minus runner-up — how close the call was.

        Distinct from `confidence`: a three-way split and a two-way tie can
        share a confidence and differ entirely in whether the second option
        was live.
        """
        ps = sorted(self.probabilities.values(), reverse=True)
        if len(ps) < 2:
            return None
        return ps[0] - ps[1]


@dataclass(frozen=True)
class Score:
    """A position on the ordered levels, which may fall between two of them.

    `score` is in level positions, counting from 0, so an n-level question
    runs 0..n-1. `normalized` puts it on 0..1, which is what makes two Score
    questions with different level counts — or a Score and a Noul —
    comparable in the same table.
    """
    qid: str
    score: float
    confidence: float | None = None
    legend: Mapping[str, str] = field(default_factory=dict)
    probabilities: Mapping[str, float] = field(default_factory=dict)

    @property
    def value(self):
        return self.score

    @property
    def n_levels(self):
        return len(self.legend) or len(self.probabilities)

    @property
    def normalized(self):
        n = self.n_levels
        return None if n < 2 else self.score / (n - 1)

    def level(self):
        """Nearest whole level.

        Provided, and deliberately not the default accessor: mean Score
        confidence measured 0.40-0.51 on a real instrument, meaning the
        distribution over levels was diffuse even where the ordering was
        sound. Rounding to a label throws that away exactly as a 0.5
        threshold does to a Noul.
        """
        return int(round(self.score))


@dataclass(frozen=True)
class Unknown:
    """An answer kind this version does not model, kept whole.

    The SDK logs a warning and skips these. Skipping is how a provider's new
    field goes unnoticed until someone needs it and finds it was discarded —
    every provider-drift bug in this package was diagnosed from a response
    that had already been thrown away. So it is carried instead, and
    `Answers.unknown` says so out loud.
    """
    qid: str
    body: Mapping[str, Any]

    @property
    def value(self):
        return None


_PARSERS = {}


def _parser(kind):
    def register(fn):
        _PARSERS[kind] = fn
        return fn
    return register


@_parser("noul")
def _parse_noul(qid, a):
    return Noul(qid=qid, p=float(a["noul"]))


@_parser("choice")
def _parse_choice(qid, a):
    return Choice(qid=qid, choice=a["choice"],
                  confidence=a.get("confidence"),
                  probabilities=dict(a.get("probabilities") or {}))


@_parser("score")
def _parse_score(qid, a):
    return Score(qid=qid, score=float(a["score"]),
                 confidence=a.get("confidence"),
                 legend=dict(a.get("legend") or {}),
                 probabilities=dict(a.get("probabilities") or {}))


class Answers(Mapping):
    """The answers to one administration, addressed by question id.

    A Mapping, so `answers["field"]` and iteration work; the typed accessors
    below are for the common case of wanting a number rather than an object.
    """

    def __init__(self, answers, asked=None, model=None, usage=None):
        self._a = dict(answers)
        self._asked = tuple(asked or answers)
        self.model = model
        self.usage = dict(usage or {})

    def __getitem__(self, qid):
        try:
            return self._a[qid]
        except KeyError:
            raise AnswerError(
                f"no answer for {qid!r}. Asked: {len(self._asked)} questions, "
                f"answered: {len(self._a)}."
            ) from None

    def __iter__(self):
        return iter(self._a)

    def __len__(self):
        return len(self._a)

    @property
    def missing(self):
        """Ids asked that came back with no answer.

        Empty on every call observed so far — the response is keyed by the
        ids we sent, which is the structural reason a Survey cannot drop,
        invent or re-case an item the way a list-valued schema field can.
        Checked anyway, because "cannot happen" is a claim with a shelf life.
        """
        return tuple(q for q in self._asked if q not in self._a)

    @property
    def unanswered(self):
        return bool(self.missing)

    @property
    def unknown(self):
        """Ids whose answer kind this version does not model."""
        return tuple(q for q, a in self._a.items() if isinstance(a, Unknown))

    def value(self, qid):
        """The answer's natural value: probability, label, or position."""
        return self[qid].value

    def p(self, qid):
        """P(true) for a Noul. Raises for anything else, rather than
        coercing a Choice's confidence or a Score's position into a number
        that would silently mean something different."""
        a = self[qid]
        if not isinstance(a, Noul):
            raise AnswerError(
                f"{qid!r} is a {type(a).__name__}, not a Noul; .p() is the "
                f"yes-probability accessor. Use .value() or the typed object."
            )
        return a.p

    def values(self):
        return {q: a.value for q, a in self._a.items()}

    def confidences(self):
        return {q: getattr(a, "confidence", None) for q, a in self._a.items()}


def parse(body, asked=None):
    """Raw System One response body -> Answers.

    `asked` is the question-id sequence administered; supplying it is what
    makes `.missing` meaningful. Unknown answer kinds are kept rather than
    skipped.
    """
    if not isinstance(body, Mapping):
        raise AnswerError(f"response body is a {type(body).__name__}, not a "
                          f"mapping — nothing to parse.")
    raw = body.get("answers")
    if raw is None:
        raise AnswerError(
            "response has no `answers` field. Body keys: "
            f"{sorted(body)}. This is provider drift, not a bad question — "
            "check the raw sidecar."
        )
    out = {}
    for qid, a in raw.items():
        kind = (a or {}).get("type")
        fn = _PARSERS.get(kind)
        if fn is None:
            out[qid] = Unknown(qid=qid, body=dict(a or {}))
            continue
        try:
            out[qid] = fn(qid, a)
        except (KeyError, TypeError, ValueError) as e:
            raise AnswerError(
                f"answer {qid!r} claims type {kind!r} but does not carry its "
                f"fields ({e}). Raw: {a!r}"
            ) from e
    return Answers(out, asked=asked, model=body.get("model"),
                   usage=body.get("usage"))
