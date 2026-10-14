"""Typed questions — the instrument a Survey administers.

A `Task`'s instrument is the system prompt: a string, built from the schema,
and the thing its cache key records as administered bytes. A Survey has no
prompt. Its instrument is a dict of typed questions, and this module owns
that dict's shape, its validation, and its canonical form.

## WHY OUR OWN TYPES RATHER THAN THE SDK'S

`typesafe_sdk` ships `Noul`, `Choice` and `Score`, and passing those would
be less code. But the canonical form of the question set is what the cache
key hashes, so an SDK version that changed a default, reordered a field or
renamed a key would silently re-key an entire corpus — a different
serialiser is not a different instrument, and the key must not say it is.
Owning the types means the administered bytes are ours to keep stable.

It costs nothing at the wire, because the API accepts the plain dict
spelling directly (the SDK's own "raw question dictionaries" escape hatch).
So `to_wire()` output IS what is sent, and `canonical()` IS what was
administered. Byte-identity between the two is the same property
`Task.instrument_text()` has, obtained the same way: by not re-rendering.

## THE FOUR SHAPES A FIELD CAN TAKE

`instructions`, each Choice option, each Score level, and a Noul's
`criteria.true`/`.false` all accept a string, an object, or an array. A
string is the place to start. An object is for when the boundary needs
naming rather than describing — the cookbook convention is
`{what, not_for, examples}` for criteria and `{question, inspect, focus}`
for instructions, where `inspect` names the state field to read and `focus`
carries the discriminating rule. Nothing here enforces those key names: they
are a convention that works, not a schema, and inventing a schema for them
would freeze someone else's prompt engineering into our types.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

#: A value acceptable as instructions or as a criterion: string, object or
#: array, per the API's structured-question support.
Spec = Any

_MAX_SCORE_LEVELS = 10
_MIN_SCORE_LEVELS = 2


class QuestionError(ValueError):
    """A question set that the API would reject, or that would mislead.

    Raised at construction rather than at call time: a malformed question in
    a 40-question batch fails the whole call, and finding out after paying
    for the other 39 is the avoidable half.
    """


def _check_spec(value, what):
    if value is None or value == "" or value == {} or value == []:
        raise QuestionError(f"{what} is empty — say what is being asked.")
    try:
        json.dumps(value)
    except (TypeError, ValueError) as e:
        raise QuestionError(
            f"{what} is not JSON-serialisable ({e}). Instructions and criteria "
            f"are sent as JSON; a value that cannot be expressed cannot be "
            f"administered."
        ) from e
    return value


@dataclass(frozen=True)
class Noul:
    """Probability that a yes/no question is true.

    The answer is ONE number and there is no separate confidence: 0.5 means
    yes and no are equally likely, not "medium". A field that wants a degree
    wants a `Score`.

    `criteria` is optional and takes `{"true": ..., "false": ...}`. Measured
    on a literary instrument (2026-09-17, 583 passages): adding criteria with
    worked examples moved the operating point without moving AUC — it places
    the threshold, it does not improve the ranking. Worth adding where a
    boolean is wanted at 0.5; not a substitute for fitting a threshold.
    """
    instructions: Spec
    criteria: Mapping[str, Spec] | None = None

    def __post_init__(self):
        _check_spec(self.instructions, "Noul instructions")
        if self.criteria is not None:
            extra = set(self.criteria) - {"true", "false"}
            if extra:
                raise QuestionError(
                    f"Noul criteria takes 'true' and 'false' only; got "
                    f"{sorted(extra)}. Per-option criteria is a Choice."
                )
            for k in ("true", "false"):
                if k in self.criteria:
                    _check_spec(self.criteria[k], f"Noul criteria.{k}")

    def to_wire(self):
        w = {"type": "noul", "instructions": self.instructions}
        if self.criteria is not None:
            w["criteria"] = dict(self.criteria)
        return w


@dataclass(frozen=True)
class Choice:
    """One option from a fixed set, with a probability for each.

    Option ORDER is preserved as given. Unordered in meaning, but the wire
    form is what the key hashes, so reordering the options is a different
    administration and is recorded as one rather than being normalised away.
    """
    instructions: Spec
    criteria: Mapping[str, Spec]

    def __post_init__(self):
        _check_spec(self.instructions, "Choice instructions")
        if not isinstance(self.criteria, Mapping):
            raise QuestionError(
                "Choice criteria maps option name -> description (or None). "
                "An ordered list of levels is a Score."
            )
        if len(self.criteria) < 2:
            raise QuestionError(
                f"Choice needs at least 2 options; got {len(self.criteria)}. "
                f"A one-option choice has no answer to give."
            )
        for k, v in self.criteria.items():
            if not isinstance(k, str) or not k:
                raise QuestionError(f"Choice option name must be a non-empty "
                                    f"string; got {k!r}.")
            if v is not None:
                _check_spec(v, f"Choice criteria[{k!r}]")

    def to_wire(self):
        return {"type": "choice", "instructions": self.instructions,
                "criteria": dict(self.criteria)}


@dataclass(frozen=True)
class Score:
    """A position on ordered levels, probability-weighted.

    The answer can fall BETWEEN levels, which is the point: it is a
    continuous measure with no threshold anywhere. Level i is position i,
    counting from 0, and the array's order is the numbering — so reordering
    the levels changes every answer's meaning, and the wire form keeps that
    order exactly as given.

    Levels must describe concrete situations and stand on their own; the
    model sees the descriptions and nothing else, and judges each on its own
    against the state.
    """
    instructions: Spec
    criteria: Sequence[Spec]

    def __post_init__(self):
        _check_spec(self.instructions, "Score instructions")
        if isinstance(self.criteria, (str, bytes)) or isinstance(
                self.criteria, Mapping):
            raise QuestionError(
                "Score criteria is an ORDERED array of level descriptions, "
                "low end first. A name -> description mapping is a Choice."
            )
        levels = list(self.criteria)
        if not _MIN_SCORE_LEVELS <= len(levels) <= _MAX_SCORE_LEVELS:
            raise QuestionError(
                f"Score takes {_MIN_SCORE_LEVELS}-{_MAX_SCORE_LEVELS} levels; "
                f"got {len(levels)}."
            )
        for i, lv in enumerate(levels):
            _check_spec(lv, f"Score criteria[{i}]")
        object.__setattr__(self, "criteria", tuple(levels))

    @property
    def n_levels(self):
        return len(self.criteria)

    def to_wire(self):
        return {"type": "score", "instructions": self.instructions,
                "criteria": list(self.criteria)}


QUESTION_TYPES = (Noul, Choice, Score)


def to_wire(questions):
    """{id: Question} -> {id: plain dict}, in the order given.

    Question ids are not sent to the model — they are how code addresses the
    answer — so the whole meaning of a question has to be inside it.
    """
    out = {}
    for qid, q in questions.items():
        if not isinstance(qid, str) or not qid:
            raise QuestionError(f"question id must be a non-empty string; "
                                f"got {qid!r}.")
        if isinstance(q, Mapping):
            # The raw-dict escape hatch, for a request field these types
            # predate. Passed through untouched and still hashed, so it is
            # part of the administration like anything else.
            out[qid] = _check_spec(dict(q), f"question {qid!r}")
            continue
        if not isinstance(q, QUESTION_TYPES):
            raise QuestionError(
                f"question {qid!r} is a {type(q).__name__}; expected Noul, "
                f"Choice, Score, or a raw dict."
            )
        out[qid] = q.to_wire()
    if not out:
        raise QuestionError("a Survey administers at least one question.")
    return out


def canonical(questions):
    """The administered bytes: the wire form, as text.

    Deliberately NOT key-sorted. Sorting would make two differently-ordered
    question sets hash alike, and order is administered — Score levels are
    positions, and Choice options are shown in the order given. What is sent
    is what is hashed.
    """
    return json.dumps(to_wire(questions), indent=2, ensure_ascii=False,
                      sort_keys=False)


def questions_sha256(questions):
    """Instrument digest for a question set."""
    return hashlib.sha256(canonical(questions).encode("utf-8")).hexdigest()
