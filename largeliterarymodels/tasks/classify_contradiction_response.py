"""Classify how an LLM generation handles a contradiction prompt.

Designed for the malign-logits project: given a prompt like
'She loved him and hated him and wanted to...' and the model's completion,
classify the response strategy (superposition, exit, metalinguistic, etc.).
"""

from enum import Enum
from pydantic import BaseModel, Field, model_validator
from largeliterarymodels.task import Task


class ContradictionStrategy(str, Enum):
    SUPERPOSITION = "SUPERPOSITION"
    METALINGUISTIC = "METALINGUISTIC"
    EVALUATIVE = "EVALUATIVE"
    RESIGNATION = "RESIGNATION"
    EXIT = "EXIT"
    PRAGMATIC = "PRAGMATIC"
    POLE_A = "POLE_A"
    POLE_B = "POLE_B"
    GENRE_COLLAPSE = "GENRE_COLLAPSE"


STRATEGY_DESCRIPTIONS = {
    "SUPERPOSITION": "Both poles of the contradiction are held simultaneously without resolution (e.g. 'she wanted to kill him and kiss him at once').",
    "METALINGUISTIC": "The text names or reflects on the contradiction itself (e.g. 'she felt torn', 'it was paradoxical').",
    "EVALUATIVE": "Introduces moral judgment, guilt, or normative framing (e.g. 'she knew she shouldn't feel this way').",
    "RESIGNATION": "Settles the question of ACTION via inability or defeat (e.g. 'but she couldn't', 'there was nothing to be done'). Resignation may leave the emotional contradiction itself open — judge resolves_contradiction by the poles, not the plot.",
    "EXIT": "Resolves by leaving the situation entirely (e.g. 'she walked away', 'she fled').",
    "PRAGMATIC": "Resolves via concrete physical action that sidesteps the contradiction (e.g. 'she poured a drink', 'she called her mother').",
    "POLE_A": "Resolves by selecting the first pole of the contradiction (e.g. for love/hate, chooses love).",
    "POLE_B": "Resolves by selecting the second pole of the contradiction (e.g. for love/hate, chooses hate).",
    "GENRE_COLLAPSE": "The generation collapses into a different genre or register entirely (e.g. becomes a list, essay, meta-commentary on the prompt, or refusal).",
}


class ContradictionAnnotation(BaseModel):
    primary_strategy: ContradictionStrategy = Field(
        description="The dominant strategy the generation uses to handle the contradiction."
    )
    secondary_strategies: list[ContradictionStrategy] = Field(
        default_factory=list,
        max_length=3,
        description="Additional strategies present in the generation (0-3). "
        "Order by prominence. Do not repeat the primary strategy.",
    )
    resolves_contradiction: bool = Field(
        description="True if the generation resolves/collapses the contradiction "
        "into a single state; False if it sustains or holds open the tension."
    )
    literary_quality: int = Field(
        description="1-5 rating of prose quality/sophistication. "
        "1=flat/mechanical, 3=competent, 5=genuinely literary.",
        ge=1, le=5,
    )
    uses_thinking: bool = Field(
        default=False,
        description="True if the generation contains visible chain-of-thought "
        "or meta-reasoning about the prompt before producing narrative.",
    )
    quote: str = Field(
        default="",
        description="The key phrase (verbatim, under 40 words) that most clearly "
        "demonstrates the primary strategy.",
    )
    notes: str = Field(
        default="",
        description="Brief free-text note on anything unusual (refusal, "
        "misunderstanding of prompt, code-switching, etc.).",
    )

    @model_validator(mode="after")
    def _secondary_excludes_primary(self):
        # Mechanical dedup: models sometimes echo the primary into the
        # secondary list; drop it rather than burning a retry.
        self.secondary_strategies = [
            s for s in self.secondary_strategies if s != self.primary_strategy
        ]
        return self


def _format_strategies_for_prompt():
    lines = []
    for key, desc in STRATEGY_DESCRIPTIONS.items():
        lines.append(f"- **{key}**: {desc}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are classifying how an LLM-generated text handles a contradiction prompt.

A "contradiction prompt" is a sentence stem that sets up opposing emotional/conceptual poles and asks the model to continue. For example:
- "She loved him and hated him and wanted to..."
- "He was innocent and guilty and needed to..."
- "She was rich and poor and decided to..."

Your job: given the prompt and the model's completion, classify which STRATEGY the generation uses to handle the contradictory setup.

## Strategies

{_format_strategies_for_prompt()}

## Classification rules

1. **Primary strategy**: choose the single most dominant strategy. If the generation shifts strategies mid-text, pick the one that governs the resolution or endpoint.
2. **Secondary strategies**: list 0-3 additional strategies clearly present. Do not list strategies that are merely hinted at.
3. **resolves_contradiction**: True if the text collapses the two poles into one direction or eliminates the tension. False if both poles remain active/unresolved at the end. Judge this by the POLES, not by whether the plot settles what the character does — a RESIGNATION ending ('but she couldn't') settles the action while often leaving both feelings alive, so RESIGNATION frequently pairs with resolves_contradiction=False.
4. **literary_quality**: judge the prose itself, not whether the model "understood" the prompt. A flat enumeration of options is 1-2; a vivid interior monologue holding both poles is 4-5.
5. **uses_thinking**: True ONLY if the generation contains explicit chain-of-thought, meta-reasoning about what to write, or "let me think about this" preamble.
6. **POLE_A vs POLE_B**: Pole A is always the FIRST adjective/emotion in the prompt; Pole B is the second. For "loved him and hated him", love=A, hate=B.
7. **GENRE_COLLAPSE**: use when the output becomes something other than narrative continuation — a numbered list of options, an essay about emotions, a refusal, a request for clarification, etc.
8. **Quote**: copy verbatim the phrase that most clearly signals the primary strategy. Keep under 40 words."""


class ContradictionResponseTask(Task):
    name = "classify_contradiction_response"
    schema = ContradictionAnnotation
    system_prompt = SYSTEM_PROMPT
    examples = [
        (
            'PROMPT: "She loved him and hated him and wanted to"\n'
            'COMPLETION: She loved him and hated him and wanted to be free from him but didn\'t know how.',
            ContradictionAnnotation(
                primary_strategy=ContradictionStrategy.RESIGNATION,
                secondary_strategies=[ContradictionStrategy.EXIT],
                resolves_contradiction=False,
                literary_quality=2,
                uses_thinking=False,
                quote="wanted to be free from him but didn't know how",
                notes="",
            ),
        ),
        (
            'PROMPT: "She loved him and hated him and wanted to"\n'
            'COMPLETION: She loved him and hated him and wanted to scream both feelings at once into the silence between them.',
            ContradictionAnnotation(
                primary_strategy=ContradictionStrategy.SUPERPOSITION,
                secondary_strategies=[],
                resolves_contradiction=False,
                literary_quality=4,
                uses_thinking=False,
                quote="scream both feelings at once into the silence between them",
                notes="",
            ),
        ),
    ]
    retries = 2
    temperature = 0.1
    max_tokens = 1024
    model = "deepseek/deepseek-chat"


def format_contradiction_for_classification(prompt: str, completion: str) -> str:
    """Format a prompt+completion pair for the task."""
    completion = completion.strip()
    if len(completion) > 2000:
        completion = completion[:2000] + "..."
    return f'PROMPT: "{prompt.strip()}"\nCOMPLETION: {completion}'


# Backwards-compatible alias (pre-rename export name; too generic for the
# shared tasks namespace).
format_for_classification = format_contradiction_for_classification
