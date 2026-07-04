"""Shared prompt text for the emotion task pair.

EmotionTask (English, Feelings Wheel) and EmotionTaskZh (Chinese, 七情/TCM)
are deliberately parallel — dual-taxonomy annotation for cross-cultural
comparison. The classification principles they share live here so the two
prompts can't silently drift apart; taxonomy-specific rules stay in each
task file.
"""

CLASSIFICATION_PRINCIPLES_CORE = """\
- **Represented**: emotions explicitly named or described in the text.
- **Implied**: emotions strongly suggested by action, dialogue, or situation without being named.
- **Invoked**: emotions the passage is designed to evoke in the reader through tone, imagery, or dramatic irony.
- **Quote verbatim**: copy a short phrase directly from the passage — do not paraphrase.
- **One entry per emotion**: if the same emotion appears multiple times, pick the strongest instance.
- **Empty is valid**: if the passage is emotionally flat or purely expository, return an empty list.\
"""
