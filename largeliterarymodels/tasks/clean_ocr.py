"""OCR text cleaning task.

Corrects common OCR errors in early modern English texts: long-s (ſ/f→s),
broken words, garbled characters, ligature artifacts, VV→W, rn→m, etc.
Preserves original spelling and capitalization conventions — does not modernize.

Designed for ECCO and similar page-level OCR corpora. Best with fast small
models (gemma-4-e2b-it ~1.4s/page, qwen3.5-35b-a3b ~5s/page).

Usage:
    from largeliterarymodels.tasks import OCRCleanTask

    task = OCRCleanTask(model="lmstudio/gemma-4-e2b-it")
    cleaned = task.run(dirty_page)           # single page
    cleaned_pages = task.map(dirty_pages)    # batch
"""

from largeliterarymodels.task import Task


SYSTEM_PROMPT = """\
You are an expert at correcting OCR errors in early modern English texts (1500-1800).

Given a passage of dirty OCR text, produce a cleaned version. Fix:
- Long-s: ſ or f used where s is meant (e.g. "hiſtory" → "history", "fhall" → "shall")
- VV/vv ligatures: "VVhen" → "When", "vvith" → "with"
- Broken words: rejoin words split across lines with hyphens (e.g. "Gentle- man" → "Gentleman", "ex-\nplain" → "explain", "ut-\nmost" → "utmost"). Remove the hyphen and join the parts into one word. Keep genuine compound words hyphenated (e.g. "well-known", "self-same").
- Common OCR substitutions: rn→m, cl→d, li→h, fi→fi
- Garbled characters: random punctuation or symbols replacing letters
- Missing or extra spaces between words
- Catchwords and page artifacts (running headers, page numbers mid-text)

Do NOT:
- Modernize spelling (keep "publick", "compleat", "hath", "doth", etc.)
- Modernize capitalization conventions (keep "the King", "a Lady", etc.)
- Expand abbreviations (keep "Mr.", "ye", "&c.")
- Change punctuation style (keep long dashes, semicolons, period usage)
- Add or remove content — output must be the same text, just cleaned

Return ONLY the cleaned text. No commentary, no explanations, no markdown."""


EXAMPLES = [
    (
        "THe Hiſtory of the moſt Renowned and Victorious Princeſs "
        "ELIZABETH, Late Queen of England. Containing all the moſt Im-\n"
        "portant and Remarkable Paſſages of State, both at Home and "
        "Abroad (ſo far as they were linked with Engliſh Affairs) during "
        "her Long and Proſperous Reign. VVritten by Mr. Cambden.",
        "The History of the most Renowned and Victorious Princess "
        "ELIZABETH, Late Queen of England. Containing all the most "
        "Important and Remarkable Passages of State, both at Home and "
        "Abroad (so far as they were linked with English Affairs) during "
        "her Long and Prosperous Reign. Written by Mr. Cambden.",
    ),
]


DEFAULT_OCR_MODEL = 'lmstudio/qwen/qwen3.6-35b-a3b'


class OCRCleanTask(Task):
    name = "clean_ocr"
    model = DEFAULT_OCR_MODEL
    schema = None
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 1
    temperature = 0.1

    def run(self, prompt, model=None, force=False, **kwargs):
        """Clean OCR text. Returns the cleaned string directly."""
        llm = self._get_llm(model)
        return llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            force=force,
        )

    def map(self, prompts, model=None, num_workers=1, force=False,
            verbose=False, **kwargs):
        """Clean multiple pages in parallel."""
        llm = self._get_llm(model)
        return llm.map(
            prompts,
            system_prompt=self.system_prompt,
            num_workers=num_workers,
            force=force,
            verbose=verbose,
        )
