"""SocialNetworkTask: extract characters, relations, events, and dialogue
from a novel via chunked sequential processing.

Usage:
    from largeliterarymodels.tasks import SocialNetworkTask

    task = SocialNetworkTask(model='lmstudio/qwen/qwen3.6-27b')
    result = task.run('_chadwyck/Eighteenth-Century_Fiction/defoe.06')

    result['characters']   # list of character dicts
    result['relations']    # list of social relation dicts
    result['events']       # list of event dicts
    result['dialogue']     # list of dialogue dicts
    result['summaries']    # list of per-chunk summary dicts
"""

import re
import sys
from copy import deepcopy

from ..task import SequentialTask


RELATION_TYPES = [
    'parent_of', 'child_of', 'sibling_of', 'spouse_of', 'married',
    'courted', 'courted_by', 'attracted_to', 'rejected',
    'serves', 'served_by', 'employed_by', 'patron_of',
    'friend_of', 'confidante_of', 'enemy_of', 'rival_of',
    'allied_with', 'betrayed', 'deceived',
    'indebted_to', 'inherited_from',
    'same_as',
]

SYSTEM_PROMPT = """\
You are extracting a social network from a novel, processing 10 passages at a time.

You receive:
1. A CHARACTER REGISTER (all characters identified so far, with IDs)
2. STORY SO FAR — a chain of short summaries from previous chunks, using character IDs
3. The next 10 PASSAGES of text

Return valid JSON with these fields:

{
  "new_characters": [
    {"id": "C12", "name": "the Linen-Draper", "gender": "male", "class": "merchant",
     "notes": "Moll's second husband",
     "intro_text": "a grave Gentleman, and a good Man; one who had a very good Trade, and a very good Reputation",
     "descriptions": ["grave Gentleman", "good Man", "good Trade", "good Reputation"]}
  ],
  "relations": [
    {"a": "C01", "b": "C12", "type": "spouse_of", "passage": "P042",
     "detail": "married after short courtship"}
  ],
  "events": [
    {"who": "C01", "what": "married", "whom": "C12", "passage": "P042",
     "where": "London", "detail": "married the Linen-Draper after a brief courtship"},
    {"who": "C01", "what": "arrived", "passage": "P220",
     "where": "Virginia", "detail": "transported to Virginia with her husband"}
  ],
  "dialogue": [
    {"speaker": "C11", "addressee": "C01", "passage": "P015",
     "gist": "declares his love and promises marriage"}
  ],
  "chunk_summary": "A ~100-word summary of JUST these 10 passages, using character IDs \
inline: e.g. 'Moll (C01) married the Draper (C12) but he squandered their fortune.'"
}

## Character rules

- Every PERSON mentioned gets an ID. Not objects or places — only people.
- Check the register before creating. Use existing IDs where possible.
- When unsure if two are the same person, create a new ID and emit a same_as \
relation later.
- Next available ID is shown in the register header.
- For each new character, include:
  - "intro_text": copy the sentence(s) from the passage that INTRODUCE this character \
to the reader — the first description or characterisation, verbatim from the text.
  - "descriptions": a list of key descriptive phrases quoted directly from the text \
(e.g. ["tall", "grave Gentleman", "exceeding Beauty"]). Extract the words the \
narrator or characters use to describe this person. Up to 8 phrases.

## Relations (closed vocabulary — use ONLY these types)

Kinship: parent_of, child_of, sibling_of, spouse_of, married
Courtship: courted, courted_by, attracted_to, rejected
Service: serves, served_by, employed_by, patron_of
Social: friend_of, confidante_of, enemy_of, rival_of, allied_with, betrayed, deceived
Economic: indebted_to, inherited_from
Identity: same_as (two IDs are the same person)

Tag each relation with the passage number where it's evidenced.

## Events

Capture actions that happen in these passages: marriages, deaths, thefts, arrivals, \
departures, betrayals, discoveries. Each event has who, what, whom (optional), \
where (optional — location if mentioned or inferable), passage, and a brief detail. \
Include arrivals/departures as events to track character movement.

## Dialogue

Capture notable speech acts: who spoke to whom, in which passage, and a one-sentence \
gist of what they said. Not every line of dialogue — just exchanges that reveal \
social relations or advance the plot.

## chunk_summary rules

- Summarize ONLY these 10 passages, around 100 words (80-120).
- Use character IDs inline: "Moll (C01) discovered her husband (C14) was her brother."
- Focus on social events: who met, married, betrayed, served, deceived whom.

Return ONLY valid JSON. No commentary before or after.
"""


class CharacterRegister:
    """Maintains a deduplicated character register with same_as merging."""

    def __init__(self):
        self.characters = {}
        self.next_id = 1
        self.merged = {}

    def add(self, char_dict):
        cid = char_dict.get('id', '')
        if not cid or not re.match(r'^C\d+$', cid):
            return
        num = int(cid[1:])
        if num >= self.next_id:
            self.next_id = num + 1
        self.characters[cid] = {
            'id': cid,
            'name': char_dict.get('name', '?'),
            'gender': char_dict.get('gender', 'unknown'),
            'class': char_dict.get('class', 'unknown'),
            'notes': char_dict.get('notes', ''),
            'aliases': [],
            'intro_text': char_dict.get('intro_text', ''),
            'descriptions': char_dict.get('descriptions', []),
        }

    def apply_same_as(self, a_id, b_id):
        if a_id not in self.characters and b_id not in self.characters:
            return
        if a_id not in self.characters:
            a_id, b_id = b_id, a_id
        if b_id not in self.characters:
            self.merged[b_id] = a_id
            return
        keep, remove = (a_id, b_id) if int(a_id[1:]) < int(b_id[1:]) else (b_id, a_id)
        removed = self.characters.pop(remove, None)
        if removed and keep in self.characters:
            keeper = self.characters[keep]
            if removed['name'] not in keeper.get('aliases', []):
                keeper.setdefault('aliases', []).append(removed['name'])
        self.merged[remove] = keep

    def resolve_id(self, cid):
        seen = set()
        while cid in self.merged and cid not in seen:
            seen.add(cid)
            cid = self.merged[cid]
        return cid

    def format_for_prompt(self):
        if not self.characters:
            return "Next available ID: C01\n(No characters yet)"
        lines = [f"Next available ID: C{self.next_id:02d}"]
        for cid in sorted(self.characters.keys(), key=lambda x: int(x[1:])):
            c = self.characters[cid]
            line = f"{cid}: {c['name']} ({c['gender']}, {c['class']})"
            if c.get('aliases'):
                line += f" / {', '.join(c['aliases'])}"
            lines.append(line)
        return '\n'.join(lines)

    def all_as_list(self):
        return list(self.characters.values())


class SocialNetworkTask(SequentialTask):
    """Extract a social network from a novel via chunked sequential processing.

    Outputs characters, typed social relations, events, dialogue, and
    per-chunk narrative summaries.
    """

    name = 'social_network'
    system_prompt = SYSTEM_PROMPT
    chunk_size = 10
    max_tokens = 8192
    temperature = 0.2

    def build_state(self):
        return {
            'register': CharacterRegister(),
            'summaries': [],
            'all_relations': [],
            'all_events': [],
            'all_dialogue': [],
        }

    def format_context(self, state):
        reg = state['register']
        summaries = state['summaries']
        parts = []
        parts.append(f"CHARACTER REGISTER:\n{reg.format_for_prompt()}")
        parts.append("")
        if summaries:
            story_parts = []
            for s in summaries:
                story_parts.append(f"[P{s['start']:03d}-P{s['end']:03d}] {s['text']}")
            parts.append(f"STORY SO FAR:\n" + '\n\n'.join(story_parts))
        else:
            parts.append("STORY SO FAR:\n(Beginning of novel)")
        return '\n'.join(parts)

    def update_state(self, state, result, chunk_idx, start, end):
        state = deepcopy(state)
        reg = state['register']

        for c in result.get('new_characters', []):
            reg.add(c)

        for r in result.get('relations', []):
            if not r.get('a') or not r.get('b') or not r.get('type'):
                continue
            r['a'] = reg.resolve_id(r['a'])
            r['b'] = reg.resolve_id(r['b'])
            if r['type'] == 'same_as':
                reg.apply_same_as(r['a'], r['b'])
            state['all_relations'].append(r)

        for e in result.get('events', []):
            if e.get('who'):
                e['who'] = reg.resolve_id(e['who'])
            if e.get('whom'):
                e['whom'] = reg.resolve_id(e['whom'])
            state['all_events'].append(e)

        for d in result.get('dialogue', []):
            if d.get('speaker'):
                d['speaker'] = reg.resolve_id(d['speaker'])
            if d.get('addressee'):
                d['addressee'] = reg.resolve_id(d['addressee'])
            state['all_dialogue'].append(d)

        chunk_summary = result.get('chunk_summary', '')
        if chunk_summary:
            state['summaries'].append({
                'start': start, 'end': end - 1, 'text': chunk_summary,
            })

        return state

    def aggregate(self, all_results, state):
        return {
            'characters': state['register'].all_as_list(),
            'relations': state['all_relations'],
            'events': state['all_events'],
            'dialogue': state['all_dialogue'],
            'summaries': state['summaries'],
        }

    def log_chunk(self, chunk_idx, start, end, elapsed, state, result):
        reg = state['register']
        n_chars = len(reg.characters)
        n_rels = len(state['all_relations'])
        n_events = len(state['all_events'])
        n_dialogue = len(state['all_dialogue'])

        new_chars = result.get('new_characters', [])
        new_names = ', '.join(
            c.get('id', '?') + '=' + c.get('name', '?')
            for c in new_chars[:4]
        )
        if len(new_chars) > 4:
            new_names += f" (+{len(new_chars)-4})"

        same_as_rels = [r for r in result.get('relations', [])
                        if r.get('type') == 'same_as']
        social_rels = [r for r in result.get('relations', [])
                       if r.get('type') != 'same_as' and r.get('a') and r.get('b')]
        events = result.get('events', [])
        dialogue = result.get('dialogue', [])
        summary = result.get('chunk_summary', '')

        status = (f"  [Chunk {chunk_idx:02d}] P{start:03d}-P{end-1:03d}  "
                  f"{elapsed:6.1f}s  {n_chars} chars, {n_rels} rels, "
                  f"{n_events} events, {n_dialogue} dialogue")
        if new_names:
            status += f"\n         new: {new_names}"
        if same_as_rels:
            merge_strs = [r['a'] + '=' + r['b'] for r in same_as_rels]
            status += f"\n         merges: {'; '.join(merge_strs)}"
        if social_rels:
            rel_strs = [r['a'] + ' ' + r['type'] + ' ' + r['b']
                        for r in social_rels[:4]]
            rel_preview = '; '.join(rel_strs)
            if len(social_rels) > 4:
                rel_preview += f" (+{len(social_rels)-4})"
            status += f"\n         rels: {rel_preview}"
        if events:
            ev_preview = '; '.join(
                (e.get('who', '?') + ' ' + e.get('what', '?'))
                for e in events[:3]
            )
            if len(events) > 3:
                ev_preview += f" (+{len(events)-3})"
            status += f"\n         events: {ev_preview}"
        if dialogue:
            dl_preview = '; '.join(
                ((d.get('speaker') or '?') + '→' + (d.get('addressee') or '?'))
                for d in dialogue[:3]
            )
            if len(dialogue) > 3:
                dl_preview += f" (+{len(dialogue)-3})"
            status += f"\n         dialogue: {dl_preview}"
        if summary:
            status += f"\n         summary: {summary[:140]}..."
        print(status, file=sys.stderr)
