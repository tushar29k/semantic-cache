"""Deterministic token-level mutations for the boundary variants.

Each mutation takes a prompt and returns a changed prompt, or None
when it does not apply (a mutation that changes nothing is not a
variant). Same transforms the hand-written boundary seeds use, applied
mechanically to every base prompt, so the variant suite measures the
transform, not the author's aim. Functions only — the generator owns
the file format.
"""

_AUX = {"do", "does", "can", "is", "will", "did", "should"}

_NUMBERS = {
    "thirty": "ninety", "ninety": "thirty",
    "seven": "thirty", "fourteen": "sixty",
    "one": "two", "two": "three", "three": "five", "five": "ten",
    "first": "second", "second": "third",
}

_ENTITIES = {
    "chennai": "mumbai", "mumbai": "chennai",
    "savings": "current", "current": "savings",
}

_DATES = {
    "yesterday": "today", "today": "tomorrow", "tomorrow": "yesterday",
    "morning": "evening", "evening": "morning",
}


def _swap(tokens, table):
    out, changed = [], False
    for tok in tokens:
        if not changed and tok in table:
            out.append(table[tok])
            changed = True
        else:
            out.append(tok)
    return " ".join(out) if changed else None


def add_negation(prompt):
    """Insert 'not' after the first auxiliary verb."""
    tokens = prompt.split()
    for i, tok in enumerate(tokens):
        if tok in _AUX:
            return " ".join(tokens[:i + 1] + ["not"] + tokens[i + 1:])
    return None


def swap_number(prompt):
    return _swap(prompt.split(), _NUMBERS)


def swap_entity(prompt):
    return _swap(prompt.split(), _ENTITIES)


def swap_date(prompt):
    return _swap(prompt.split(), _DATES)


def drop_last(prompt):
    """Drop the final token — one shared token fewer, same length class."""
    tokens = prompt.split()
    if len(tokens) < 4:
        return None
    return " ".join(tokens[:-1])


def scramble(prompt):
    """Reverse token order: identical multiset, destroyed word order."""
    tokens = prompt.split()
    if len(tokens) < 4:
        return None
    return " ".join(reversed(tokens))


MUTATIONS = {
    "add_negation": add_negation,
    "swap_number": swap_number,
    "swap_entity": swap_entity,
    "swap_date": swap_date,
    "drop_last": drop_last,
    "scramble": scramble,
}
