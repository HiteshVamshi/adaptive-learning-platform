from __future__ import annotations

import re


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "find",
    "from",
    "how",
    "if",
    "in",
    "is",
    "of",
    "on",
    "or",
    "show",
    "solve",
    "state",
    "the",
    "to",
    "use",
    "what",
    "when",
    "with",
    "write",
}


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    return re.sub(r"\s+", " ", lowered)


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    return re.findall(r"[a-z0-9]+", normalized)


def significant_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS]
