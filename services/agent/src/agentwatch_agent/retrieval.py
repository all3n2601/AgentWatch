"""Deterministic in-process BM25 search over a task's reference documents.

This is the retrieval step on a CPU laptop without a vector database: real ranked lookup over
fixed documents, recorded as a `retrieval` step. It does not exercise Qdrant or network
retrieval latency; those belong to the resource-measurement tasks.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

RETRIEVAL_METHOD = "bm25-markdown-sections-v1"
_TOKEN = re.compile(r"[a-z0-9_]+")
_K1 = 1.2
_B = 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Passage:
    document: str
    heading: str
    text: str


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float


def split_sections(document: str, text: str) -> list[Passage]:
    """One passage per Markdown section; text before the first heading belongs to the title."""
    passages, heading, lines = [], document, []
    for line in text.splitlines():
        if line.startswith("#"):
            if any(part.strip() for part in lines):
                passages.append(Passage(document, heading, "\n".join(lines).strip()))
            heading, lines = line.lstrip("#").strip(), []
        else:
            lines.append(line)
    if any(part.strip() for part in lines):
        passages.append(Passage(document, heading, "\n".join(lines).strip()))
    return passages


class DocumentIndex:
    def __init__(self, passages: list[Passage]):
        if not passages:
            raise ValueError("A document index needs at least one passage")
        self.passages = passages
        self._terms = [Counter(tokenize(p.heading + " " + p.text)) for p in passages]
        self._lengths = [sum(terms.values()) for terms in self._terms]
        self._average_length = sum(self._lengths) / len(self._lengths)
        document_frequency = Counter(term for terms in self._terms for term in terms)
        count = len(passages)
        self._idf = {
            term: math.log(1 + (count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    @classmethod
    def from_directory(cls, docs: Path) -> "DocumentIndex":
        passages = []
        for path in sorted(docs.glob("*.md")):
            passages.extend(split_sections(path.name, path.read_text()))
        return cls(passages)

    def search(self, query: str, limit: int = 3) -> list[Hit]:
        terms = set(tokenize(query))
        hits = []
        for index, passage in enumerate(self.passages):
            frequencies, length = self._terms[index], self._lengths[index]
            score = 0.0
            for term in terms & frequencies.keys():
                frequency = frequencies[term]
                norm = _K1 * (1 - _B + _B * length / self._average_length)
                score += self._idf[term] * frequency * (_K1 + 1) / (frequency + norm)
            if score > 0:
                hits.append((-score, passage.document, index, Hit(passage, score)))
        # Ties break on document name and position, so equal scores never reorder between runs.
        return [hit for *_, hit in sorted(hits)[:limit]]
