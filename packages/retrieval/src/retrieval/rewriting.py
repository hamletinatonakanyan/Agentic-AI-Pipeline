"""Query rewriting with the LLM.

- multi_query: a few other phrasings, searched together. Helps recall when the user's words differ
  from the document's.
- hyde: the LLM writes a short passage that would answer the question, and we search with it too.
  An answer looks more like a document than a question does.
- step_back: also search a more general version of the question.

The original query always comes first and is always searched, so a rewrite can only add recall.
"""

from pydantic import BaseModel

from llm import LLM
from retrieval.models import RewriteStrategy

PROMPT_VERSION = "1"

SYSTEM = (
    "You help a document search engine. Always write in the same language as the user's question. "
    "Return only what is asked for, with no preface and no explanation."
)


class _Queries(BaseModel):
    queries: list[str]


def _dedupe(original: str, extra: list[str], limit: int) -> list[str]:
    seen = {original.strip().lower()}
    out = [original]
    for q in extra:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
        if len(out) > limit:
            break
    return out


class LLMRewriter:
    def __init__(self, llm: LLM, extra_queries: int = 3) -> None:
        self._llm = llm
        self._n = extra_queries

    def rewrite(self, query: str, strategy: RewriteStrategy) -> list[str]:
        if strategy == "none":
            return [query]
        if strategy == "multi_query":
            prompt = (
                f"Write {self._n} different search queries that could find documents answering "
                f"this question. Vary the wording and the angle.\n\nQuestion: {query}"
            )
            result = self._llm.complete_json(prompt, _Queries, system=SYSTEM)
            return _dedupe(query, result.queries, self._n)
        if strategy == "hyde":
            prompt = (
                "Write a short passage (3 to 5 sentences) of the kind that would appear in a "
                f"document answering this question. It is fine if you are unsure of the facts.\n\n"
                f"Question: {query}"
            )
            return _dedupe(query, [self._llm.complete(prompt, system=SYSTEM)], 1)
        if strategy == "step_back":
            prompt = (
                "Write one more general question that covers the background needed to answer "
                f"this question.\n\nQuestion: {query}"
            )
            return _dedupe(query, [self._llm.complete(prompt, system=SYSTEM)], 1)
        raise ValueError(f"unknown rewrite strategy {strategy!r}")
