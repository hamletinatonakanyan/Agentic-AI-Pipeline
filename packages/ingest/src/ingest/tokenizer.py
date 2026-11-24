"""A dependency-free approximate tokenizer.

Real model tokenizers need a download and differ per model. Chunk sizes only need to be roughly
right, so this approximates: a Latin or other spaced word counts one token per 4 characters, each
CJK character counts one token, and each punctuation mark counts one token. That is close to the
usual "about 4 characters per token" for English and safe for any language.

Because it returns character ranges, chunkers can cut the original text exactly, and chunks
carry exact `char_start` and `char_end` for citations.
"""

import re

_CJK = "぀-ヿ㐀-䶿一-鿿가-힯"
_TOKEN = re.compile(rf"[{_CJK}]|[^\W{_CJK}]{{1,4}}|[^\w\s]")


class ApproxTokenizer:
    def spans(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in _TOKEN.finditer(text)]

    def count(self, text: str) -> int:
        return sum(1 for _ in _TOKEN.finditer(text))
