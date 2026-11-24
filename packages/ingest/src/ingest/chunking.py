"""Chunking: cut page text into pieces of about N tokens, with exact character offsets.

Strategies
- fixed:     every N tokens, stepping N - overlap.
- recursive: split on paragraphs, then lines, sentences and words until each part fits, then merge
             neighbouring parts up to N tokens. Chunks overlap by up to `overlap` tokens.
- markdown:  like recursive, but never crosses a `#` heading.

Tables are always split by rows, whatever the strategy, and the header row is repeated in every
chunk so that each chunk makes sense alone. A table chunk's text is not one slice of the source
(the header is added), so its `char_start` and `char_end` cover the rows it contains.
"""

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from ingest.models import ChunkOptions, Document, TextChunk
from ingest.tables import find_tables
from ingest.tokenizer import ApproxTokenizer

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_CJK_CHAR = re.compile("[぀-ヿ㐀-䶿一-鿿가-힯]")

# Tried in order: paragraph, line, sentence, word.
_SEPARATORS = [
    re.compile(r"\n\s*\n"),
    re.compile(r"\n"),
    re.compile(r"(?<=[.!?;])\s+|(?<=[。！？])"),
    re.compile(r"\s+"),
]


@dataclass(frozen=True)
class Piece:
    text: str
    start: int
    end: int
    section_path: str
    tokens: int


class _Counter:
    """Token arithmetic over one text: how many tokens lie in a character range."""

    def __init__(self, text: str, tokenizer: ApproxTokenizer) -> None:
        self.spans = tokenizer.spans(text)
        self.starts = [s for s, _ in self.spans]

    def count(self, a: int, b: int) -> int:
        return bisect_left(self.starts, b) - bisect_left(self.starts, a)

    def index(self, pos: int) -> int:
        """Index of the first token that starts at or after `pos`."""
        return bisect_left(self.starts, pos)


# ----- headings ---------------------------------------------------------------------------


def _headings(text: str) -> list[tuple[int, str]]:
    """(character offset, full heading path) for each heading line. Ignores code fences."""
    out: list[tuple[int, str]] = []
    stack: list[tuple[int, str]] = []
    in_fence = False
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            m = _HEADING.match(stripped)
            if m:
                level = len(m.group(1))
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, m.group(2)))
                out.append((pos, " > ".join(t for _, t in stack)))
        pos += len(line)
    return out


def _blocks(
    text: str, strategy: str, headings: list[tuple[int, str]]
) -> list[tuple[str, int, int]]:
    """Split the page into ("table" | "text", start, end) blocks."""
    blocks: list[tuple[str, int, int]] = []
    cursor = 0
    for a, b in [*find_tables(text), (len(text), len(text))]:
        if a > cursor:
            blocks.append(("text", cursor, a))
        if b > a:
            blocks.append(("table", a, b))
        cursor = b

    if strategy != "markdown":
        return blocks
    cuts = [pos for pos, _ in headings]
    out: list[tuple[str, int, int]] = []
    for kind, a, b in blocks:
        if kind == "table":
            out.append((kind, a, b))
            continue
        edges = [a, *[c for c in cuts if a < c < b], b]
        out += [("text", x, y) for x, y in zip(edges, edges[1:], strict=False)]
    return out


# ----- prose chunkers -----------------------------------------------------------------------


def _fixed(a: int, b: int, counter: _Counter, size: int, overlap: int) -> list[tuple[int, int]]:
    ia, ib = counter.index(a), counter.index(b)
    if ib <= ia:
        return []
    step = size - overlap
    out = []
    i = ia
    while i < ib:
        j = min(i + size, ib)
        out.append((counter.spans[i][0], counter.spans[j - 1][1]))
        if j == ib:
            break
        i += step
    return out


def _hard_windows(a: int, b: int, counter: _Counter, size: int) -> list[tuple[int, int]]:
    """Last resort for a piece with no separators (for example a long CJK string)."""
    ia, ib = counter.index(a), counter.index(b)
    out = []
    for i in range(ia, ib, size):
        j = min(i + size, ib)
        out.append((a if i == ia else counter.spans[i][0], b if j == ib else counter.spans[j][0]))
    return out or [(a, b)]


def _pieces(text: str, a: int, b: int, level: int) -> list[tuple[int, int]]:
    """Cut [a, b) at the first separator level that occurs in it. Each piece keeps its trailing
    separator, so the pieces tile the range exactly."""
    segment = text[a:b]
    while level < len(_SEPARATORS):
        cuts = {0, len(segment)}
        cuts.update(
            m.end() for m in _SEPARATORS[level].finditer(segment) if 0 < m.end() < len(segment)
        )
        if len(cuts) > 2:
            edges = sorted(cuts)
            return [(a + x, a + y) for x, y in zip(edges, edges[1:], strict=False)], level
        level += 1
    return [(a, b)], level


def _split(text: str, a: int, b: int, level: int, counter: _Counter, size: int):
    """Chunk ranges for [a, b), each at most `size` tokens.

    Hierarchical, like a table of contents: small neighbouring pieces are packed together, but a
    piece that is too big on its own is split further on its own, and its parts are never packed
    with its neighbours. Boundaries therefore settle at paragraph (and sentence) edges, so editing
    one paragraph changes the chunks around it and leaves the rest of the file's chunks the same.
    """
    if counter.count(a, b) <= size:
        return [(a, b)]
    pieces, level = _pieces(text, a, b, level)
    if len(pieces) == 1:  # no separator left: hard-cut by tokens
        return _hard_windows(a, b, counter, size)

    out: list[tuple[int, int]] = []
    group: tuple[int, int] | None = None
    for s, e in pieces:
        if counter.count(s, e) > size:
            if group:
                out.append(group)
                group = None
            out += _split(text, s, e, level + 1, counter, size)
        elif group and counter.count(group[0], e) <= size:
            group = (group[0], e)
        else:
            if group:
                out.append(group)
            group = (s, e)
    if group:
        out.append(group)
    return out


def _mid_word(text: str, pos: int) -> bool:
    return (
        0 < pos < len(text)
        and text[pos - 1].isalnum()
        and text[pos].isalnum()
        and not _CJK_CHAR.match(text[pos])
        and not _CJK_CHAR.match(text[pos - 1])
    )


def _overlap_start(text: str, counter: _Counter, seg_start: int, k: int, floor: int) -> int:
    """Where a new chunk starts so it repeats about `k` tokens before `seg_start`."""
    target = counter.index(seg_start)
    idx = max(target - k, 0)
    while idx < target and _mid_word(text, counter.spans[idx][0]):
        idx += 1  # do not start in the middle of a word
    pos = seg_start if idx >= target else counter.spans[idx][0]
    return max(pos, floor)


def _recursive(
    text: str, a: int, b: int, counter: _Counter, size: int, overlap: int
) -> list[tuple[int, int]]:
    ranges = _split(text, a, b, 0, counter, size)
    if not ranges or overlap == 0:
        return ranges
    out = [ranges[0]]
    for s, e in ranges[1:]:
        prev_start, prev_end = out[-1]
        room = max(size - counter.count(s, e), 0)  # keep the overlap within the size budget
        # Never repeat the whole previous chunk: the next one must start after it started.
        k = min(overlap, room, max(counter.count(prev_start, prev_end) - 1, 0))
        out.append((_overlap_start(text, counter, s, k, floor=prev_start), e))
    return out


# ----- tables ---------------------------------------------------------------------------------


def _table_pieces(text: str, a: int, b: int, size: int, tokenizer: ApproxTokenizer):
    lines = text[a:b].splitlines(keepends=True)
    offsets, pos = [], a
    for line in lines:
        offsets.append(pos)
        pos += len(line)

    header = lines[0].rstrip("\r\n") + "\n" + lines[1].rstrip("\r\n")
    header_tokens = len(tokenizer.spans(header))
    rows = list(range(2, len(lines)))
    if not rows:
        return [(header, offsets[0], offsets[1] + len(lines[1].rstrip("\r\n")))]

    out = []
    batch: list[int] = []
    used = header_tokens

    def flush() -> None:
        body = "\n".join(lines[i].rstrip("\r\n") for i in batch)
        last = batch[-1]
        out.append(
            (
                header + "\n" + body,
                offsets[batch[0]],
                offsets[last] + len(lines[last].rstrip("\r\n")),
            )
        )

    for i in rows:
        n = len(tokenizer.spans(lines[i]))
        if batch and used + n > size:
            flush()
            batch, used = [], header_tokens
        batch.append(i)
        used += n
    flush()
    return out


# ----- public API -------------------------------------------------------------------------------


def chunk_text(
    text: str, opts: ChunkOptions, tokenizer: ApproxTokenizer | None = None
) -> list[Piece]:
    tokenizer = tokenizer or ApproxTokenizer()
    counter = _Counter(text, tokenizer)
    headings = _headings(text)
    heading_pos = [p for p, _ in headings]

    def section_at(pos: int) -> str:
        i = bisect_right(heading_pos, pos) - 1
        return headings[i][1] if i >= 0 else ""

    pieces: list[Piece] = []
    for kind, a, b in _blocks(text, opts.strategy, headings):
        if kind == "table":
            for body, start, end in _table_pieces(text, a, b, opts.chunk_size, tokenizer):
                pieces.append(
                    Piece(body, start, end, section_at(start), len(tokenizer.spans(body)))
                )
            continue
        if opts.strategy == "fixed":
            ranges = _fixed(a, b, counter, opts.chunk_size, opts.overlap)
        else:
            ranges = _recursive(text, a, b, counter, opts.chunk_size, opts.overlap)
        for x, y in ranges:
            raw = text[x:y]
            body = raw.strip()
            if not body:
                continue
            start = x + (len(raw) - len(raw.lstrip()))
            pieces.append(
                Piece(body, start, start + len(body), section_at(start), counter.count(x, y))
            )
    return pieces


def chunk_document(
    doc: Document, opts: ChunkOptions, tokenizer: ApproxTokenizer | None = None
) -> list[TextChunk]:
    """Chunk every page. Page number, sheet label and parser travel with each chunk."""
    tokenizer = tokenizer or ApproxTokenizer()
    chunks: list[TextChunk] = []
    for page in doc.pages:
        for piece in chunk_text(page.text, opts, tokenizer):
            chunks.append(
                TextChunk(
                    text=piece.text,
                    page_no=page.page_no,
                    label=page.label,
                    section_path=piece.section_path,
                    char_start=piece.start,
                    char_end=piece.end,
                    tokens=piece.tokens,
                    parser=page.parser,
                )
            )
    return chunks
