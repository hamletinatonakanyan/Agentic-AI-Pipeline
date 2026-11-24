"""Shared test helpers for ingest: fixture builders and fakes. No network, no Docker."""

import io

import pytest
from docx import Document as DocxDocument
from openpyxl import Workbook

from ingest.models import UploadedFile


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages: list[str | None]) -> bytes:
    """A real, minimal PDF. A page of text has a text layer. `None` makes a blank page,
    which stands in for a scanned page (no text layer)."""
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    kids = []
    next_id = 4
    for text in pages:
        page_id = next_id
        kids.append(page_id)
        page = "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >>"
        if text is not None:
            content_id = next_id + 1
            stream = f"BT /F1 12 Tf 72 720 Td ({_pdf_escape(text)}) Tj ET".encode("latin-1")
            objects[content_id] = (
                b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
            )
            page += f" /Contents {content_id} 0 R"
            next_id += 2
        else:
            next_id += 1
        objects[page_id] = (page + " >>").encode()
    objects[2] = (
        "<< /Type /Pages /Kids [%s] /Count %d >>" % (" ".join(f"{k} 0 R" for k in kids), len(kids))
    ).encode()

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for obj_id in sorted(objects):
        offsets[obj_id] = len(out)
        out += b"%d 0 obj\n" % obj_id + objects[obj_id] + b"\nendobj\n"
    xref_at = len(out)
    size = max(objects) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for obj_id in range(1, size):
        out += b"%010d 00000 n \n" % offsets[obj_id]
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref_at)
    return bytes(out)


def build_docx() -> bytes:
    doc = DocxDocument()
    doc.add_heading("Quarterly report", level=1)
    doc.add_paragraph("Intro paragraph about the quarter.")
    doc.add_heading("Numbers", level=2)
    table = doc.add_table(rows=3, cols=2)
    for r, (a, b) in enumerate([("Region", "Sales"), ("North", "10"), ("South", "20")]):
        table.cell(r, 0).text = a
        table.cell(r, 1).text = b
    doc.add_paragraph("Closing remarks.")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def build_xlsx() -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "Sales"
    ws.append(["Region", "Units", "Price"])
    ws.append(["North", 10, 2.5])
    ws.append(["South", 20.0, 3])
    second = wb.create_sheet("Notes")
    second.append(["Note"])
    second.append(["pipe | inside"])
    wb.create_sheet("Empty")
    hidden = wb.create_sheet("Secret")
    hidden.append(["x"])
    hidden.sheet_state = "hidden"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


@pytest.fixture
def make_pdf():
    return build_pdf


@pytest.fixture
def docx_bytes() -> bytes:
    return build_docx()


@pytest.fixture
def xlsx_bytes() -> bytes:
    return build_xlsx()


def upload(name: str, data: bytes | str) -> UploadedFile:
    return UploadedFile(name=name, data=data.encode() if isinstance(data, str) else data)


@pytest.fixture
def make_upload():
    return upload


class DictCache:
    """In-memory ParseCache."""

    def __init__(self) -> None:
        self.data: dict[tuple[str, int, str], str] = {}

    def get(self, content_hash: str, page_no: int, tier: str) -> str | None:
        return self.data.get((content_hash, page_no, tier))

    def put(self, content_hash: str, page_no: int, tier: str, text: str) -> None:
        self.data[(content_hash, page_no, tier)] = text


class FakeRemote:
    """A PdfParser that records calls and returns canned text per page."""

    def __init__(self, texts: dict[int, str] | None = None, fail: Exception | None = None) -> None:
        self.texts = texts or {}
        self.fail = fail
        self.calls: list[list[int]] = []

    def parse(self, pdf: bytes, pages: list[int]) -> dict[int, str]:
        self.calls.append(list(pages))
        if self.fail:
            raise self.fail
        return {p: self.texts.get(p, f"Scanned text of page {p}") for p in pages}


@pytest.fixture
def cache() -> DictCache:
    return DictCache()


@pytest.fixture
def remote_cls():
    """The FakeRemote class, so a test can build one with its own canned texts."""
    return FakeRemote


# ----- fakes for the service ---------------------------------------------------------------------


from common import AppError  # noqa: E402
from ingest.manifest import Manifest  # noqa: E402
from ingest.models import CollectionInfo, CollectionSpec, PointRecord  # noqa: E402
from ingest.parsers import FileParser  # noqa: E402
from ingest.parsers.pdf import PdfRouter  # noqa: E402
from ingest.service import IngestService  # noqa: E402
from ingest.settings import IngestSettings  # noqa: E402
from ingest.testing import FakeDense, FakeSparse  # noqa: E402


class FakeStore:
    """In-memory VectorStore with the same rules as the Qdrant adapter."""

    def __init__(self) -> None:
        self.collections: dict[str, dict[str, PointRecord]] = {}
        self.specs: dict[str, CollectionSpec] = {}

    def get_collection(self, name):
        if name not in self.collections:
            return None
        return CollectionInfo(name=name, points=len(self.collections[name]), spec=self.specs[name])

    def list_collections(self):
        return [self.get_collection(n) for n in sorted(self.collections)]

    def ensure_collection(self, name, spec):
        if name not in self.collections:
            self.collections[name], self.specs[name] = {}, spec
        elif self.specs[name] != spec:
            raise AppError("collection_mismatch", "different settings", 409)

    def upsert(self, collection, points):
        for p in points:
            self.collections[collection][p.id] = p
        return len(points)

    def delete_by_source(self, collection, source_id, keep_ids=None):
        keep = keep_ids or set()
        doomed = [
            pid
            for pid, p in self.collections[collection].items()
            if p.payload["source_id"] == source_id and pid not in keep
        ]
        for pid in doomed:
            del self.collections[collection][pid]
        return len(doomed)

    def count(self, collection, source_id=None):
        pts = self.collections.get(collection, {}).values()
        return sum(1 for p in pts if source_id is None or p.payload["source_id"] == source_id)

    def list_point_ids(self, collection, source_id):
        return {
            pid
            for pid, p in self.collections.get(collection, {}).items()
            if p.payload["source_id"] == source_id
        }

    def set_payloads(self, collection, updates):
        for pid, payload in updates.items():
            self.collections[collection][pid].payload.update(payload)  # merge, like Qdrant

    def points(self, collection):
        return self.collections[collection]


class Rig:
    """A service wired to fakes, with handles to inspect them."""

    def __init__(self, remote=None, mode="llamaparse", **settings) -> None:
        self.settings = IngestSettings(
            _env_file=None,
            default_collection="docs",
            chunk_size=40,
            chunk_overlap=8,
            manifest_path=":memory:",
            **settings,
        )
        self.store = FakeStore()
        self.dense = FakeDense()
        self.sparse = FakeSparse()
        self.manifest = Manifest(":memory:")
        self.remote = remote
        pdf = PdfRouter(threshold=50, tier="fast", mode=mode, remote=remote, cache=self.manifest)
        self.service = IngestService(
            settings=self.settings,
            store=self.store,
            dense=self.dense,
            sparse=self.sparse,
            parser=FileParser(pdf),
            manifest=self.manifest,
            clock=lambda: 1_700_000_000.0,
        )


@pytest.fixture
def rig_factory():
    return Rig


@pytest.fixture
def rig() -> Rig:
    return Rig()
