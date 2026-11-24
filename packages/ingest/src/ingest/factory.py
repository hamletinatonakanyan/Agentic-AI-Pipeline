"""Builds the real thing from settings. Nothing here connects or downloads at build time:
Qdrant is contacted on first use, and fastembed models load on first embed."""

from dataclasses import dataclass

from qdrant_client import QdrantClient

from ingest.embedding import FastEmbedDense, FastEmbedSparse, resolve_prefixes
from ingest.jobs import JobRunner
from ingest.manifest import Manifest
from ingest.parsers import FileParser
from ingest.parsers.llamaparse import LlamaParseClient
from ingest.parsers.pdf import PdfRouter
from ingest.protocols import DenseEmbedder, SparseEmbedder
from ingest.service import IngestService
from ingest.settings import IngestSettings
from ingest.store import QdrantVectorStore


@dataclass
class Ingest:
    """What the apps hold: the service, the job runner, and the manifest to close at shutdown."""

    service: IngestService
    jobs: JobRunner
    manifest: Manifest

    def close(self) -> None:
        self.jobs.shutdown()
        self.manifest.close()


def build_ingest(
    settings: IngestSettings,
    *,
    client: QdrantClient | None = None,
    dense: DenseEmbedder | None = None,
    sparse: SparseEmbedder | None = None,
    inline_jobs: bool = False,
) -> Ingest:
    """The real wiring. Tests inject an in-memory `client`, fake embedders and `inline_jobs`."""
    manifest = Manifest(settings.manifest_path)

    remote = None
    reason = "LLAMA_CLOUD_API_KEY is not set"
    if settings.pdf_parser == "llamaparse" and settings.llama_cloud_api_key:
        remote = LlamaParseClient(
            api_key=settings.llama_cloud_api_key,
            tier=settings.llamaparse_tier,
            base_url=settings.llamaparse_base_url,
            timeout_s=settings.llamaparse_timeout_s,
        )
    pdf = PdfRouter(
        threshold=settings.scanned_text_threshold,
        tier=settings.llamaparse_tier,
        mode=settings.pdf_parser,
        remote=remote,
        cache=manifest,
        remote_unavailable_reason=reason,
    )

    query_prefix, doc_prefix = resolve_prefixes(
        settings.embed_model, settings.query_prefix, settings.doc_prefix
    )
    client = client or QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=30,
        check_compatibility=False,  # do not contact the server while building
    )
    service = IngestService(
        settings=settings,
        store=QdrantVectorStore(client),
        dense=dense
        or FastEmbedDense(
            settings.embed_model,
            query_prefix=query_prefix,
            doc_prefix=doc_prefix,
            batch_size=settings.embed_batch_size,
        ),
        sparse=sparse or FastEmbedSparse(settings.sparse_model),
        parser=FileParser(pdf),
        manifest=manifest,
    )
    return Ingest(service=service, jobs=JobRunner(manifest, inline=inline_jobs), manifest=manifest)
