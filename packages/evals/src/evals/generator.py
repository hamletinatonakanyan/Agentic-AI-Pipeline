"""Draft evaluation questions from chunks with the LLM. Every draft needs human review."""

import uuid

from pydantic import BaseModel

from evals.models import DatasetItem, Evidence
from evals.protocols import ChunkRef
from llm import LLM

SYSTEM = (
    "You write test questions for a document search system. Write in the same language as the "
    "passage. Return only what is asked."
)


class _Draft(BaseModel):
    question: str
    answer: str


class QuestionGenerator:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def draft(self, chunk: ChunkRef) -> DatasetItem:
        prompt = (
            "Read the PASSAGE. Write one question that a person could ask without having seen "
            "it, and that the passage fully answers. Do not mention 'the passage' or 'the text'. "
            "Then write the answer using only the passage, in one or two sentences.\n\n"
            f"PASSAGE:\n{chunk.text}"
        )
        draft = self._llm.complete_json(prompt, _Draft, system=SYSTEM)
        return DatasetItem(
            id=uuid.uuid4().hex[:10],
            question=draft.question.strip(),
            reference_answer=draft.answer.strip(),
            relevant_chunk_ids=[chunk.chunk_id],
            status="draft",
            evidence=[
                Evidence(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source_name=chunk.source_name,
                    page=chunk.page,
                )
            ],
        )
