"""Strict, shared output contract for optional explanation providers."""
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class Citation(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    doc_id: str = Field(min_length=1, max_length=128)
    quote: str = Field(min_length=1, max_length=6000,
                       description='Exact contiguous quotation from a supplied document')


class Claim(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    fact_id: str = Field(min_length=1, max_length=256)
    value: str | None = Field(description='Copy the required fact value exactly; retain nulls')


class FactDraft(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    claims: list[Claim] = Field(min_length=1, max_length=512,
                               description='Every supplied required fact exactly once; no new facts')
    next_action: Literal['review_candidate', 'resolve_evidence', 'retain_assessment']
    citations: list[Citation] = Field(max_length=128)
