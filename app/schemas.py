from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


Mode = Literal["precise", "balanced", "recall"]


class Entity(BaseModel):
    label: str
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    text: str
    score: float = Field(..., ge=0.0, le=1.0)


class DetectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    mode: Optional[Mode] = None
    mask: bool = False
    mask_char: str = "[REDACTED]"
    labels: Optional[list[str]] = None


class DetectMeta(BaseModel):
    model: str
    mode: Mode
    entity_count: int
    processing_ms: int
    chunks_processed: Optional[int] = None
    input_tokens: Optional[int] = None


class DetectResponse(BaseModel):
    text: str
    entities: list[Entity]
    masked_text: Optional[str] = None
    meta: DetectMeta


class MaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    mode: Optional[Mode] = None
    mask_char: str = "[REDACTED]"
    labels: Optional[list[str]] = None


class MaskResponse(BaseModel):
    masked_text: str
    entity_count: int
    processing_ms: int


class BatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    mode: Optional[Mode] = None
    mask_char: str = "[REDACTED]"
    labels: Optional[list[str]] = None


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[BatchItem] = Field(..., min_length=1)
    mask: bool = False


class BatchItemError(BaseModel):
    code: str
    message: str


class BatchItemResult(BaseModel):
    status: Literal["ok", "error"]
    entities: Optional[list[Entity]] = None
    masked_text: Optional[str] = None
    meta: Optional[dict] = None
    error: Optional[BatchItemError] = None


class BatchMeta(BaseModel):
    model: str
    batch_size: int
    processing_ms: int


class BatchResponse(BaseModel):
    results: list[BatchItemResult]
    meta: BatchMeta
