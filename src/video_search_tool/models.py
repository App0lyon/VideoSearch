"""Domain models used by the search pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class TranscriptSegment:
    segment_id: str
    text: str
    start_seconds: float
    end_seconds: float
    source: str = "asr"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TranscriptSegment":
        return cls(
            segment_id=str(payload["segment_id"]),
            text=str(payload["text"]),
            start_seconds=float(payload["start_seconds"]),
            end_seconds=float(payload["end_seconds"]),
            source=str(payload.get("source", "asr")),
        )


@dataclass(slots=True)
class VideoChunk:
    chunk_id: str
    text: str
    start_seconds: float
    end_seconds: float
    segment_ids: list[str]
    embedding: list[float] = field(default_factory=list)
    modality: str = "audio"
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VideoChunk":
        return cls(
            chunk_id=str(payload["chunk_id"]),
            text=str(payload["text"]),
            start_seconds=float(payload["start_seconds"]),
            end_seconds=float(payload["end_seconds"]),
            segment_ids=[str(item) for item in payload.get("segment_ids", [])],
            modality=str(payload.get("modality", "audio")),
            label=str(payload.get("label", "")),
            embedding=[float(item) for item in payload.get("embedding", [])],
        )


@dataclass(slots=True)
class VideoIndex:
    video_id: str
    video_path: str
    indexed_at: str
    transcript_segments: list[TranscriptSegment]
    chunks: list[VideoChunk]
    metadata: dict[str, str] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "indexed_at": self.indexed_at,
            "metadata": dict(self.metadata),
            "summary": self.summary,
            "transcript_segments": [segment.to_dict() for segment in self.transcript_segments],
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VideoIndex":
        return cls(
            video_id=str(payload["video_id"]),
            video_path=str(payload["video_path"]),
            indexed_at=str(payload["indexed_at"]),
            metadata={str(key): str(value) for key, value in payload.get("metadata", {}).items()},
            summary=str(payload.get("summary", "")),
            transcript_segments=[
                TranscriptSegment.from_dict(item) for item in payload.get("transcript_segments", [])
            ],
            chunks=[VideoChunk.from_dict(item) for item in payload.get("chunks", [])],
        )

    @property
    def video_path_obj(self) -> Path:
        return Path(self.video_path)


@dataclass(slots=True)
class SearchResult:
    rank: int
    chunk_id: str
    start_seconds: float
    end_seconds: float
    text: str
    modality: str
    label: str
    semantic_score: float
    rerank_score: float | None
    final_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VideoAnswer:
    question: str
    answer: str
    summary: str
    evidence: list[SearchResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "summary": self.summary,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(slots=True)
class FrameSample:
    frame_id: str
    timestamp_seconds: float
    image_path: str
