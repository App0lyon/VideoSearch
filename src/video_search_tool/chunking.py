"""Transcript chunking utilities."""

from __future__ import annotations

from .config import ChunkingConfig
from .models import TranscriptSegment, VideoChunk


class TranscriptChunker:
    """Build semantic-search chunks from transcript segments."""

    def __init__(self, config: ChunkingConfig) -> None:
        self._config = config

    def build_chunks(self, video_id: str, segments: list[TranscriptSegment]) -> list[VideoChunk]:
        if not segments:
            return []

        chunks: list[VideoChunk] = []
        start_index = 0

        while start_index < len(segments):
            chunk_segments: list[TranscriptSegment] = []
            chunk_start = segments[start_index].start_seconds
            chunk_text_length = 0
            current_index = start_index

            while current_index < len(segments):
                segment = segments[current_index]
                candidate_duration = segment.end_seconds - chunk_start
                separator_length = 1 if chunk_segments else 0
                candidate_text_length = chunk_text_length + separator_length + len(segment.text)

                if chunk_segments and (
                    candidate_duration > self._config.max_chunk_duration_seconds
                    or candidate_text_length > self._config.max_chunk_characters
                ):
                    break

                chunk_segments.append(segment)
                chunk_text_length = candidate_text_length
                current_index += 1

            if not chunk_segments:
                chunk_segments = [segments[start_index]]
                current_index = start_index + 1

            index = len(chunks)
            chunks.append(
                VideoChunk(
                    chunk_id=f"{video_id}-chunk-{index:04d}",
                    text=" ".join(item.text.strip() for item in chunk_segments if item.text.strip()),
                    start_seconds=chunk_segments[0].start_seconds,
                    end_seconds=chunk_segments[-1].end_seconds,
                    segment_ids=[item.segment_id for item in chunk_segments],
                )
            )

            next_index = current_index - self._config.overlap_segments
            if next_index <= start_index:
                next_index = start_index + 1
            start_index = next_index

        return chunks
