"""High-level indexing and search services."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

from .chunking import TranscriptChunker
from .config import AppConfig
from .exceptions import MediaProcessingError
from .models import SearchResult, VideoChunk, VideoIndex, utc_now_iso


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot_product = sum(l_value * r_value for l_value, r_value in zip(left, right))
    return dot_product / (left_norm * right_norm)


class VideoIndexer:
    """Create searchable indexes for video files."""

    def __init__(
        self,
        config: AppConfig,
        audio_extractor,
        transcriber,
        embedder,
        *,
        frame_extractor=None,
        frame_captioner=None,
        summary_generator=None,
    ) -> None:
        self._config = config
        self._audio_extractor = audio_extractor
        self._transcriber = transcriber
        self._embedder = embedder
        self._frame_extractor = frame_extractor
        self._frame_captioner = frame_captioner
        self._summary_generator = summary_generator
        self._chunker = TranscriptChunker(config.chunking)

    def index_video(
        self,
        video_path: str | Path,
        *,
        video_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
        working_directory: str | Path | None = None,
    ) -> VideoIndex:
        resolved_video_path = Path(video_path).resolve()
        resolved_video_id = video_id or resolved_video_path.stem
        resolved_metadata = dict(metadata or {})

        if working_directory is None:
            with tempfile.TemporaryDirectory(prefix="video-search-") as temporary_directory:
                return self._index_video_impl(
                    resolved_video_path,
                    resolved_video_id,
                    resolved_metadata,
                    Path(temporary_directory),
                )
        return self._index_video_impl(
            resolved_video_path,
            resolved_video_id,
            resolved_metadata,
            Path(working_directory),
        )

    def _index_video_impl(
        self,
        video_path: Path,
        video_id: str,
        metadata: dict[str, str],
        working_directory: Path,
    ) -> VideoIndex:
        transcript_segments = self._build_transcript_segments(video_path, working_directory)
        audio_chunks = self._chunker.build_chunks(video_id, transcript_segments)
        for chunk in audio_chunks:
            chunk.modality = "audio"
            chunk.label = f"transcript @ {chunk.start_seconds:.2f}s"

        visual_chunks = self._build_visual_chunks(video_id, video_path, working_directory)
        all_chunks = audio_chunks + visual_chunks
        embeddings = (
            self._embedder.embed_texts([chunk.text for chunk in all_chunks], input_type="document")
            if all_chunks
            else []
        )

        populated_chunks: list[VideoChunk] = []
        for chunk, embedding in zip(all_chunks, embeddings):
            populated_chunks.append(
                VideoChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    start_seconds=chunk.start_seconds,
                    end_seconds=chunk.end_seconds,
                    segment_ids=chunk.segment_ids,
                    modality=chunk.modality,
                    label=chunk.label,
                    embedding=embedding,
                )
            )

        summary = ""
        if self._summary_generator is not None:
            summary = self._summary_generator.summarize(
                transcript_segments=transcript_segments,
                visual_chunks=[chunk for chunk in populated_chunks if chunk.modality == "visual"],
            )

        return VideoIndex(
            video_id=video_id,
            video_path=str(video_path),
            indexed_at=utc_now_iso(),
            transcript_segments=transcript_segments,
            chunks=populated_chunks,
            metadata=metadata,
            summary=summary,
        )

    def _build_transcript_segments(self, video_path: Path, working_directory: Path):
        try:
            audio_path = self._audio_extractor.extract(video_path, working_directory)
        except MediaProcessingError:
            if self._config.media.continue_without_audio:
                return []
            raise
        return self._transcriber.transcribe(audio_path)

    def _build_visual_chunks(self, video_id: str, video_path: Path, working_directory: Path) -> list[VideoChunk]:
        if self._frame_extractor is None or self._frame_captioner is None:
            return []
        frames = self._frame_extractor.extract_frames(video_path, working_directory)
        return self._frame_captioner.caption_frames(video_id, frames)


class VideoSearcher:
    """Run semantic search over a previously created video index."""

    def __init__(self, config: AppConfig, embedder, reranker=None) -> None:
        self._config = config
        self._embedder = embedder
        self._reranker = reranker

    def search(
        self,
        index: VideoIndex,
        query: str,
        *,
        top_k: int | None = None,
        candidate_pool_size: int | None = None,
    ) -> list[SearchResult]:
        result_limit = top_k or self._config.search.result_limit
        pool_size = candidate_pool_size or self._config.search.candidate_pool_size
        query_embedding = self._embedder.embed_texts([query], input_type="query")[0]

        scored_candidates = []
        for chunk in index.chunks:
            semantic_score = cosine_similarity(query_embedding, chunk.embedding)
            if semantic_score >= self._config.search.min_semantic_score:
                scored_candidates.append((chunk, semantic_score))

        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        shortlisted = scored_candidates[:pool_size]
        if not shortlisted:
            return []

        rerank_scores = self._rerank(query, [chunk for chunk, _ in shortlisted])
        ordered = sorted(
            (
                (
                    chunk,
                    semantic_score,
                    rerank_scores[position],
                    rerank_scores[position] if rerank_scores[position] is not None else semantic_score,
                )
                for position, (chunk, semantic_score) in enumerate(shortlisted)
            ),
            key=lambda item: (item[3], item[1]),
            reverse=True,
        )

        results: list[SearchResult] = []
        for rank, (chunk, semantic_score, rerank_score, final_score) in enumerate(ordered[:result_limit], start=1):
            results.append(
                SearchResult(
                    rank=rank,
                    chunk_id=chunk.chunk_id,
                    start_seconds=chunk.start_seconds,
                    end_seconds=chunk.end_seconds,
                    text=chunk.text,
                    modality=chunk.modality,
                    label=chunk.label,
                    semantic_score=semantic_score,
                    rerank_score=rerank_score,
                    final_score=final_score,
                )
            )
        return results

    def _rerank(self, query: str, chunks: Iterable[VideoChunk]) -> list[float | None]:
        chunk_list = list(chunks)
        if not self._config.search.reranking_enabled or self._reranker is None:
            return [None for _ in chunk_list]
        return self._reranker.rerank(query, [chunk.text for chunk in chunk_list])
