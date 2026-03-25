"""High-level assistant features: frame captioning, summary generation, and QA."""

from __future__ import annotations

from pathlib import Path

from .config import AppConfig
from .models import FrameSample, SearchResult, TranscriptSegment, VideoAnswer, VideoChunk, VideoIndex


def _truncate(text: str, max_characters: int) -> str:
    if len(text) <= max_characters:
        return text
    return text[: max_characters - 3].rstrip() + "..."


class VideoFrameCaptioner:
    """Describe extracted video frames so they become searchable."""

    def __init__(self, config: AppConfig, chat_client) -> None:
        self._config = config
        self._chat_client = chat_client

    def caption_frames(self, video_id: str, frames: list[FrameSample]) -> list[VideoChunk]:
        chunks: list[VideoChunk] = []
        for index, frame in enumerate(frames):
            caption = self._chat_client.complete_multimodal(
                prompt=(
                    "Describe this video frame for search and QA. "
                    "Mention landmarks, monuments, people, actions, scenery, and visible text. "
                    "Be concise and factual."
                ),
                image_paths=[Path(frame.image_path)],
                max_tokens=self._config.generation.frame_caption_max_tokens,
            )
            chunks.append(
                VideoChunk(
                    chunk_id=f"{video_id}-visual-{index:04d}",
                    text=caption,
                    start_seconds=frame.timestamp_seconds,
                    end_seconds=frame.timestamp_seconds,
                    segment_ids=[],
                    modality="visual",
                    label=f"frame @ {frame.timestamp_seconds:.2f}s",
                )
            )
        return chunks


class VideoSummaryGenerator:
    """Generate a video-level summary from transcript and visual evidence."""

    def __init__(self, config: AppConfig, chat_client) -> None:
        self._config = config
        self._chat_client = chat_client

    def summarize(
        self,
        *,
        transcript_segments: list[TranscriptSegment],
        visual_chunks: list[VideoChunk],
    ) -> str:
        transcript_text = " ".join(segment.text.strip() for segment in transcript_segments if segment.text.strip())
        visual_notes = "\n".join(
            f"- {chunk.label}: {chunk.text.strip()}" for chunk in visual_chunks if chunk.text.strip()
        )
        if not transcript_text and not visual_notes:
            return ""

        prompt = (
            "Summarize this video for a search assistant.\n\n"
            f"Transcript:\n{_truncate(transcript_text or '<no transcript>', 4000)}\n\n"
            f"Visual evidence:\n{_truncate(visual_notes or '<no visual evidence>', 4000)}\n\n"
            "Write a concise summary that captures what happens in the video and what can be searched for."
        )
        return self._chat_client.complete_text(
            prompt=prompt,
            max_tokens=self._config.generation.summary_max_tokens,
        )


class VideoAssistant:
    """Answer natural-language questions about a loaded video index."""

    def __init__(self, config: AppConfig, searcher, chat_client) -> None:
        self._config = config
        self._searcher = searcher
        self._chat_client = chat_client

    def answer(self, index: VideoIndex, question: str) -> VideoAnswer:
        evidence = self._searcher.search(
            index,
            question,
            top_k=self._config.search.answer_evidence_count,
        )
        evidence_block = self._build_evidence_block(evidence)
        prompt = (
            "You answer questions about one video.\n"
            "Use the provided evidence only. If the answer is uncertain, say so clearly.\n"
            "If the user asks for a moment in the video, include the best timestamp.\n\n"
            f"Video summary:\n{index.summary or '<no summary available>'}\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            f"Question: {question}"
        )
        answer = self._chat_client.complete_text(
            prompt=prompt,
            max_tokens=self._config.generation.answer_max_tokens,
        )
        return VideoAnswer(
            question=question,
            answer=answer,
            summary=index.summary,
            evidence=evidence,
        )

    @staticmethod
    def _build_evidence_block(results: list[SearchResult]) -> str:
        if not results:
            return "<no evidence found>"
        return "\n".join(
            (
                f"- [{result.modality}] {result.start_seconds:.2f}s -> {result.end_seconds:.2f}s "
                f"({result.label or result.chunk_id}): {result.text}"
            )
            for result in results
        )
