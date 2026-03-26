from pathlib import Path
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from video_search_tool.config import AppConfig, SearchConfig
from video_search_tool.models import FrameSample, TranscriptSegment, VideoChunk, VideoIndex
from video_search_tool.service import VideoIndexer, VideoSearcher


class FakeExtractor:
    def extract(self, video_path: Path, working_directory: Path) -> Path:
        output_path = working_directory / "audio.wav"
        output_path.write_bytes(b"fake")
        return output_path


class FakeTranscriber:
    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        return [
            TranscriptSegment("segment-0000", "a person enters the room", 0.0, 4.0),
            TranscriptSegment("segment-0001", "the speaker explains vector search", 4.0, 8.0),
            TranscriptSegment("segment-0002", "the demo ends after applause", 8.0, 11.0),
        ]


class SilentTranscriber:
    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        return []


class FakeEmbedder:
    def embed_texts(self, texts: list[str], *, input_type: str = "document") -> list[list[float]]:
        vectors = []
        for text in texts:
            if "vector search" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "applause" in text:
                vectors.append([0.0, 1.0, 0.0])
            elif "person" in text:
                vectors.append([0.5, 0.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


class FakeReranker:
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        scores = []
        for passage in passages:
            scores.append(10.0 if "vector search" in passage else 1.0)
        return scores


class FakeFrameExtractor:
    def extract_frames(self, video_path: Path, working_directory: Path) -> list[FrameSample]:
        return [
            FrameSample("frame-000000", 0.0, str(working_directory / "frame-000000.jpg")),
            FrameSample("frame-000001", 6.0, str(working_directory / "frame-000001.jpg")),
        ]


class FakeFrameCaptioner:
    def caption_frames(self, video_id: str, frames: list[FrameSample]) -> list[VideoChunk]:
        return [
            VideoChunk(
                f"{video_id}-visual-0000",
                "the Arc de Triomphe is visible in the scene",
                6.0,
                6.0,
                [],
                modality="visual",
                label="frame @ 6.00s",
            )
        ]


class FakeSummaryGenerator:
    def summarize(self, *, transcript_segments, visual_chunks) -> str:
        return "A short demo video with speech and a visible monument."


class ServiceTests(unittest.TestCase):
    def test_indexer_generates_index(self) -> None:
        config = AppConfig()
        indexer = VideoIndexer(config, FakeExtractor(), FakeTranscriber(), FakeEmbedder())
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "demo.mp4"
            video_path.write_bytes(b"video")

            index = indexer.index_video(video_path, metadata={"source": "unit-test"})

            self.assertEqual(index.video_id, "demo")
            self.assertEqual(index.metadata["source"], "unit-test")
            self.assertGreaterEqual(len(index.chunks), 1)
            self.assertTrue(all(chunk.embedding for chunk in index.chunks))

    def test_indexer_generates_summary_and_visual_chunks(self) -> None:
        config = AppConfig()
        indexer = VideoIndexer(
            config,
            FakeExtractor(),
            FakeTranscriber(),
            FakeEmbedder(),
            frame_extractor=FakeFrameExtractor(),
            frame_captioner=FakeFrameCaptioner(),
            summary_generator=FakeSummaryGenerator(),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "demo.mp4"
            video_path.write_bytes(b"video")

            index = indexer.index_video(video_path)

        self.assertEqual(index.summary, "A short demo video with speech and a visible monument.")
        self.assertTrue(any(chunk.modality == "visual" for chunk in index.chunks))
        self.assertTrue(any("Arc de Triomphe" in chunk.text for chunk in index.chunks))

    def test_indexer_handles_video_without_transcript(self) -> None:
        config = AppConfig()
        indexer = VideoIndexer(
            config,
            FakeExtractor(),
            SilentTranscriber(),
            FakeEmbedder(),
            frame_extractor=FakeFrameExtractor(),
            frame_captioner=FakeFrameCaptioner(),
            summary_generator=FakeSummaryGenerator(),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "silent.mp4"
            video_path.write_bytes(b"video")

            index = indexer.index_video(video_path)

        self.assertEqual(index.transcript_segments, [])
        self.assertTrue(any(chunk.modality == "visual" for chunk in index.chunks))

    def test_searcher_prefers_reranked_match(self) -> None:
        config = AppConfig(
            search=SearchConfig(candidate_pool_size=3, result_limit=2, min_semantic_score=0.0)
        )
        searcher = VideoSearcher(config, FakeEmbedder(), FakeReranker())
        index = VideoIndex(
            video_id="demo",
            video_path="demo.mp4",
            indexed_at="2026-03-25T00:00:00+00:00",
            transcript_segments=[],
            chunks=[
                VideoChunk("demo-chunk-0000", "a person enters the room", 0.0, 4.0, [], [0.5, 0.0, 0.0]),
                VideoChunk(
                    "demo-chunk-0001",
                    "the speaker explains vector search",
                    4.0,
                    8.0,
                    [],
                    [1.0, 0.0, 0.0],
                ),
                VideoChunk(
                    "demo-chunk-0002",
                    "the demo ends after applause",
                    8.0,
                    11.0,
                    [],
                    [0.0, 1.0, 0.0],
                ),
            ],
        )

        results = searcher.search(index, "vector search")

        self.assertEqual(results[0].chunk_id, "demo-chunk-0001")
        self.assertEqual(results[0].rerank_score, 10.0)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].modality, "audio")


if __name__ == "__main__":
    unittest.main()
