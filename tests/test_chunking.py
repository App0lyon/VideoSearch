import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from video_search_tool.chunking import TranscriptChunker
from video_search_tool.config import ChunkingConfig
from video_search_tool.models import TranscriptSegment


def build_segment(index: int, start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(
        segment_id=f"segment-{index:04d}",
        text=text,
        start_seconds=start,
        end_seconds=end,
    )


class TranscriptChunkerTests(unittest.TestCase):
    def test_chunker_respects_limits_and_overlap(self) -> None:
        chunker = TranscriptChunker(
            ChunkingConfig(
                max_chunk_duration_seconds=12.0,
                max_chunk_characters=80,
                overlap_segments=1,
            )
        )
        segments = [
            build_segment(0, 0.0, 5.0, "alpha"),
            build_segment(1, 5.0, 10.0, "beta"),
            build_segment(2, 10.0, 15.0, "gamma"),
            build_segment(3, 15.0, 20.0, "delta"),
        ]

        chunks = chunker.build_chunks("demo", segments)

        self.assertEqual(
            [chunk.segment_ids for chunk in chunks],
            [
                ["segment-0000", "segment-0001"],
                ["segment-0001", "segment-0002"],
                ["segment-0002", "segment-0003"],
                ["segment-0003"],
            ],
        )
        self.assertEqual(chunks[0].text, "alpha beta")
        self.assertEqual(chunks[1].start_seconds, 5.0)
        self.assertEqual(chunks[-1].end_seconds, 20.0)


if __name__ == "__main__":
    unittest.main()
