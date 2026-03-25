from pathlib import Path
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from video_search_tool.models import TranscriptSegment, VideoChunk, VideoIndex
from video_search_tool.storage import IndexStore


class IndexStoreTests(unittest.TestCase):
    def test_index_store_round_trip(self) -> None:
        store = IndexStore()
        index = VideoIndex(
            video_id="demo",
            video_path="demo.mp4",
            indexed_at="2026-03-25T00:00:00+00:00",
            metadata={"title": "Demo"},
            summary="Demo summary",
            transcript_segments=[
                TranscriptSegment("segment-0000", "hello world", 0.0, 1.0),
            ],
            chunks=[
                VideoChunk(
                    "demo-chunk-0000",
                    "hello world",
                    0.0,
                    1.0,
                    ["segment-0000"],
                    modality="audio",
                    label="transcript @ 0.00s",
                    embedding=[0.1, 0.2],
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = store.save(index, Path(temporary_directory) / "index.json")
            loaded = store.load(output_path)

        self.assertEqual(loaded.video_id, "demo")
        self.assertEqual(loaded.metadata, {"title": "Demo"})
        self.assertEqual(loaded.summary, "Demo summary")
        self.assertEqual(loaded.chunks[0].modality, "audio")
        self.assertEqual(loaded.chunks[0].embedding, [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
