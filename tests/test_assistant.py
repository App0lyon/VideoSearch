from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from video_search_tool.assistant import VideoAssistant
from video_search_tool.config import AppConfig, SearchConfig
from video_search_tool.models import VideoChunk, VideoIndex
from video_search_tool.service import VideoSearcher


class FakeEmbedder:
    def embed_texts(self, texts: list[str], *, input_type: str = "document") -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "arc de triomphe" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "vector search" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


class FakeReranker:
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        return [50.0 if "arc de triomphe" in passage.lower() else 1.0 for passage in passages]


class FakeChatClient:
    def complete_text(self, *, prompt: str, max_tokens: int) -> str:
        if "Arc de Triomphe" in prompt or "arc de triomphe" in prompt:
            return "The Arc de Triomphe appears at about 12.00s."
        return "I cannot determine the answer."


class AssistantTests(unittest.TestCase):
    def test_assistant_uses_retrieved_evidence(self) -> None:
        config = AppConfig(
            search=SearchConfig(
                candidate_pool_size=4,
                result_limit=4,
                min_semantic_score=0.0,
                reranking_enabled=True,
                answer_evidence_count=3,
            )
        )
        searcher = VideoSearcher(config, FakeEmbedder(), FakeReranker())
        assistant = VideoAssistant(config, searcher, FakeChatClient())
        index = VideoIndex(
            video_id="paris",
            video_path="paris.mp4",
            indexed_at="2026-03-25T00:00:00+00:00",
            transcript_segments=[],
            chunks=[
                VideoChunk(
                    "paris-visual-0000",
                    "The Arc de Triomphe appears in the center of the frame.",
                    12.0,
                    12.0,
                    [],
                    modality="visual",
                    label="frame @ 12.00s",
                    embedding=[1.0, 0.0, 0.0],
                ),
                VideoChunk(
                    "paris-audio-0000",
                    "The narrator talks about the city.",
                    1.0,
                    3.0,
                    [],
                    modality="audio",
                    label="transcript @ 1.00s",
                    embedding=[0.0, 0.0, 1.0],
                ),
            ],
            summary="A travel video in Paris.",
        )

        answer = assistant.answer(index, "When does the Arc de Triomphe appear?")

        self.assertIn("12.00s", answer.answer)
        self.assertEqual(answer.evidence[0].modality, "visual")


if __name__ == "__main__":
    unittest.main()
