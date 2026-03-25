"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys

from .assistant import VideoAssistant, VideoFrameCaptioner, VideoSummaryGenerator
from .config import AppConfig
from .media import FFmpegAudioExtractor, FFmpegFrameExtractor
from .nim import (
    NvidiaChatClient,
    NvidiaEmbeddingClient,
    NvidiaRerankingClient,
    build_transcription_client,
)
from .service import VideoIndexer, VideoSearcher
from .storage import IndexStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index videos and run text search over them.")
    parser.add_argument("--config", help="Path to a JSON configuration file.", default=None)

    subparsers = parser.add_subparsers(dest="command", required=True)

    dump_config_parser = subparsers.add_parser("print-config", help="Print the default configuration.")
    dump_config_parser.set_defaults(handler=handle_print_config)

    index_parser = subparsers.add_parser("index", help="Create a searchable index from a video file.")
    index_parser.add_argument("--video", required=True, help="Path to the source video file.")
    index_parser.add_argument("--output", required=True, help="Path to the generated JSON index.")
    index_parser.add_argument("--video-id", default=None, help="Optional stable identifier for the video.")
    index_parser.add_argument(
        "--work-dir",
        default=None,
        help="Optional working directory used for extracted audio files.",
    )
    index_parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Metadata entries in key=value form. Can be repeated.",
    )
    index_parser.set_defaults(handler=handle_index)

    search_parser = subparsers.add_parser("search", help="Search an existing video index.")
    search_parser.add_argument("--index", required=True, help="Path to the JSON index.")
    search_parser.add_argument("--query", required=True, help="Natural-language search query.")
    search_parser.add_argument("--top-k", type=int, default=None, help="Override the number of results.")
    search_parser.add_argument(
        "--candidate-pool",
        type=int,
        default=None,
        help="Override the number of semantic candidates before reranking.",
    )
    search_parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text.")
    search_parser.set_defaults(handler=handle_search)

    ask_parser = subparsers.add_parser("ask", help="Ask a question about an indexed video.")
    ask_parser.add_argument("--index", required=True, help="Path to the JSON index.")
    ask_parser.add_argument("--question", required=True, help="Question to answer about the video.")
    ask_parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text.")
    ask_parser.set_defaults(handler=handle_ask)
    return parser


def parse_metadata(raw_entries: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for entry in raw_entries:
        key, separator, value = entry.partition("=")
        if not separator:
            raise ValueError(f"Invalid metadata entry: {entry!r}. Expected key=value.")
        metadata[key.strip()] = value.strip()
    return metadata


def build_runtime(config: AppConfig) -> tuple[VideoIndexer, VideoSearcher, IndexStore]:
    extractor = FFmpegAudioExtractor(config.media)
    frame_extractor = FFmpegFrameExtractor(config.media)
    embedder = NvidiaEmbeddingClient(config)
    chat_client = NvidiaChatClient(config)
    transcriber = build_transcription_client(config)
    reranker = NvidiaRerankingClient(config) if config.search.reranking_enabled else None
    frame_captioner = VideoFrameCaptioner(config, chat_client)
    summary_generator = VideoSummaryGenerator(config, chat_client)
    indexer = VideoIndexer(
        config,
        extractor,
        transcriber,
        embedder,
        frame_extractor=frame_extractor,
        frame_captioner=frame_captioner,
        summary_generator=summary_generator,
    )
    searcher = VideoSearcher(config, embedder, reranker=reranker)
    return indexer, searcher, IndexStore()


def build_assistant_runtime(config: AppConfig) -> tuple[VideoIndexer, VideoSearcher, VideoAssistant, IndexStore]:
    indexer, searcher, store = build_runtime(config)
    assistant = VideoAssistant(config, searcher, NvidiaChatClient(config))
    return indexer, searcher, assistant, store


def handle_print_config(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    print(config.to_json())
    return 0


def handle_index(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    indexer, _, store = build_runtime(config)
    metadata = parse_metadata(args.metadata)
    index = indexer.index_video(
        args.video,
        video_id=args.video_id,
        metadata=metadata,
        working_directory=args.work_dir,
    )
    output_path = store.save(index, args.output)
    print(f"Index saved to {output_path}")
    if index.summary:
        print()
        print("Summary:")
        print(index.summary)
    return 0


def handle_search(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    _, searcher, store = build_runtime(config)
    index = store.load(args.index)
    results = searcher.search(
        index,
        args.query,
        top_k=args.top_k,
        candidate_pool_size=args.candidate_pool,
    )

    if args.json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
        return 0

    if not results:
        print("No matching segments found.")
        return 0

    for result in results:
        print(
            f"[{result.rank}] [{result.modality}] {result.start_seconds:.2f}s - {result.end_seconds:.2f}s "
            f"(semantic={result.semantic_score:.3f}, rerank={result.rerank_score}, final={result.final_score:.3f})"
        )
        if result.label:
            print(result.label)
        print(result.text)
        print()
    return 0


def handle_ask(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    _, _, assistant, store = build_assistant_runtime(config)
    index = store.load(args.index)
    answer = assistant.answer(index, args.question)

    if args.json:
        print(json.dumps(answer.to_dict(), indent=2))
        return 0

    print(answer.answer)
    if answer.evidence:
        print()
        print("Evidence:")
        for item in answer.evidence:
            print(
                f"- [{item.modality}] {item.start_seconds:.2f}s - {item.end_seconds:.2f}s "
                f"{item.label or item.chunk_id}"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
