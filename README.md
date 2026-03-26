# Video Search Tool

This project is a configurable backend for building a video assistant around a single video.

The assistant can:

- extract speech when the video has audio
- describe visual moments from sampled frames
- generate a summary of the video
- search the video with natural-language queries
- answer questions about the video from retrieved evidence
- return timestamps for matched moments

## What The Project Does

The runtime builds a searchable index for one video by combining audio evidence and visual evidence.

The current pipeline is:

1. Extract audio from the video with `ffmpeg`.
2. Run speech transcription with NVIDIA Riva if audio is available.
3. Sample frames from the video with `ffmpeg`.
4. Caption those frames with a NVIDIA vision-language model.
5. Chunk the transcript into search passages.
6. Merge transcript chunks and visual captions into one searchable corpus.
7. Embed that corpus with a NVIDIA embedding model.
8. Optionally rerank search hits with a NVIDIA reranking model.
9. Generate a video-level summary.
10. Answer questions from retrieved evidence.

The important design point is that the assistant does not search the raw video directly. It searches an intermediate index made of:

- transcript chunks
- visual frame captions
- embeddings for retrieval
- a generated summary

## What It Is Capable Of

- Audio transcription when the video has usable speech.
- Visual indexing even when the video has no transcript.
- Timestamped search results.
- Search over both spoken content and visible content.
- Video-level summary generation.
- Question answering over one indexed video.
- Cached JSON indexes that can be reused later.

### What the timestamps mean

- For transcript matches, timestamps refer to the transcript chunk span.
- For visual matches, timestamps refer to the sampled frame timestamp.

### Current practical limitations

- Visual search is based on sampled frames plus generated captions, not object detection or full video grounding.
- If a frame sample misses an event, that moment may not be retrievable.
- If the vision-language caption is vague or wrong, retrieval quality drops.
- Silent videos can still be indexed visually, but transcript-based questions obviously will not help.
- QA is retrieval-augmented generation, not guaranteed factual reasoning over the raw video.
- The system currently works video-by-video. Multi-video conversational memory is not implemented as a first-class feature.

## Project Structure

The repository is organized as:

- `src/`: backend package
- `tests/`: unit tests runnable with `uv`
- `experiments/notebook/`: notebooks for live testing
- `experiments/videos/`: local sample videos for experiments

The main package lives under `src/video_search_tool`.

## Core Components

### Configuration

`AppConfig` centralizes the whole runtime configuration:

- API endpoints
- model ids
- media extraction parameters
- chunking parameters
- embedding settings
- search settings
- generation settings
- transcription backend settings

File: [config.py](c:/Users/prxch/Desktop/VideoSearch/src/video_search_tool/config.py)

### Media extraction

Two `ffmpeg`-based extractors are used:

- audio extraction for transcription
- frame extraction for visual indexing

File: [media.py](c:/Users/prxch/Desktop/VideoSearch/src/video_search_tool/media.py)

### NVIDIA clients

The NVIDIA integration layer handles:

- embeddings
- reranking
- multimodal chat completions
- Riva transcription

File: [nim.py](c:/Users/prxch/Desktop/VideoSearch/src/video_search_tool/nim.py)

### Indexing and search

`VideoIndexer` creates the final `VideoIndex`.

`VideoSearcher` performs semantic retrieval and optional reranking over the indexed chunks.

File: [service.py](c:/Users/prxch/Desktop/VideoSearch/src/video_search_tool/service.py)

### Assistant layer

The assistant layer adds:

- frame captioning
- summary generation
- question answering using retrieved evidence

File: [assistant.py](c:/Users/prxch/Desktop/VideoSearch/src/video_search_tool/assistant.py)

### Persistence

Indexes are saved as JSON and reloaded later.

File: [storage.py](c:/Users/prxch/Desktop/VideoSearch/src/video_search_tool/storage.py)

## Data Model

The persisted video index contains:

- `video_id`
- `video_path`
- `indexed_at`
- `metadata`
- `summary`
- `transcript_segments`
- `chunks`

Each chunk is either:

- an `audio` chunk from transcript text
- a `visual` chunk from a frame caption

Each chunk also stores:

- `start_seconds`
- `end_seconds`
- `label`
- `embedding`

This is why one search query can return either a spoken moment or a visual moment.

## Default NVIDIA Stack

The default configuration currently uses:

- Transcription: `nvidia/parakeet-1_1b-rnnt-multilingual-asr`
- Embeddings: `nvidia/llama-nemotron-embed-1b-v2`
- Reranking: `nvidia/llama-nemotron-rerank-1b-v2`
- Vision-language chat / summary / frame captioning / QA: `nvidia/nemotron-nano-12b-v2-vl`

### Important API details

- The Riva path is selected with the hosted `function-id`.
- The embedding model is asymmetric.
- Indexed chunks are embedded with `input_type="passage"`.
- User queries are embedded with `input_type="query"`.

## Installation

### Python environment

This project is designed for `uv`.

Install the project and the notebook/dev tools with:

```bash
uv sync --all-groups
```

### ffmpeg

`ffmpeg` must be installed and available on `PATH`.

On Windows, a practical install path is:

```powershell
winget install -e --id Gyan.FFmpeg
```

Verify it with:

```powershell
ffmpeg -version
```

### Environment variables

The code expects a NVIDIA API key in:

```bash
NVIDIA_API_KEY
```

Example PowerShell session:

```powershell
$env:NVIDIA_API_KEY="your-key-here"
```

## Quick Start

### Run tests

```bash
uv run python -m unittest discover -s tests -v
```

### Print the default configuration

```bash
uv run video-search print-config
```

### Index a video

```bash
uv run video-search index --video path/to/video.mp4 --output artifacts/video.index.json
```

This will:

- extract media evidence
- create embeddings
- build a summary
- store the JSON index

### Search a video

```bash
uv run video-search search --index artifacts/video.index.json --query "when does the Arc de Triomphe appear?"
```

### Ask a question about a video

```bash
uv run video-search ask --index artifacts/video.index.json --question "What happens in this video and when does the main monument appear?"
```

## Notebook Workflow

The main live testing notebook is:

- [01_full_feature_test.ipynb](c:/Users/prxch/Desktop/VideoSearch/experiments/notebook/01_full_feature_test.ipynb)

It is meant to test the real backend end to end on local videos.

The notebook covers:

- path setup
- prerequisite checks
- index creation and cache reuse
- summary inspection
- transcript preview
- search tests
- QA tests
- manual playground cells

Indexes generated by the notebook are cached under:

- `experiments/notebook/artifacts/indexes`

Temporary extraction files are written under:

- `experiments/notebook/artifacts/work`

## Configuration

The CLI accepts a JSON config through `--config`.

Every major part of the runtime is parameterized:

- API endpoints
- selected model ids
- frame sampling rate
- maximum frame count
- whether indexing should continue without audio
- transcript chunk size
- embedding batch size
- search result limits
- reranking behavior
- summary / QA generation token budgets
- Riva backend settings

Minimal example:

```json
{
  "api": {
    "api_key_env": "NVIDIA_API_KEY",
    "embedding_url": "https://integrate.api.nvidia.com/v1/embeddings",
    "reranking_url": "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-1b-v2/reranking",
    "chat_url": "https://integrate.api.nvidia.com/v1/chat/completions",
    "transcription_url": "https://integrate.api.nvidia.com/v1/audio/transcriptions",
    "timeout_seconds": 120.0
  },
  "models": {
    "transcription_model": "nvidia/parakeet-1_1b-rnnt-multilingual-asr",
    "embedding_model": "nvidia/llama-nemotron-embed-1b-v2",
    "reranking_model": "nvidia/llama-nemotron-rerank-1b-v2",
    "vision_language_model": "nvidia/nemotron-nano-12b-v2-vl"
  },
  "media": {
    "ffmpeg_binary": "ffmpeg",
    "audio_sample_rate": 16000,
    "frame_interval_seconds": 2.0,
    "max_frames": 120,
    "continue_without_audio": true
  },
  "chunking": {
    "max_chunk_duration_seconds": 24.0,
    "max_chunk_characters": 700,
    "overlap_segments": 1
  },
  "search": {
    "candidate_pool_size": 8,
    "result_limit": 5,
    "min_semantic_score": 0.1,
    "reranking_enabled": true,
    "answer_evidence_count": 6
  },
  "generation": {
    "frame_caption_max_tokens": 200,
    "summary_max_tokens": 400,
    "answer_max_tokens": 500,
    "temperature": 0.2
  },
  "transcription": {
    "backend": "riva_grpc",
    "server_uri": "grpc.nvcf.nvidia.com:443",
    "use_ssl": true,
    "function_id": "71203149-d3b7-4460-8231-1be2543a1fca",
    "riva_model_name": null,
    "language_code": "en-US",
    "max_alternatives": 1,
    "enable_word_time_offsets": true,
    "automatic_punctuation": true,
    "profanity_filter": false,
    "verbatim_transcripts": false
  }
}
```

## How Search Works

When you run a query:

1. The query is embedded with the NVIDIA embedding model as a `query`.
2. Stored chunks are already embedded as `passage`.
3. Cosine similarity produces an initial candidate set.
4. Optional reranking reorders the best candidates.
5. Results are returned with text, modality, and timestamps.

Search can retrieve:

- transcript matches
- visual frame-caption matches

## How Question Answering Works

When you ask a question:

1. The system first searches the indexed chunks.
2. The best hits become the evidence set.
3. The assistant prompt includes:
   - the video summary
   - the retrieved evidence
   - the user question
4. The NVIDIA chat model generates the final answer.

That means QA quality depends directly on:

- frame sampling quality
- frame caption quality
- transcript quality
- retrieval quality

## Testing

The repository includes unit tests for:

- chunking
- storage round-trip
- indexing behavior
- silent-video handling
- multimodal indexing
- assistant QA behavior

Run them with:

```bash
uv run python -m unittest discover -s tests -v
```

Main test files:

- [test_chunking.py](c:/Users/prxch/Desktop/VideoSearch/tests/test_chunking.py)
- [test_service.py](c:/Users/prxch/Desktop/VideoSearch/tests/test_service.py)
- [test_storage.py](c:/Users/prxch/Desktop/VideoSearch/tests/test_storage.py)
- [test_assistant.py](c:/Users/prxch/Desktop/VideoSearch/tests/test_assistant.py)

## Known Constraints

- This is a backend-oriented project, not a finished UI chatbot application.
- There is no persistent conversation memory layer beyond the current indexed video.
- Visual retrieval is approximate because it relies on sampled frames and generated captions.
- Some NVIDIA endpoints are strict about request shape; the implementation reflects the current API contracts used by this project.
- End-to-end quality depends on the selected model endpoints remaining compatible with the configured request formats.

## Summary

This project is currently a multimodal video indexing and retrieval backend with a QA layer on top.

It is capable of:

- turning one video into a searchable index
- handling videos with or without useful audio
- searching for visible moments and spoken moments
- generating summaries
- answering grounded questions with timestamps

It is not yet a perfect video-grounding system, but it is a solid modular base for a chatbot-like assistant over local videos.
