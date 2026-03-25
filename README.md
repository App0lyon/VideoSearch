## Video Search Tool

This project provides a configurable backend for building a video assistant around one video at a time.
The implementation is organized around two folders:

- `src`: the backend package
- `tests`: a simple test suite that can run inside a `uv` environment

### What It Does

The default pipeline is:

1. Extract audio from a video with `ffmpeg`.
2. Transcribe the audio with a NVIDIA model through Riva gRPC by default.
3. Sample frames from the video and caption them with a NVIDIA vision-language model.
4. Chunk the transcript and combine it with visual captions into one searchable index.
5. Embed the searchable evidence with a NVIDIA embedding model.
6. Generate a video summary.
7. Support timestamp search and question answering over that indexed video.

This makes flows like these possible:

- "When does the Arc de Triomphe appear?"
- "Show me the timestamp where the waterfall is visible."
- "What happens in this video?"
- "Summarize the video, then answer questions about it."

### Default NVIDIA Models

The package is parameterized through `AppConfig`, and defaults to this NVIDIA stack:

- Transcription: `nvidia/parakeet-1_1b-rnnt-multilingual-asr`
- Embeddings: `nvidia/llama-nemotron-embed-1b-v2`
- Reranking: `nvidia/llama-nemotron-rerank-1b-v2`
- Vision-language summary / frame captioning / QA: `nvidia/nemotron-nano-12b-v2-vl`

You can override every endpoint, model id, chunking parameter, and search setting through a JSON config file.
The transcription layer supports two backends:

- `riva_grpc` (default), aligned with NVIDIA's Riva examples for Parakeet speech models
- `http`, if you want to target a hosted HTTP transcription endpoint instead

### Quick Start With uv

```bash
uv sync
uv run python -m unittest discover -s tests
```

Print the default config:

```bash
uv run video-search print-config
```

Create an index:

```bash
uv run video-search index --video path/to/video.mp4 --output artifacts/video.index.json
```

Search an index:

```bash
uv run video-search search --index artifacts/video.index.json --query "when does the Arc de Triomphe appear?"
```

Ask a question about an index:

```bash
uv run video-search ask --index artifacts/video.index.json --question "What happens in this video and when does the main monument appear?"
```

### JSON Configuration

The CLI accepts `--config path/to/config.json`.

Example:

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
    "frame_interval_seconds": 2.0,
    "max_frames": 120,
    "continue_without_audio": true
  },
  "transcription": {
    "backend": "riva_grpc",
    "server_uri": "grpc.nvcf.nvidia.com:443",
    "use_ssl": true,
    "function_id": "71203149-d3b7-4460-8231-1be2543a1fca",
    "language_code": "en-US",
    "max_alternatives": 1,
    "enable_word_time_offsets": true,
    "automatic_punctuation": true,
    "profanity_filter": false,
    "verbatim_transcripts": false
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
  }
}
```