"""Configuration objects and helpers."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class ApiConfig:
    api_key_env: str = "NVIDIA_API_KEY"
    embedding_url: str = "https://integrate.api.nvidia.com/v1/embeddings"
    reranking_url: str = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-1b-v2/reranking"
    chat_url: str = "https://integrate.api.nvidia.com/v1/chat/completions"
    transcription_url: str = "https://integrate.api.nvidia.com/v1/audio/transcriptions"
    timeout_seconds: float = 120.0

    def resolve_api_key(self) -> str:
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise ConfigurationError(
                f"Missing NVIDIA API key. Set the {self.api_key_env} environment variable."
            )
        return api_key


@dataclass(frozen=True, slots=True)
class ModelConfig:
    transcription_model: str = "nvidia/parakeet-1_1b-rnnt-multilingual-asr"
    embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2"
    reranking_model: str = "nvidia/llama-nemotron-rerank-1b-v2"
    vision_language_model: str = "nvidia/nemotron-nano-12b-v2-vl"


@dataclass(frozen=True, slots=True)
class MediaConfig:
    ffmpeg_binary: str = "ffmpeg"
    audio_codec: str = "pcm_s16le"
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    extracted_audio_suffix: str = ".wav"
    frame_interval_seconds: float = 2.0
    max_frames: int = 120
    frame_image_extension: str = ".jpg"
    frame_quality: int = 2
    continue_without_audio: bool = True


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    max_chunk_duration_seconds: float = 24.0
    max_chunk_characters: int = 700
    overlap_segments: int = 1


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    batch_size: int = 16
    normalize_vectors: bool = True


@dataclass(frozen=True, slots=True)
class SearchConfig:
    candidate_pool_size: int = 8
    result_limit: int = 5
    min_semantic_score: float = 0.1
    reranking_enabled: bool = True
    answer_evidence_count: int = 6


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    frame_caption_max_tokens: int = 200
    summary_max_tokens: int = 400
    answer_max_tokens: int = 500
    temperature: float = 0.2


@dataclass(frozen=True, slots=True)
class TranscriptionConfig:
    backend: str = "riva_grpc"
    server_uri: str = "grpc.nvcf.nvidia.com:443"
    use_ssl: bool = True
    function_id: str = "71203149-d3b7-4460-8231-1be2543a1fca"
    riva_model_name: str | None = None
    language_code: str = "en-US"
    max_alternatives: int = 1
    enable_word_time_offsets: bool = True
    automatic_punctuation: bool = True
    profanity_filter: bool = False
    verbatim_transcripts: bool = False
    language: str | None = None
    prompt: str | None = None
    response_format: str = "verbose_json"
    temperature: float = 0.0


@dataclass(frozen=True, slots=True)
class AppConfig:
    api: ApiConfig = field(default_factory=ApiConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    media: MediaConfig = field(default_factory=MediaConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AppConfig":
        return cls(
            api=ApiConfig(**payload.get("api", {})),
            models=ModelConfig(**payload.get("models", {})),
            media=MediaConfig(**payload.get("media", {})),
            chunking=ChunkingConfig(**payload.get("chunking", {})),
            embedding=EmbeddingConfig(**payload.get("embedding", {})),
            search=SearchConfig(**payload.get("search", {})),
            generation=GenerationConfig(**payload.get("generation", {})),
            transcription=TranscriptionConfig(**payload.get("transcription", {})),
        )

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "AppConfig":
        if config_path is None:
            return cls()
        path = Path(config_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)
