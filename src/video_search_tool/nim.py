"""NVIDIA API clients."""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Iterable

from .config import AppConfig
from .exceptions import ConfigurationError, DependencyUnavailableError, NvidiaApiError
from .models import TranscriptSegment


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(item * item for item in vector))
    if norm == 0:
        return vector[:]
    return [item / norm for item in vector]


def batched(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _build_multipart_body(
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
) -> tuple[bytes, str]:
    boundary = f"boundary-{uuid.uuid4().hex}"
    content_type = f"multipart/form-data; boundary={boundary}"

    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8")
        )

    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks.append(f"--{boundary}\r\n".encode("utf-8"))
    chunks.append(
        (
            f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(file_path.read_bytes())
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), content_type


class NvidiaHttpClient:
    """Minimal HTTP wrapper around NVIDIA-hosted NIM endpoints."""

    def __init__(self, config: AppConfig) -> None:
        self._timeout = config.api.timeout_seconds
        self._api_key = config.api.resolve_api_key()

    def post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        return self._execute(request)

    def post_multipart(
        self,
        url: str,
        fields: dict[str, str],
        file_field: str,
        file_path: Path,
    ) -> dict[str, Any]:
        body, content_type = _build_multipart_body(fields, file_field, file_path)
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": content_type,
            },
            method="POST",
        )
        return self._execute(request)

    def _execute(self, request: urllib.request.Request) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise NvidiaApiError(f"NVIDIA API returned HTTP {error.code}: {details}") from error
        except urllib.error.URLError as error:
            raise NvidiaApiError(f"Could not reach NVIDIA API: {error.reason}") from error


def _file_to_data_url(file_path: Path) -> str:
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


class NvidiaEmbeddingClient:
    """Create embeddings through NVIDIA NIM."""

    def __init__(self, config: AppConfig, http_client: NvidiaHttpClient | None = None) -> None:
        self._config = config
        self._http = http_client or NvidiaHttpClient(config)

    def embed_texts(self, texts: list[str], *, input_type: str = "document") -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for batch in batched(texts, self._config.embedding.batch_size):
            payload = {
                "model": self._config.models.embedding_model,
                "input": batch,
                "input_type": input_type,
                "encoding_format": "float",
            }
            response = self._http.post_json(self._config.api.embedding_url, payload)
            data = sorted(response.get("data", []), key=lambda item: int(item.get("index", 0)))
            if len(data) != len(batch):
                raise NvidiaApiError("Embedding response size did not match the request size.")
            for item in data:
                vector = [float(value) for value in item["embedding"]]
                if self._config.embedding.normalize_vectors:
                    vector = normalize_vector(vector)
                all_embeddings.append(vector)
        return all_embeddings


class NvidiaChatClient:
    """Call NVIDIA chat-completions compatible endpoints."""

    def __init__(self, config: AppConfig, http_client: NvidiaHttpClient | None = None) -> None:
        self._config = config
        self._http = http_client or NvidiaHttpClient(config)

    def complete_text(
        self,
        *,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int,
        temperature: float | None = None,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.complete_messages(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def complete_multimodal(
        self,
        *,
        prompt: str,
        image_paths: list[Path],
        model: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int,
        temperature: float | None = None,
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _file_to_data_url(image_path)},
                }
            )

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        return self.complete_messages(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def complete_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int,
        temperature: float | None = None,
    ) -> str:
        payload = {
            "model": model or self._config.models.vision_language_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self._config.generation.temperature if temperature is None else temperature,
        }
        response = self._http.post_json(self._config.api.chat_url, payload)
        choices = response.get("choices", [])
        if not choices:
            raise NvidiaApiError("Chat response did not include any choices.")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            return " ".join(str(item.get("text", "")) for item in content if isinstance(item, dict)).strip()
        return str(content).strip()


class NvidiaRerankingClient:
    """Rerank text passages through NVIDIA NIM."""

    def __init__(self, config: AppConfig, http_client: NvidiaHttpClient | None = None) -> None:
        self._config = config
        self._http = http_client or NvidiaHttpClient(config)

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []

        payload = {
            "model": self._config.models.reranking_model,
            "query": {"text": query},
            "passages": [{"text": passage} for passage in passages],
        }
        response = self._http.post_json(self._config.api.reranking_url, payload)
        items = response.get("rankings") or response.get("data") or response.get("results") or []
        if not isinstance(items, list):
            raise NvidiaApiError("Unexpected reranking response shape.")

        scores: list[float | None] = [None] * len(passages)
        for position, item in enumerate(items):
            index = int(item.get("index", position))
            value = item.get("logit", item.get("score", item.get("relevance_score", 0.0)))
            if 0 <= index < len(passages):
                scores[index] = float(value)

        return [score if score is not None else 0.0 for score in scores]


class NvidiaTranscriptionClient:
    """Transcribe extracted audio through NVIDIA NIM."""

    def __init__(self, config: AppConfig, http_client: NvidiaHttpClient | None = None) -> None:
        self._config = config
        self._http = http_client or NvidiaHttpClient(config)

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        fields = {
            "model": self._config.models.transcription_model,
            "response_format": self._config.transcription.response_format,
            "temperature": str(self._config.transcription.temperature),
        }
        if self._config.transcription.language:
            fields["language"] = self._config.transcription.language
        if self._config.transcription.prompt:
            fields["prompt"] = self._config.transcription.prompt

        response = self._http.post_multipart(
            self._config.api.transcription_url,
            fields=fields,
            file_field="file",
            file_path=audio_path,
        )
        return self._parse_response(response)

    def _parse_response(self, payload: dict[str, Any]) -> list[TranscriptSegment]:
        segments_payload = payload.get("segments")
        if isinstance(segments_payload, list) and segments_payload:
            segments: list[TranscriptSegment] = []
            for index, item in enumerate(segments_payload):
                segments.append(
                    TranscriptSegment(
                        segment_id=str(item.get("id", f"segment-{index:04d}")),
                        text=str(item.get("text", "")).strip(),
                        start_seconds=float(item.get("start", 0.0)),
                        end_seconds=float(item.get("end", item.get("start", 0.0))),
                    )
                )
            return segments

        text = str(payload.get("text", "")).strip()
        if not text:
            raise NvidiaApiError("Transcription response did not contain text or segments.")

        duration = float(payload.get("duration", 0.0))
        return [
            TranscriptSegment(
                segment_id="segment-0000",
                text=text,
                start_seconds=0.0,
                end_seconds=duration,
            )
        ]


def _duration_to_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "seconds"):
        return float(value.seconds) + (float(getattr(value, "nanos", 0)) / 1_000_000_000.0)
    if hasattr(value, "microseconds"):
        return float(value.microseconds) / 1_000_000.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _word_start_seconds(word: Any) -> float | None:
    for attribute_name in ("start_time", "start_offset", "start"):
        value = getattr(word, attribute_name, None)
        converted = _duration_to_seconds(value)
        if converted is not None:
            return converted
    return None


def _word_end_seconds(word: Any) -> float | None:
    for attribute_name in ("end_time", "end_offset", "end"):
        value = getattr(word, attribute_name, None)
        converted = _duration_to_seconds(value)
        if converted is not None:
            return converted
    return None


class RivaGrpcTranscriptionClient:
    """Transcribe extracted audio with NVIDIA Riva gRPC APIs."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._api_key = config.api.resolve_api_key()

        try:
            import riva.client  # type: ignore[import-not-found]
        except ImportError as error:
            raise DependencyUnavailableError(
                "nvidia-riva-client is required for the default Riva gRPC transcription backend."
            ) from error

        self._riva_client = riva.client

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        metadata_args = [
            ["function-id", self._config.transcription.function_id],
            ["authorization", f"Bearer {self._api_key}"],
        ]
        options = [
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ]

        auth = self._riva_client.Auth(
            uri=self._config.transcription.server_uri,
            use_ssl=self._config.transcription.use_ssl,
            metadata_args=metadata_args,
            options=options,
        )
        asr_service = self._riva_client.ASRService(auth)
        recognition_config_kwargs = {
            "language_code": self._config.transcription.language_code,
            "max_alternatives": self._config.transcription.max_alternatives,
            "profanity_filter": self._config.transcription.profanity_filter,
            "enable_automatic_punctuation": self._config.transcription.automatic_punctuation,
            "verbatim_transcripts": self._config.transcription.verbatim_transcripts,
            "enable_word_time_offsets": self._config.transcription.enable_word_time_offsets,
        }
        if self._config.transcription.riva_model_name:
            recognition_config_kwargs["model"] = self._config.transcription.riva_model_name

        recognition_config = self._riva_client.RecognitionConfig(
            **recognition_config_kwargs,
        )

        try:
            response = asr_service.offline_recognize(audio_path.read_bytes(), recognition_config)
        except Exception as error:
            raise NvidiaApiError(f"Riva offline recognition failed: {error}") from error
        return self._parse_response(response)

    def _parse_response(self, response: Any) -> list[TranscriptSegment]:
        results = getattr(response, "results", None)
        if not results:
            raise NvidiaApiError("Riva transcription returned no results.")

        segments: list[TranscriptSegment] = []
        last_end = 0.0
        for index, result in enumerate(results):
            alternatives = getattr(result, "alternatives", None) or []
            if not alternatives:
                continue
            alternative = alternatives[0]
            transcript = str(getattr(alternative, "transcript", "")).strip()
            if not transcript:
                continue

            words = list(getattr(alternative, "words", []) or [])
            start_seconds = _word_start_seconds(words[0]) if words else None
            end_seconds = _word_end_seconds(words[-1]) if words else None

            if start_seconds is None:
                start_seconds = last_end
            if end_seconds is None:
                end_seconds = start_seconds

            last_end = end_seconds
            segments.append(
                TranscriptSegment(
                    segment_id=f"segment-{index:04d}",
                    text=transcript,
                    start_seconds=float(start_seconds),
                    end_seconds=float(end_seconds),
                )
            )

        if not segments:
            raise NvidiaApiError("Riva transcription returned results without usable transcripts.")
        return segments


def build_transcription_client(config: AppConfig):
    backend = config.transcription.backend.strip().lower()
    if backend == "riva_grpc":
        return RivaGrpcTranscriptionClient(config)
    if backend == "http":
        return NvidiaTranscriptionClient(config)
    raise ConfigurationError(f"Unsupported transcription backend: {config.transcription.backend}")
