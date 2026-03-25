"""Media extraction helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .config import MediaConfig
from .exceptions import DependencyUnavailableError, MediaProcessingError
from .models import FrameSample


class FFmpegAudioExtractor:
    """Extract a mono PCM WAV track from a video file."""

    def __init__(self, config: MediaConfig) -> None:
        self._config = config

    def extract(self, video_path: Path, working_directory: Path) -> Path:
        if not video_path.exists():
            raise MediaProcessingError(f"Video file does not exist: {video_path}")

        working_directory.mkdir(parents=True, exist_ok=True)
        output_path = working_directory / f"{video_path.stem}{self._config.extracted_audio_suffix}"
        command = [
            self._config.ffmpeg_binary,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            self._config.audio_codec,
            "-ar",
            str(self._config.audio_sample_rate),
            "-ac",
            str(self._config.audio_channels),
            str(output_path),
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as error:
            raise DependencyUnavailableError(
                f"ffmpeg is required but was not found on PATH: {self._config.ffmpeg_binary}"
            ) from error

        if completed.returncode != 0:
            raise MediaProcessingError(
                "ffmpeg failed to extract audio.\n"
                f"Command: {' '.join(command)}\n"
                f"stderr: {completed.stderr.strip()}"
            )

        return output_path


class FFmpegFrameExtractor:
    """Extract evenly spaced frames from a video file."""

    def __init__(self, config: MediaConfig) -> None:
        self._config = config

    def extract_frames(self, video_path: Path, working_directory: Path) -> list[FrameSample]:
        if not video_path.exists():
            raise MediaProcessingError(f"Video file does not exist: {video_path}")

        frames_directory = working_directory / "frames"
        frames_directory.mkdir(parents=True, exist_ok=True)
        output_pattern = frames_directory / f"frame-%06d{self._config.frame_image_extension}"
        command = [
            self._config.ffmpeg_binary,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps=1/{self._config.frame_interval_seconds}",
            "-q:v",
            str(self._config.frame_quality),
            "-frames:v",
            str(self._config.max_frames),
            str(output_pattern),
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as error:
            raise DependencyUnavailableError(
                f"ffmpeg is required but was not found on PATH: {self._config.ffmpeg_binary}"
            ) from error

        if completed.returncode != 0:
            raise MediaProcessingError(
                "ffmpeg failed to extract frames.\n"
                f"Command: {' '.join(command)}\n"
                f"stderr: {completed.stderr.strip()}"
            )

        frame_paths = sorted(frames_directory.glob(f"*{self._config.frame_image_extension}"))
        if not frame_paths:
            raise MediaProcessingError(f"No frames were extracted from {video_path}.")

        frame_samples: list[FrameSample] = []
        for index, frame_path in enumerate(frame_paths):
            timestamp_seconds = index * self._config.frame_interval_seconds
            frame_samples.append(
                FrameSample(
                    frame_id=f"frame-{index:06d}",
                    timestamp_seconds=timestamp_seconds,
                    image_path=str(frame_path),
                )
            )
        return frame_samples
