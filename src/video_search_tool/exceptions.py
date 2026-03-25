"""Project specific exceptions."""


class VideoSearchError(Exception):
    """Base exception for the package."""


class ConfigurationError(VideoSearchError):
    """Raised when configuration is invalid or incomplete."""


class DependencyUnavailableError(VideoSearchError):
    """Raised when an external dependency such as ffmpeg is missing."""


class MediaProcessingError(VideoSearchError):
    """Raised when a media file cannot be processed."""


class NvidiaApiError(VideoSearchError):
    """Raised when a NVIDIA API request fails."""
