import os
from enum import Enum
from typing import Dict, List


class Provider(Enum):
    GROQ = "groq"
    OPENAI = "openai"


# Get default provider from environment or fallback to GROQ
_default_provider_str = os.getenv("DEFAULT_PROVIDER", "groq").lower()
try:
    DEFAULT_PROVIDER = Provider(_default_provider_str)
except ValueError:
    DEFAULT_PROVIDER = Provider.GROQ

# Groq models available for transcription
GROQ_MODELS = [
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "distil-whisper-large-v3-en",
]

# OpenAI models available for transcription
OPENAI_MODELS = [
    "whisper-1",
]

# Combined models dictionary
AVAILABLE_MODELS: Dict[Provider, List[str]] = {
    Provider.GROQ: GROQ_MODELS,
    Provider.OPENAI: OPENAI_MODELS,
}

# Default models for each provider (from environment or fallback)
DEFAULT_MODELS: Dict[Provider, str] = {
    Provider.GROQ: os.getenv("DEFAULT_GROQ_MODEL", "whisper-large-v3-turbo"),
    Provider.OPENAI: os.getenv("DEFAULT_OPENAI_MODEL", "whisper-1"),
}

# Supported audio formats
SUPPORTED_AUDIO_FORMATS = [
    "mp3",
    "mp4",
    "mpeg",
    "mpga",
    "m4a",
    "wav",
    "webm",
    "flac",
    "ogg",
    "opus",
]

# Maximum file size (in bytes) - 25MB
MAX_FILE_SIZE = 25 * 1024 * 1024
