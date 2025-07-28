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

# Supported language codes for transcription
# Format: ISO 639-1 language codes
SUPPORTED_LANGUAGES = {
    "af": "afrikaans",
    "ar": "arabic", 
    "hy": "armenian",
    "az": "azerbaijani",
    "be": "belarusian",
    "bs": "bosnian",
    "bg": "bulgarian",
    "ca": "catalan",
    "zh": "chinese",
    "hr": "croatian",
    "cs": "czech",
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "gl": "galician",
    "de": "german",
    "el": "greek",
    "he": "hebrew",
    "hi": "hindi",
    "hu": "hungarian",
    "is": "icelandic",
    "id": "indonesian",
    "it": "italian",
    "ja": "japanese",
    "kn": "kannada",
    "kk": "kazakh",
    "ko": "korean",
    "lv": "latvian",
    "lt": "lithuanian",
    "mk": "macedonian",
    "ms": "malay",
    "mr": "marathi",
    "mi": "maori",
    "ne": "nepali",
    "no": "norwegian",
    "fa": "persian",
    "pl": "polish",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "sr": "serbian",
    "sk": "slovak",
    "sl": "slovenian",
    "es": "spanish",
    "sw": "swahili",
    "sv": "swedish",
    "tl": "tagalog",
    "ta": "tamil",
    "th": "thai",
    "tr": "turkish",
    "uk": "ukrainian",
    "ur": "urdu",
    "vi": "vietnamese",
    "cy": "welsh"
}

# Default language for transcription
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "auto")  # Auto-detect language
