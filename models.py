from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from constants import Provider

class TranscriptionRequest(BaseModel):
    provider: Optional[Provider] = Field(None, description="Provider para transcrição (groq ou openai)")
    model: Optional[str] = Field(None, description="Modelo específico para usar na transcrição")
    language: Optional[str] = Field(None, description="Código do idioma (ISO 639-1) ou 'auto' para detecção automática")

class TranscriptionResponse(BaseModel):
    transcription: str = Field(..., description="Texto transcrito do áudio")
    provider: str = Field(..., description="Provider usado na transcrição")
    model: str = Field(..., description="Modelo usado na transcrição")
    language: str = Field(..., description="Idioma usado na transcrição")
    filename: str = Field(..., description="Nome do arquivo original")

class ModelsResponse(BaseModel):
    providers: List[str] = Field(..., description="Lista de providers disponíveis")
    models: Dict[str, List[str]] = Field(..., description="Modelos disponíveis por provider")
    default_provider: str = Field(..., description="Provider padrão")
    default_models: Dict[str, str] = Field(..., description="Modelos padrão por provider")
    supported_languages: Dict[str, str] = Field(..., description="Idiomas suportados (código: nome)")
    default_language: str = Field(..., description="Idioma padrão")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Descrição do erro")