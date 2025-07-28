from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from constants import Provider

class TranscriptionRequest(BaseModel):
    provider: Optional[Provider] = Field(None, description="Provider para transcrição (groq ou openai)")
    model: Optional[str] = Field(None, description="Modelo específico para usar na transcrição")

class TranscriptionResponse(BaseModel):
    transcription: str = Field(..., description="Texto transcrito do áudio")
    provider: str = Field(..., description="Provider usado na transcrição")
    model: str = Field(..., description="Modelo usado na transcrição")
    filename: str = Field(..., description="Nome do arquivo original")

class ModelsResponse(BaseModel):
    providers: List[str] = Field(..., description="Lista de providers disponíveis")
    models: Dict[str, List[str]] = Field(..., description="Modelos disponíveis por provider")
    default_provider: str = Field(..., description="Provider padrão")
    default_models: Dict[str, str] = Field(..., description="Modelos padrão por provider")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Descrição do erro")