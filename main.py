import logging
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from constants import Provider
from models import ErrorResponse, ModelsResponse, TranscriptionResponse
from transcription_service import TranscriptionService

load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Audio Transcription API",
    description="API para transcrição de áudio usando Groq e OpenAI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

transcription_service = TranscriptionService()


@app.get("/", summary="Health Check")
async def health_check():
    logger.info("Health check requested")
    return {"status": "ok", "message": "Audio Transcription API is running"}


@app.get(
    "/models",
    response_model=ModelsResponse,
    summary="Listar modelos disponíveis",
    description="Retorna todos os providers e modelos disponíveis para transcrição",
)
async def get_available_models():
    try:
        result = transcription_service.get_available_models()
        return result
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro ao obter modelos: {str(e)}")


@app.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcrever áudio",
    description="Faz upload de um arquivo de áudio e retorna a transcrição",
    responses={
        200: {
            "model": TranscriptionResponse,
            "description": "Transcrição realizada com sucesso",
        },
        400: {
            "model": ErrorResponse,
            "description": "Erro na validação do arquivo ou parâmetros",
        },
        503: {"model": ErrorResponse, "description": "Serviço não disponível"},
        500: {"model": ErrorResponse, "description": "Erro interno"},
    },
)
async def transcribe_audio(
    file: UploadFile = File(..., description="Arquivo de áudio para transcrição"),
    provider: Optional[str] = Form(None, description="Provider: groq ou openai"),
    model: Optional[str] = Form(None, description="Modelo específico para usar"),
):
    logger.info(
        f"Starting transcription for file: {file.filename}, provider: {provider}, model: {model}"
    )

    try:
        # Validar provider se fornecido
        provider_enum = None
        if provider:
            try:
                provider_enum = Provider(provider.lower())
                logger.info(f"Using provider: {provider_enum.value}")
            except ValueError:
                available_providers = [p.value for p in Provider]
                error_msg = f"Provider inválido '{provider}'. Use: {', '.join(available_providers)}"
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)

        # Realizar transcrição
        result = await transcription_service.transcribe_audio(
            file=file, provider=provider_enum, model=model
        )

        logger.info(
            f"Transcription completed successfully for {file.filename} using {result['provider']}/{result['model']}"
        )
        return result

    except HTTPException as e:
        logger.error(f"HTTP error during transcription: {e.status_code} - {e.detail}")
        raise
    except Exception as e:
        error_msg = f"Erro interno durante transcrição: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(
        f"HTTP Exception: {exc.status_code} - {exc.detail} - Path: {request.url.path}"
    )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    error_msg = f"Erro interno não tratado: {str(exc)}"
    logger.error(
        f"Unhandled exception: {error_msg} - Path: {request.url.path}", exc_info=True
    )
    return JSONResponse(status_code=500, content={"detail": error_msg})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
