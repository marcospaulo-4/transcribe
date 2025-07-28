import logging
import os
import tempfile
from typing import Any, Dict, Optional

import aiofiles
from fastapi import HTTPException, UploadFile
from groq import AsyncGroq
from openai import AsyncOpenAI

from constants import (
    AVAILABLE_MODELS,
    DEFAULT_LANGUAGE,
    DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    MAX_FILE_SIZE,
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_LANGUAGES,
    Provider,
)


class TranscriptionService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.groq_client = None
        self.openai_client = None
        self._initialize_clients()

    def _initialize_clients(self) -> None:
        groq_api_key = os.getenv("GROQ_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if groq_api_key:
            try:
                self.groq_client = AsyncGroq(api_key=groq_api_key)
                self.logger.info("Groq client inicializado com sucesso")
            except Exception as e:
                self.logger.error(f"Erro ao inicializar cliente Groq: {str(e)}")
        else:
            self.logger.warning("GROQ_API_KEY não encontrada nas variáveis de ambiente")

        if openai_api_key:
            try:
                self.openai_client = AsyncOpenAI(api_key=openai_api_key)
                self.logger.info("OpenAI client inicializado com sucesso")
            except Exception as e:
                self.logger.error(f"Erro ao inicializar cliente OpenAI: {str(e)}")
        else:
            self.logger.warning(
                "OPENAI_API_KEY não encontrada nas variáveis de ambiente"
            )

    def _validate_file(self, file: UploadFile) -> None:
        self.logger.info(f"Validando arquivo: {file.filename}")

        if not file.filename:
            error_msg = "Arquivo sem nome"
            self.logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in SUPPORTED_AUDIO_FORMATS:
            error_msg = f"Formato '{file_extension}' não suportado. Formatos aceitos: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
            self.logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        if file.size and file.size > MAX_FILE_SIZE:
            size_mb = file.size / (1024 * 1024)
            max_size_mb = MAX_FILE_SIZE // (1024 * 1024)
            error_msg = f"Arquivo muito grande ({size_mb:.1f}MB). Tamanho máximo: {max_size_mb}MB"
            self.logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        self.logger.info(f"Arquivo válido: {file.filename} ({file.size} bytes)")

    def _validate_language(self, language: Optional[str] = None) -> str:
        """Valida o código de idioma fornecido."""
        selected_language = language or DEFAULT_LANGUAGE
        
        if selected_language == "auto":
            self.logger.info("Usando detecção automática de idioma")
            return selected_language
        
        if selected_language not in SUPPORTED_LANGUAGES:
            supported_codes = ", ".join(list(SUPPORTED_LANGUAGES.keys())[:10]) + "..."
            error_msg = f"Código de idioma '{selected_language}' não suportado. Códigos suportados: {supported_codes}, 'auto'"
            self.logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        language_name = SUPPORTED_LANGUAGES[selected_language]
        self.logger.info(f"Idioma válido: {selected_language} ({language_name})")
        return selected_language

    def _validate_provider_and_model(
        self, provider: Provider, model: Optional[str] = None
    ) -> str:
        self.logger.info(f"Validando provider: {provider.value}, model: {model}")

        if provider == Provider.GROQ and not self.groq_client:
            error_msg = "Groq API não configurada. Verifique a variável GROQ_API_KEY"
            self.logger.error(error_msg)
            raise HTTPException(status_code=503, detail=error_msg)

        if provider == Provider.OPENAI and not self.openai_client:
            error_msg = (
                "OpenAI API não configurada. Verifique a variável OPENAI_API_KEY"
            )
            self.logger.error(error_msg)
            raise HTTPException(status_code=503, detail=error_msg)

        selected_model = model or DEFAULT_MODELS[provider]

        if selected_model not in AVAILABLE_MODELS[provider]:
            available_models = ", ".join(AVAILABLE_MODELS[provider])
            error_msg = f"Modelo '{selected_model}' não disponível para {provider.value}. Modelos disponíveis: {available_models}"
            self.logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        self.logger.info(
            f"Provider e modelo válidos: {provider.value}/{selected_model}"
        )
        return selected_model

    async def _transcribe_with_groq(self, audio_file_path: str, model: str, language: str = "auto") -> str:
        if not self.groq_client:
            error_msg = "Groq client não inicializado"
            self.logger.error(error_msg)
            raise HTTPException(status_code=503, detail=error_msg)

        try:
            self.logger.info(f"Iniciando transcrição com Groq usando modelo: {model}, idioma: {language}")
            
            transcription_params = {
                "file": None,  # será definido no with
                "model": model,
                "response_format": "text",
            }
            
            # Apenas adicionar language se não for auto-detect
            if language != "auto":
                transcription_params["language"] = language
            
            with open(audio_file_path, "rb") as audio_file:
                transcription_params["file"] = audio_file
                transcription = await self.groq_client.audio.transcriptions.create(
                    **transcription_params
                )

            # O Groq retorna uma string diretamente quando response_format="text"
            # ou um objeto com .text quando response_format é outro formato
            if isinstance(transcription, str):
                result_text = transcription
            else:
                result_text = transcription.text if hasattr(transcription, 'text') else str(transcription)

            self.logger.info(
                f"Transcrição Groq concluída. Tamanho do texto: {len(result_text)} caracteres"
            )
            return result_text

        except Exception as e:
            error_msg = f"Erro na transcrição com Groq: {str(e)}"
            self.logger.error(f"{error_msg}. Tipo da resposta: {type(transcription) if 'transcription' in locals() else 'Não definido'}", exc_info=True)
            raise HTTPException(status_code=500, detail=error_msg)

    async def _transcribe_with_openai(self, audio_file_path: str, model: str, language: str = "auto") -> str:
        if not self.openai_client:
            error_msg = "OpenAI client não inicializado"
            self.logger.error(error_msg)
            raise HTTPException(status_code=503, detail=error_msg)

        try:
            self.logger.info(f"Iniciando transcrição com OpenAI usando modelo: {model}, idioma: {language}")
            
            transcription_params = {
                "file": None,  # será definido no with
                "model": model,
                "response_format": "text",
            }
            
            # Apenas adicionar language se não for auto-detect
            if language != "auto":
                transcription_params["language"] = language
            
            with open(audio_file_path, "rb") as audio_file:
                transcription_params["file"] = audio_file
                transcription = await self.openai_client.audio.transcriptions.create(
                    **transcription_params
                )

            # O OpenAI retorna uma string diretamente quando response_format="text"
            # ou um objeto com .text quando response_format é outro formato
            if isinstance(transcription, str):
                result_text = transcription
            else:
                result_text = transcription.text
            self.logger.info(
                f"Transcrição OpenAI concluída. Tamanho do texto: {len(result_text)} caracteres"
            )
            return result_text

        except Exception as e:
            error_msg = f"Erro na transcrição com OpenAI: {str(e)}"
            self.logger.error(f"{error_msg}. Tipo da resposta: {type(transcription) if 'transcription' in locals() else 'Não definido'}", exc_info=True)
            raise HTTPException(status_code=500, detail=error_msg)

    async def transcribe_audio(
        self,
        file: UploadFile,
        provider: Optional[Provider] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        selected_provider = provider or DEFAULT_PROVIDER
        self.logger.info(
            f"Iniciando processo de transcrição para {file.filename} com {selected_provider.value}"
        )

        # Validar arquivo
        self._validate_file(file)

        # Validar provider e modelo
        selected_model = self._validate_provider_and_model(selected_provider, model)
        
        # Validar idioma
        selected_language = self._validate_language(language)

        # Criar arquivo temporário
        temp_file_path = None
        try:
            file_extension = file.filename.split(".")[-1] if file.filename else "tmp"
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{file_extension}"
            ) as temp_file:
                temp_file_path = temp_file.name

            self.logger.info(f"Salvando arquivo temporário: {temp_file_path}")

            # Ler e salvar conteúdo do arquivo
            try:
                content = await file.read()
                if not content:
                    raise HTTPException(status_code=400, detail="Arquivo vazio")

                async with aiofiles.open(temp_file_path, "wb") as f:
                    await f.write(content)

                self.logger.info(
                    f"Arquivo salvo com sucesso. Tamanho: {len(content)} bytes"
                )

            except Exception as e:
                error_msg = f"Erro ao processar arquivo: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise HTTPException(status_code=400, detail=error_msg)

            # Fazer transcrição
            try:
                if selected_provider == Provider.GROQ:
                    transcription = await self._transcribe_with_groq(
                        temp_file_path, selected_model, selected_language
                    )
                else:
                    transcription = await self._transcribe_with_openai(
                        temp_file_path, selected_model, selected_language
                    )

                if not transcription or not transcription.strip():
                    self.logger.warning("Transcrição resultou em texto vazio")
                    transcription = "[Nenhum conteúdo detectado no áudio]"

                result = {
                    "transcription": transcription.strip(),
                    "provider": selected_provider.value,
                    "model": selected_model,
                    "language": selected_language,
                    "filename": file.filename or "arquivo_sem_nome",
                }

                self.logger.info(
                    f"Transcrição concluída com sucesso para {file.filename}"
                )
                return result

            except HTTPException:
                raise
            except Exception as e:
                error_msg = f"Erro durante transcrição: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise HTTPException(status_code=500, detail=error_msg)

        finally:
            # Limpar arquivo temporário
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    self.logger.info(f"Arquivo temporário removido: {temp_file_path}")
                except Exception as e:
                    self.logger.warning(
                        f"Erro ao remover arquivo temporário {temp_file_path}: {str(e)}"
                    )

    def get_available_models(self) -> Dict[str, Any]:
        try:
            result = {
                "providers": [provider.value for provider in Provider],
                "models": {
                    provider.value: models
                    for provider, models in AVAILABLE_MODELS.items()
                },
                "default_provider": DEFAULT_PROVIDER.value,
                "default_models": {
                    provider.value: model for provider, model in DEFAULT_MODELS.items()
                },
                "supported_languages": SUPPORTED_LANGUAGES,
                "default_language": DEFAULT_LANGUAGE,
            }
            self.logger.info("Modelos disponíveis obtidos com sucesso")
            return result
        except Exception as e:
            error_msg = f"Erro ao obter modelos disponíveis: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise HTTPException(status_code=500, detail=error_msg)
