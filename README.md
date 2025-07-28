# Audio Transcription API

API simples para transcrição de áudio usando Groq e OpenAI Whisper.

## Características

- 🚀 **Async**: Processamento assíncrono para melhor performance
- 🔄 **Multi-provider**: Suporte a Groq e OpenAI
- 📝 **Documentação**: Swagger UI automática
- ⚙️ **Configurável**: Provider e modelo padrão configuráveis
- 🎵 **Formatos**: Suporte a múltiplos formatos de áudio

## Instalação

```bash
# Instalar UV (se não tiver)
pip install uv

# Instalar dependências
uv sync

# Configurar variáveis de ambiente
cp .env.example .env
# Edite o .env com suas API keys
```

## Configuração

Configure pelo menos uma API key no arquivo `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

## Execução

```bash
# Desenvolvimento
uv run uvicorn main:app --reload

# Produção
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

## Endpoints

### GET `/models`
Lista todos os providers e modelos disponíveis.

### POST `/transcribe`
Transcreve um arquivo de áudio.

**Parâmetros:**
- `file`: Arquivo de áudio (obrigatório)
- `provider`: "groq" ou "openai" (opcional)
- `model`: Modelo específico (opcional)

**Formatos suportados:**
mp3, mp4, mpeg, mpga, m4a, wav, webm, flac, ogg

**Tamanho máximo:** 25MB

## Exemplo de uso

```bash
# Transcrição com provider padrão
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.mp3"

# Transcrição com provider específico
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.mp3" \
  -F "provider=groq" \
  -F "model=whisper-large-v3-turbo"
```

## Documentação

Acesse `http://localhost:8000/docs` para a documentação interativa Swagger UI.