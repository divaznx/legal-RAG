# Launch Open WebUI wired to the Lexis API.
#
# Start the API first (it is the "model provider" Open WebUI talks to):
#   uvicorn api:app --port 8000
#
# Requires uv (https://docs.astral.sh/uv). The first run downloads
# Open WebUI and a Python 3.11 runtime into uv's cache (~1 GB, one time).
# UI comes up at http://localhost:3000 with a single model named "lexis".

$env:OPENAI_API_BASE_URL = "http://localhost:8000/v1"
$env:OPENAI_API_KEY = "lexis"        # any non-empty value; Lexis ignores it
$env:ENABLE_OLLAMA_API = "false"     # Ollama is reached only through Lexis
$env:WEBUI_AUTH = "false"            # local single-user mode, no login page
$env:ENABLE_WEB_SEARCH = "false"     # answers must come from ingested docs only
$env:DATA_DIR = Join-Path $PSScriptRoot "data\open-webui"

# Lexis does all retrieval, so Open WebUI's own RAG is dead weight. Disabling
# it skips the ~90 MB sentence-transformers download that otherwise stalls the
# first startup (and fails entirely when offline) — the noisy red HuggingFace
# WARNING lines during boot come from that download, not from a real error.
$env:RAG_EMBEDDING_ENGINE = "ollama"       # do not load a local embedding model
$env:AUDIO_STT_ENGINE = "openai"           # skip the local speech-to-text model too
$env:HF_HUB_OFFLINE = "1"                  # never reach out to HuggingFace at boot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

Write-Host "Starting Open WebUI on http://localhost:3000 (Lexis API must be running on :8000)..." -ForegroundColor Cyan
uvx --python 3.11 open-webui@latest serve --port 3000
