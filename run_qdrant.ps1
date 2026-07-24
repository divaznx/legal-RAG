# Launch the standalone Qdrant vector-DB server (v1.18.3, Windows binary in
# qdrant_server\). With QDRANT_URL=http://localhost:6333 in .env, the API,
# Streamlit UI, and CLI all share this server and can run at the same time —
# the embedded store's one-process-at-a-time limit no longer applies.
#
# Start this FIRST, before the API / UI / CLI. Data lives in
# qdrant_server\storage. Stop with Ctrl+C.

Set-Location (Join-Path $PSScriptRoot "qdrant_server")
Write-Host "Starting Qdrant server on http://localhost:6333 ..." -ForegroundColor Cyan
.\qdrant.exe
