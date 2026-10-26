@echo off
echo ============================================================================
echo   DawatDesk Voice Concierge - Haveli Grill ^& Courtyard (F-7 Markaz)
echo   Starting Custom Non-LLM Brain Backend (Port 8000) ^& React UI (Port 5173)
echo ============================================================================

start "DawatDesk Backend (Port 8000)" cmd /k "cd /d %~dp0backend && .venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000 --reload"
start "DawatDesk Frontend (Port 5173)" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Backend API:  http://127.0.0.1:8000/docs
echo Frontend UI:  http://localhost:5173
echo.
