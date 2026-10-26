"""
FastAPI Backend for DawatDesk Voice Concierge (Haveli Grill & Courtyard, Islamabad).

Exposes REST endpoints for:
- Conversational turn processing with full `DecisionTrace` explainability (`/api/turn`)
- Audio blob voice turn processing (`/api/voice`)
- Session inspection & reset (`/api/session/{session_id}`, `/api/reset`)
- Live SQLite table & booking inventory (`/api/bookings`, `/api/tables`)
- Offline TTS audio synthesis (`/api/tts`)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.brain.dialogue_manager import DialogueManager
from app.brain.tts_stt import synthesize_tts_base64, synthesize_tts_wav, transcribe_audio_bytes
from app.models import (
    RESTAURANT_BRANCH,
    RESTAURANT_NAME,
    RESTAURANT_PHONE,
    TTSRequest,
    TurnRequest,
    TurnResponse,
)

app = FastAPI(
    title="DawatDesk Voice Concierge API",
    description=(
        "Custom Non-LLM Voice Assistant Brain for Haveli Grill & Courtyard (F-7 Markaz, Islamabad). "
        "Powered by a 5-stage deterministic & calibrated scikit-learn pipeline + SQLite inventory."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global DialogueManager instance (with SQLite InventoryManager & trained Classifier)
dialogue_manager = DialogueManager()


@app.get("/health")
@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "restaurant": RESTAURANT_NAME,
        "branch": RESTAURANT_BRANCH,
        "phone": RESTAURANT_PHONE,
        "brain_engine": "Custom 5-Stage Non-LLM Pipeline (Normalizer + TF-IDF/CalibratedSVC + Context Slot Extractor + HFSM Digression Stack + SQLite Inventory)",
        "llm_used": False,
    }


@app.post("/turn", response_model=TurnResponse)
@app.post("/api/turn", response_model=TurnResponse)
def process_conversation_turn(req: TurnRequest) -> TurnResponse:
    """
    Processes a single caller utterance through the 5-stage non-LLM brain
    and returns the assistant response, updated session state, and rich `DecisionTrace`.
    """
    result = dialogue_manager.process_turn(
        text=req.text,
        session_id=req.session_id,
        reference_date=req.reference_date,
    )
    if req.include_tts:
        result.audio_base64 = synthesize_tts_base64(result.response_text)
    return result


@app.post("/voice", response_model=TurnResponse)
@app.post("/api/voice", response_model=TurnResponse)
async def process_voice_upload(
    audio: Optional[UploadFile] = File(default=None),
    session_id: Optional[str] = Form(default=None),
    transcript: Optional[str] = Form(default=None),
    reference_date: Optional[str] = Form(default=None),
) -> TurnResponse:
    """
    Accepts an uploaded audio blob (and/or browser Web Speech transcript),
    transcribes locally without paid API keys, runs the non-LLM brain, and
    returns both text + base64 WAV audio.
    """
    raw_bytes = await audio.read() if audio is not None else b""
    spoken_text = transcribe_audio_bytes(raw_bytes, fallback_transcript=transcript)
    result = dialogue_manager.process_turn(
        text=spoken_text,
        session_id=session_id,
        reference_date=reference_date,
    )
    result.audio_base64 = synthesize_tts_base64(result.response_text)
    return result


@app.get("/session/{session_id}")
@app.get("/api/session/{session_id}")
def get_session_state(session_id: str) -> Dict[str, Any]:
    if session_id not in dialogue_manager.sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    sess = dialogue_manager.sessions[session_id]
    return sess.model_dump()


@app.post("/session/{session_id}/reset")
@app.post("/api/session/{session_id}/reset")
def reset_single_session(session_id: str) -> Dict[str, Any]:
    sess = dialogue_manager.reset_session(session_id)
    return {"status": "reset", "session": sess.model_dump()}


@app.get("/bookings")
@app.get("/api/bookings")
def list_all_bookings(include_cancelled: bool = False) -> Dict[str, Any]:
    return {
        "bookings": dialogue_manager.inventory.list_bookings(include_cancelled=include_cancelled),
        "tables": dialogue_manager.inventory.list_tables(),
    }


@app.get("/tables")
@app.get("/api/tables")
def list_all_tables() -> Dict[str, Any]:
    return {
        "tables": dialogue_manager.inventory.list_tables(),
    }


@app.post("/reset")
@app.post("/api/reset")
def reset_system(reference_date: Optional[str] = None) -> Dict[str, Any]:
    """Resets all active call sessions and re-seeds the SQLite inventory database."""
    dialogue_manager.reset_all(reference_date=reference_date)
    return {
        "status": "reset_complete",
        "bookings": dialogue_manager.inventory.list_bookings(),
    }


@app.post("/tts")
@app.post("/api/tts")
def generate_tts_audio(req: TTSRequest) -> Response:
    """Synthesizes WAV speech audio for the given text without external API keys."""
    wav_bytes = synthesize_tts_wav(req.text)
    return Response(content=wav_bytes, media_type="audio/wav")
