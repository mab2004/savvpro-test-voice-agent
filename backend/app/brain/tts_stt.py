"""
Lightweight, zero-API-key Speech-to-Text (STT) and Text-to-Speech (TTS) helpers.

- In the browser, the React/Next.js frontend uses the native Web Speech API
  (`SpeechRecognition` + `speechSynthesis`) for instant real-time voice I/O.
- On the backend, `/api/tts` can also synthesize WAV audio offline using `pyttsx3`
  (OS SAPI5/NSSpeech/eSpeak) if installed, or a deterministic clean WAV audio
  chime/voice-envelope fallback so endpoints work in any headless CI environment
  without paid API keys.
"""

from __future__ import annotations

import base64
import io
import math
import struct
import tempfile
import wave
from pathlib import Path
from typing import Optional


def _generate_fallback_wav_bytes(text: str, sample_rate: int = 16000) -> bytes:
    """
    Generates a clean, valid PCM WAV audio buffer whose duration and cadence
    are proportional to the assistant's spoken response words (used when
    headless OS TTS engines like pyttsx3 are not installed).
    """
    words = max(1, len((text or "hello").split()))
    duration_sec = min(2.5, max(0.45, words * 0.08))
    num_samples = int(sample_rate * duration_sec)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)

        frames = bytearray()
        for i in range(num_samples):
            t = i / sample_rate
            # Warm two-tone pleasant voice-chime envelope (440 Hz + 554.37 Hz major third)
            envelope = math.sin(math.pi * min(1.0, t / duration_sec)) * 0.25
            syllable_mod = 0.65 + 0.35 * math.sin(2.0 * math.pi * 4.5 * t)
            sample_val = envelope * syllable_mod * (
                0.6 * math.sin(2.0 * math.pi * 440.0 * t)
                + 0.4 * math.sin(2.0 * math.pi * 554.37 * t)
            )
            pcm_int = max(-32767, min(32767, int(sample_val * 32767)))
            frames.extend(struct.pack("<h", pcm_int))

        wf.writeframes(bytes(frames))

    return buf.getvalue()


def synthesize_tts_wav(text: str) -> bytes:
    """
    Synthesizes speech audio as WAV bytes without any external cloud API key.
    Tries local `pyttsx3` first; falls back to `_generate_fallback_wav_bytes`.
    """
    try:
        import pyttsx3  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.save_to_file(text, str(tmp_path))
            engine.runAndWait()
            if tmp_path.exists() and tmp_path.stat().st_size > 128:
                return tmp_path.read_bytes()
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
    except Exception:
        pass

    return _generate_fallback_wav_bytes(text)


def synthesize_tts_base64(text: str) -> str:
    """Returns base64-encoded WAV audio for JSON transport."""
    wav_bytes = synthesize_tts_wav(text)
    return base64.b64encode(wav_bytes).decode("ascii")


def transcribe_audio_bytes(
    audio_bytes: bytes,
    fallback_transcript: Optional[str] = None,
) -> str:
    """
    Transcribes uploaded audio bytes using local `faster-whisper` if installed,
    or uses `fallback_transcript` when provided by the client.
    Zero external LLM/cloud API keys required.
    """
    if fallback_transcript and fallback_transcript.strip():
        return fallback_transcript.strip()

    try:
        from faster_whisper import WhisperModel  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)
        try:
            model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(str(tmp_path))
            text = " ".join(seg.text.strip() for seg in segments).strip()
            if text:
                return text
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
    except Exception:
        pass

    return ""
