"""
Rap Transcription API Server (Integrated)

Whisper transcription with automatic post-processing for rap-specific corrections.

Usage:
    uvicorn src.api.rap_server:app --reload --port 8000
    
    # Or run directly:
    python -m src.api.rap_server

Endpoints:
    POST /transcribe     - Transcribe audio file
    POST /transcribe/raw - Transcribe without post-processing
    GET  /health         - Health check
    GET  /stats          - Post-processor statistics
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
sys.path.insert(0, '.')

import time
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.inference.whisper_engine import WhisperTranscriber
from postprocessor.rap_postprocessor import RapPostProcessor, CorrectionResult


# =============================================================================
# App Setup
# =============================================================================

app = FastAPI(
    title="Rap Transcription API",
    description="AI-powered rap transcription using Whisper + intelligent post-processing",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Global State (lazy loaded)
# =============================================================================

_transcriber: Optional[WhisperTranscriber] = None
_postprocessor: Optional[RapPostProcessor] = None


def get_transcriber(model_size: str = "medium") -> WhisperTranscriber:
    """Get or create Whisper transcriber."""
    global _transcriber
    if _transcriber is None or _transcriber.model_size != model_size:
        _transcriber = WhisperTranscriber(model_size=model_size)
    return _transcriber


def get_postprocessor(aggressive: bool = True) -> RapPostProcessor:
    """Get or create post-processor."""
    global _postprocessor
    if _postprocessor is None or _postprocessor.aggressive != aggressive:
        _postprocessor = RapPostProcessor(aggressive=aggressive)
    return _postprocessor


# =============================================================================
# Response Models
# =============================================================================

class TranscriptionResponse(BaseModel):
    """Response from transcription endpoint."""
    text: str
    raw_text: str
    corrections_applied: List[str]
    processing_time_seconds: float
    model_size: str
    post_processed: bool


class RawTranscriptionResponse(BaseModel):
    """Response from raw transcription (no post-processing)."""
    text: str
    segments: Optional[List[dict]] = None
    processing_time_seconds: float
    model_size: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_size: Optional[str]
    postprocessor_loaded: bool


class StatsResponse(BaseModel):
    """Post-processor statistics."""
    slang_corrections: int
    explicit_corrections: int
    dialect_corrections: int
    phrase_corrections: int
    total: int


# =============================================================================
# Helper Functions
# =============================================================================

async def save_upload(file: UploadFile) -> Path:
    """Save uploaded file to temp location."""
    suffix = Path(file.filename).suffix if file.filename else '.mp3'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        return Path(tmp.name)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/")
async def root():
    """API information."""
    return {
        "name": "Rap Transcription API",
        "version": "2.0.0",
        "description": "Whisper + intelligent post-processing for rap lyrics",
        "endpoints": {
            "POST /transcribe": "Transcribe with post-processing (recommended)",
            "POST /transcribe/raw": "Transcribe without post-processing",
            "GET /health": "Health check",
            "GET /stats": "Post-processor statistics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    global _transcriber, _postprocessor
    return HealthResponse(
        status="healthy",
        model_loaded=_transcriber is not None,
        model_size=_transcriber.model_size if _transcriber else None,
        postprocessor_loaded=_postprocessor is not None
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Get post-processor statistics."""
    processor = get_postprocessor()
    return StatsResponse(**processor.get_stats())


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    model_size: str = Query("medium", description="Whisper model: tiny, base, small, medium"),
    aggressive: bool = Query(True, description="Use aggressive post-processing")
):
    """
    Transcribe audio file with automatic post-processing.
    
    This is the recommended endpoint for rap transcription. It:
    1. Runs Whisper to get raw transcription
    2. Applies intelligent post-processing to fix common errors
    3. Returns both raw and corrected text
    """
    # Save uploaded file
    audio_path = await save_upload(file)
    
    try:
        # Get transcriber and post-processor
        transcriber = get_transcriber(model_size)
        postprocessor = get_postprocessor(aggressive)
        
        # Transcribe
        start_time = time.time()
        result = transcriber.transcribe(str(audio_path))
        raw_text = result.get("text", "") if isinstance(result, dict) else str(result)
        
        # Post-process
        correction_result = postprocessor.process(raw_text, track_changes=True)
        elapsed = time.time() - start_time
        
        return TranscriptionResponse(
            text=correction_result.corrected,
            raw_text=raw_text,
            corrections_applied=correction_result.changes,
            processing_time_seconds=round(elapsed, 2),
            model_size=model_size,
            post_processed=True
        )
        
    finally:
        # Cleanup temp file
        audio_path.unlink(missing_ok=True)


@app.post("/transcribe/raw", response_model=RawTranscriptionResponse)
async def transcribe_raw(
    file: UploadFile = File(...),
    model_size: str = Query("medium", description="Whisper model: tiny, base, small, medium"),
    include_segments: bool = Query(False, description="Include word-level segments")
):
    """
    Transcribe audio file without post-processing.
    
    Use this endpoint if you want raw Whisper output.
    """
    audio_path = await save_upload(file)
    
    try:
        transcriber = get_transcriber(model_size)
        
        start_time = time.time()
        result = transcriber.transcribe(str(audio_path))
        elapsed = time.time() - start_time
        
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        segments = result.get("segments") if include_segments and isinstance(result, dict) else None
        
        return RawTranscriptionResponse(
            text=text,
            segments=segments,
            processing_time_seconds=round(elapsed, 2),
            model_size=model_size
        )
        
    finally:
        audio_path.unlink(missing_ok=True)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n🎤 Starting Rap Transcription API Server...")
    print("   Docs: http://localhost:8000/docs")
    print("   Health: http://localhost:8000/health\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
