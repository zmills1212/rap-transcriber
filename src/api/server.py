"""
FastAPI server for rap transcription.
Provides REST API endpoints for transcription.
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
sys.path.insert(0, '.')

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import tempfile
import shutil
from pathlib import Path
import time
import uuid

from src.inference.engine import InferenceEngine
from src.inference.postprocessor import TranscriptionCleaner


# Initialize FastAPI app
app = FastAPI(
    title="Rap Transcription API",
    description="AI-powered transcription for rap music with slang recognition",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (lazy loaded)
_engine: Optional[InferenceEngine] = None
_cleaner: Optional[TranscriptionCleaner] = None

# Job storage for async processing
_jobs = {}


# ============== Pydantic Models ==============

class TranscriptionResponse(BaseModel):
    """Response model for transcription."""
    text: str
    audio_path: Optional[str] = None
    duration_seconds: Optional[float] = None
    processing_time_seconds: Optional[float] = None
    phonemes: Optional[List[str]] = None


class TranscriptionRequest(BaseModel):
    """Request model for text-based operations."""
    text: str
    format_as_lyrics: bool = False
    normalize_slang: bool = False


class JobStatus(BaseModel):
    """Status of an async transcription job."""
    job_id: str
    status: str  # pending, processing, completed, failed
    result: Optional[TranscriptionResponse] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: Optional[str] = None


# ============== Helper Functions ==============

def get_engine() -> InferenceEngine:
    """Get or initialize the inference engine."""
    global _engine
    if _engine is None:
        print("Loading transcription model...")
        _engine = InferenceEngine()
        print(f"Model loaded on {_engine.device}")
    return _engine


def get_cleaner() -> TranscriptionCleaner:
    """Get or initialize the text cleaner."""
    global _cleaner
    if _cleaner is None:
        _cleaner = TranscriptionCleaner()
    return _cleaner


async def save_upload_file(upload_file: UploadFile) -> Path:
    """Save uploaded file to temporary location."""
    suffix = Path(upload_file.filename).suffix or '.mp3'
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(upload_file.file, tmp)
        return Path(tmp.name)


def process_transcription(
    audio_path: Path,
    format_as_lyrics: bool = False,
    return_phonemes: bool = False
) -> TranscriptionResponse:
    """Process audio file and return transcription."""
    engine = get_engine()
    cleaner = get_cleaner()
    
    start_time = time.time()
    
    # Transcribe
    result = engine.transcribe(
        audio_path,
        return_phonemes=return_phonemes,
        return_timing=True
    )
    
    # Get raw text
    raw_text = result['text']
    if isinstance(raw_text, list):
        raw_text = ' '.join(map(str, raw_text))
    
    # Clean text
    cleaned_text = cleaner.clean(raw_text, format_as_lyrics=format_as_lyrics)
    
    processing_time = time.time() - start_time
    
    response = TranscriptionResponse(
        text=cleaned_text,
        audio_path=str(audio_path),
        processing_time_seconds=round(processing_time, 3)
    )
    
    if return_phonemes and 'phonemes' in result:
        response.phonemes = result['phonemes']
    
    if 'timing' in result:
        response.duration_seconds = result['timing'].get('total')
    
    return response


# ============== API Endpoints ==============

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Rap Transcription API",
        "version": "1.0.0",
        "endpoints": {
            "POST /transcribe": "Transcribe audio file",
            "POST /transcribe/async": "Start async transcription",
            "GET /job/{job_id}": "Get async job status",
            "POST /format": "Format/clean text",
            "GET /health": "Health check"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global _engine
    
    return HealthResponse(
        status="healthy",
        model_loaded=_engine is not None,
        device=_engine.device if _engine else None
    )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    format_as_lyrics: bool = False,
    return_phonemes: bool = False
):
    """
    Transcribe an audio file.
    
    - **file**: Audio file (mp3, wav, flac, m4a)
    - **format_as_lyrics**: Format output as lyrics with line breaks
    - **return_phonemes**: Include phoneme predictions
    """
    # Validate file type
    allowed_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )
    
    # Save uploaded file
    tmp_path = await save_upload_file(file)
    
    try:
        # Process transcription
        response = process_transcription(
            tmp_path,
            format_as_lyrics=format_as_lyrics,
            return_phonemes=return_phonemes
        )
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup temp file
        if tmp_path.exists():
            tmp_path.unlink()


@app.post("/transcribe/async", response_model=JobStatus)
async def transcribe_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    format_as_lyrics: bool = False
):
    """
    Start async transcription job.
    
    Returns job_id to check status later.
    """
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    tmp_path = await save_upload_file(file)
    
    # Initialize job
    _jobs[job_id] = {
        'status': 'pending',
        'result': None,
        'error': None,
        'tmp_path': tmp_path
    }
    
    # Add background task
    background_tasks.add_task(
        run_async_transcription,
        job_id,
        tmp_path,
        format_as_lyrics
    )
    
    return JobStatus(job_id=job_id, status='pending')


def run_async_transcription(job_id: str, audio_path: Path, format_as_lyrics: bool):
    """Background task for async transcription."""
    try:
        _jobs[job_id]['status'] = 'processing'
        
        result = process_transcription(audio_path, format_as_lyrics)
        
        _jobs[job_id]['status'] = 'completed'
        _jobs[job_id]['result'] = result
        
    except Exception as e:
        _jobs[job_id]['status'] = 'failed'
        _jobs[job_id]['error'] = str(e)
        
    finally:
        # Cleanup
        if audio_path.exists():
            audio_path.unlink()


@app.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of async transcription job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _jobs[job_id]
    
    return JobStatus(
        job_id=job_id,
        status=job['status'],
        result=job['result'],
        error=job['error']
    )


@app.post("/format", response_model=dict)
async def format_text(request: TranscriptionRequest):
    """
    Format/clean transcription text.
    
    - **text**: Text to format
    - **format_as_lyrics**: Format as lyrics with line breaks
    - **normalize_slang**: Expand slang to standard English
    """
    cleaner = get_cleaner()
    
    # Update cleaner settings
    cleaner.postprocessor.normalize_slang = request.normalize_slang
    
    formatted = cleaner.clean(request.text, format_as_lyrics=request.format_as_lyrics)
    
    return {
        "original": request.text,
        "formatted": formatted
    }


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    print("Starting Rap Transcription API...")
    # Optionally preload model
    # get_engine()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Shutting down...")
    # Cleanup any remaining temp files
    for job in _jobs.values():
        if 'tmp_path' in job and Path(job['tmp_path']).exists():
            Path(job['tmp_path']).unlink()


# ============== Run Server ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ============== WebSocket Support ==============

from src.api.websocket import add_websocket_routes

# Add WebSocket routes
add_websocket_routes(app)
