# Rap Transcription API Documentation

## Base URL
```
http://localhost:8000
```

## Authentication

Currently no authentication required.

---

## Endpoints

### GET /

Returns API information.

**Response:**
```json
{
  "name": "Rap Transcription API",
  "version": "1.0.0",
  "endpoints": {...}
}
```

---

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "mps"
}
```

---

### POST /transcribe

Transcribe an audio file.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (audio file)

**Query Parameters:**
- `format_as_lyrics` (bool): Format output as lyrics
- `return_phonemes` (bool): Include phoneme predictions

**Response:**
```json
{
  "text": "transcribed text here",
  "audio_path": "/tmp/...",
  "processing_time_seconds": 1.234,
  "phonemes": ["F", "IH1", "N", "AH0"]
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.mp3"
```

---

### POST /transcribe/async

Start async transcription job.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (audio file)

**Response:**
```json
{
  "job_id": "uuid-here",
  "status": "pending"
}
```

---

### GET /job/{job_id}

Get async job status.

**Response:**
```json
{
  "job_id": "uuid-here",
  "status": "completed",
  "result": {
    "text": "transcribed text"
  }
}
```

**Status values:** `pending`, `processing`, `completed`, `failed`

---

### POST /format

Format/clean transcription text.

**Request:**
```json
{
  "text": "i'm finna get bread",
  "format_as_lyrics": false,
  "normalize_slang": true
}
```

**Response:**
```json
{
  "original": "i'm finna get bread",
  "formatted": "I'm fixing to get bread"
}
```

---

## WebSocket

### /ws/transcribe

Streaming transcription endpoint.

**Protocol:**

Client sends:
```json
{"type": "audio", "data": "<base64>", "format": "wav"}
{"type": "transcribe"}
```

Server sends:
```json
{"type": "partial", "text": "partial..."}
{"type": "final", "text": "complete transcription"}
```

---

## Error Handling

Errors return appropriate HTTP status codes:

- `400` - Bad request (invalid file type, etc.)
- `404` - Not found (invalid job_id)
- `500` - Server error

Error response format:
```json
{
  "detail": "Error message here"
}
```
