import sys
sys.path.insert(0, ".")

"""
WebSocket handler for real-time streaming transcription.
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Optional
import base64

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from src.inference.engine import InferenceEngine, StreamingInferenceEngine
from src.inference.postprocessor import TranscriptionCleaner


# Global instances
_streaming_engine: Optional[StreamingInferenceEngine] = None
_cleaner: Optional[TranscriptionCleaner] = None


def get_streaming_engine() -> StreamingInferenceEngine:
    """Get or initialize streaming inference engine."""
    global _streaming_engine
    if _streaming_engine is None:
        print("Loading streaming transcription model...")
        _streaming_engine = StreamingInferenceEngine(
            chunk_duration=5.0,
            overlap_duration=0.5
        )
        print(f"Streaming model loaded on {_streaming_engine.device}")
    return _streaming_engine


def get_cleaner() -> TranscriptionCleaner:
    """Get or initialize text cleaner."""
    global _cleaner
    if _cleaner is None:
        _cleaner = TranscriptionCleaner()
    return _cleaner


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept and track new connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove connection from tracking."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_json(self, websocket: WebSocket, data: dict):
        """Send JSON message to client."""
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(data)
    
    async def broadcast(self, data: dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            await self.send_json(connection, data)


# Global connection manager
manager = ConnectionManager()


async def handle_transcription_stream(websocket: WebSocket):
    """
    Handle streaming transcription over WebSocket.
    
    Protocol:
        Client sends:
            {"type": "audio", "data": "<base64 encoded audio>", "format": "wav"}
            {"type": "end"}
        
        Server sends:
            {"type": "partial", "text": "partial transcription..."}
            {"type": "final", "text": "final transcription", "processing_time": 1.23}
            {"type": "error", "message": "error description"}
    """
    await manager.connect(websocket)
    
    engine = get_streaming_engine()
    cleaner = get_cleaner()
    
    audio_buffer = b''
    
    try:
        while True:
            # Receive message
            message = await websocket.receive_json()
            msg_type = message.get('type')
            
            if msg_type == 'audio':
                # Decode audio chunk
                audio_data = base64.b64decode(message.get('data', ''))
                audio_buffer += audio_data
                
                # Send acknowledgment
                await manager.send_json(websocket, {
                    'type': 'ack',
                    'bytes_received': len(audio_buffer)
                })
            
            elif msg_type == 'transcribe':
                # Process buffered audio
                if not audio_buffer:
                    await manager.send_json(websocket, {
                        'type': 'error',
                        'message': 'No audio data received'
                    })
                    continue
                
                # Save to temp file
                audio_format = message.get('format', 'wav')
                with tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=f'.{audio_format}'
                ) as tmp:
                    tmp.write(audio_buffer)
                    tmp_path = Path(tmp.name)
                
                try:
                    # Send processing status
                    await manager.send_json(websocket, {
                        'type': 'status',
                        'message': 'Processing audio...'
                    })
                    
                    # Streaming transcription
                    import time
                    start_time = time.time()
                    
                    results = engine.transcribe_streaming(tmp_path)
                    
                    # Send partial results as they come
                    full_text = []
                    for chunk_result in results:
                        chunk_text = chunk_result['text']
                        if isinstance(chunk_text, list):
                            chunk_text = ' '.join(map(str, chunk_text))
                        
                        cleaned = cleaner.clean(chunk_text, format_as_lyrics=False)
                        full_text.append(cleaned)
                        
                        await manager.send_json(websocket, {
                            'type': 'partial',
                            'chunk': chunk_result['chunk'],
                            'start_time': chunk_result['start_time'],
                            'end_time': chunk_result['end_time'],
                            'text': cleaned
                        })
                    
                    processing_time = time.time() - start_time
                    
                    # Send final result
                    await manager.send_json(websocket, {
                        'type': 'final',
                        'text': ' '.join(full_text),
                        'chunks': len(results),
                        'processing_time': round(processing_time, 3)
                    })
                    
                finally:
                    # Cleanup
                    if tmp_path.exists():
                        tmp_path.unlink()
                    audio_buffer = b''
            
            elif msg_type == 'clear':
                # Clear audio buffer
                audio_buffer = b''
                await manager.send_json(websocket, {
                    'type': 'cleared'
                })
            
            elif msg_type == 'ping':
                # Keepalive
                await manager.send_json(websocket, {
                    'type': 'pong'
                })
            
            elif msg_type == 'end':
                # Client ending session
                break
            
            else:
                await manager.send_json(websocket, {
                    'type': 'error',
                    'message': f'Unknown message type: {msg_type}'
                })
    
    except WebSocketDisconnect:
        pass
    
    except Exception as e:
        try:
            await manager.send_json(websocket, {
                'type': 'error',
                'message': str(e)
            })
        except:
            pass
    
    finally:
        manager.disconnect(websocket)


def add_websocket_routes(app):
    """Add WebSocket routes to FastAPI app."""
    
    @app.websocket("/ws/transcribe")
    async def websocket_transcribe(websocket: WebSocket):
        """WebSocket endpoint for streaming transcription."""
        await handle_transcription_stream(websocket)
    
    @app.websocket("/ws/health")
    async def websocket_health(websocket: WebSocket):
        """WebSocket health check endpoint."""
        await websocket.accept()
        await websocket.send_json({
            'status': 'connected',
            'message': 'WebSocket connection healthy'
        })
        await websocket.close()


if __name__ == "__main__":
    print("✅ WebSocket module loaded successfully!")
    print("")
    print("WebSocket endpoints:")
    print("  /ws/transcribe - Streaming transcription")
    print("  /ws/health     - Connection health check")
    print("")
    print("Protocol:")
    print("  Send: {'type': 'audio', 'data': '<base64>', 'format': 'wav'}")
    print("  Send: {'type': 'transcribe'}")
    print("  Recv: {'type': 'partial', 'text': '...'}")
    print("  Recv: {'type': 'final', 'text': '...'}")
