"""
Client for interacting with the Rap Transcription API.
"""
import requests
from pathlib import Path
from typing import Optional, Dict
import time


class TranscriptionClient:
    """
    Client for the Rap Transcription API.
    
    Usage:
        client = TranscriptionClient("http://localhost:8000")
        result = client.transcribe("audio.mp3")
        print(result['text'])
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict:
        """Check API health status."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def transcribe(
        self,
        audio_path: str,
        format_as_lyrics: bool = False,
        return_phonemes: bool = False
    ) -> Dict:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file
            format_as_lyrics: Format output as lyrics
            return_phonemes: Include phoneme predictions
            
        Returns:
            Transcription result dict
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        with open(audio_path, 'rb') as f:
            files = {'file': (audio_path.name, f)}
            params = {
                'format_as_lyrics': format_as_lyrics,
                'return_phonemes': return_phonemes
            }
            
            response = requests.post(
                f"{self.base_url}/transcribe",
                files=files,
                params=params
            )
        
        response.raise_for_status()
        return response.json()
    
    def transcribe_async(
        self,
        audio_path: str,
        format_as_lyrics: bool = False,
        poll_interval: float = 1.0,
        timeout: float = 300.0
    ) -> Dict:
        """
        Transcribe audio asynchronously and wait for result.
        
        Args:
            audio_path: Path to audio file
            format_as_lyrics: Format output as lyrics
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait
            
        Returns:
            Transcription result dict
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Start async job
        with open(audio_path, 'rb') as f:
            files = {'file': (audio_path.name, f)}
            params = {'format_as_lyrics': format_as_lyrics}
            
            response = requests.post(
                f"{self.base_url}/transcribe/async",
                files=files,
                params=params
            )
        
        response.raise_for_status()
        job_data = response.json()
        job_id = job_data['job_id']
        
        # Poll for completion
        start_time = time.time()
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Transcription timed out after {timeout}s")
            
            status = self.get_job_status(job_id)
            
            if status['status'] == 'completed':
                return status['result']
            elif status['status'] == 'failed':
                raise RuntimeError(f"Transcription failed: {status['error']}")
            
            time.sleep(poll_interval)
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get status of async transcription job."""
        response = requests.get(f"{self.base_url}/job/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def format_text(
        self,
        text: str,
        format_as_lyrics: bool = False,
        normalize_slang: bool = False
    ) -> Dict:
        """
        Format/clean transcription text.
        
        Args:
            text: Text to format
            format_as_lyrics: Format as lyrics
            normalize_slang: Expand slang
            
        Returns:
            Dict with original and formatted text
        """
        response = requests.post(
            f"{self.base_url}/format",
            json={
                'text': text,
                'format_as_lyrics': format_as_lyrics,
                'normalize_slang': normalize_slang
            }
        )
        
        response.raise_for_status()
        return response.json()


def demo():
    """Demo the API client."""
    print("=" * 50)
    print("RAP TRANSCRIPTION API CLIENT DEMO")
    print("=" * 50)
    print("")
    print("This client connects to the API server.")
    print("Start the server first with:")
    print("  python scripts/run_api.py")
    print("")
    print("Then use the client:")
    print("")
    print("  from src.api.client import TranscriptionClient")
    print("  client = TranscriptionClient('http://localhost:8000')")
    print("  result = client.transcribe('audio.mp3')")
    print("  print(result['text'])")
    print("")
    
    # Test without server
    print("Testing client initialization...")
    client = TranscriptionClient()
    print(f"   ✅ Client created")
    print(f"   ✅ Base URL: {client.base_url}")


if __name__ == "__main__":
    demo()
