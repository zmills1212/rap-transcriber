"""
Inference engine for rap transcription.
Handles audio-to-text transcription with the trained model.
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, List, Dict, Optional
import time

from src.data.audio_processor import AudioProcessor
from src.data.feature_extractor import FeatureExtractor
from src.models.rap_transcriber import RapTranscriber, create_model


class InferenceEngine:
    """
    High-level inference engine for rap transcription.
    
    Handles the complete pipeline:
        Audio file -> Preprocessing -> Model -> Decoding -> Text
    """
    
    def __init__(
        self,
        model: Optional[RapTranscriber] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        audio_processor: Optional[AudioProcessor] = None,
        feature_extractor: Optional[FeatureExtractor] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Pre-loaded model (optional)
            checkpoint_path: Path to model checkpoint (optional)
            device: Device to run on (auto-detected if None)
            audio_processor: Custom audio processor (optional)
            feature_extractor: Custom feature extractor (optional)
        """
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        # Load model
        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path is not None:
            self.model = self._load_checkpoint(checkpoint_path)
        else:
            # Create default model (for testing)
            self.model = create_model().to(self.device)
        
        self.model.eval()
        
        # Initialize processors
        self.audio_processor = audio_processor or AudioProcessor()
        self.feature_extractor = feature_extractor or FeatureExtractor()
        
        # Tokenizer (will be set separately)
        self.tokenizer = None
    
    def _load_checkpoint(self, checkpoint_path: str) -> RapTranscriber:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'config' in checkpoint and checkpoint['config']:
            model = create_model(checkpoint['config'])
        else:
            model = create_model()
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        print(f"Loaded model from: {checkpoint_path}")
        return model
    
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for text decoding."""
        self.tokenizer = tokenizer
    
    def preprocess(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """
        Preprocess audio file to features.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Feature tensor ready for model
        """
        # Load and process audio
        waveform = self.audio_processor.process(audio_path, trim_pad=False)
        
        # Extract features
        features = self.feature_extractor.extract(waveform)
        
        # Add batch dimension: [n_mels, time] -> [1, time, n_mels]
        features = features.transpose(0, 1).unsqueeze(0)
        
        return features.to(self.device)
    
    @torch.no_grad()
    def transcribe(
        self,
        audio_path: Union[str, Path],
        return_phonemes: bool = False,
        return_timing: bool = False
    ) -> Dict[str, any]:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file
            return_phonemes: Whether to return phoneme predictions
            return_timing: Whether to return timing information
            
        Returns:
            Dictionary with transcription results
        """
        start_time = time.time()
        
        # Preprocess
        preprocess_start = time.time()
        features = self.preprocess(audio_path)
        preprocess_time = time.time() - preprocess_start
        
        # Forward pass
        inference_start = time.time()
        outputs = self.model(features)
        inference_time = time.time() - inference_start
        
        # Decode text
        decode_start = time.time()
        text_tokens = self.model.text_head.decode_greedy(outputs['text_logits'])
        
        # Convert to text if tokenizer available
        if self.tokenizer is not None:
            text = self.tokenizer.decode(text_tokens[0])
        else:
            text = text_tokens[0]  # Return raw token IDs
        
        decode_time = time.time() - decode_start
        
        total_time = time.time() - start_time
        
        # Build result
        result = {
            'text': text,
            'audio_path': str(audio_path)
        }
        
        # Optional: phonemes
        if return_phonemes:
            phonemes = self.model.phoneme_head.decode_greedy(outputs['phoneme_logits'])
            result['phonemes'] = phonemes[0]
        
        # Optional: timing
        if return_timing:
            result['timing'] = {
                'total': total_time,
                'preprocess': preprocess_time,
                'inference': inference_time,
                'decode': decode_time
            }
        
        return result
    
    @torch.no_grad()
    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        return_phonemes: bool = False
    ) -> List[Dict[str, any]]:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            return_phonemes: Whether to return phoneme predictions
            
        Returns:
            List of transcription results
        """
        results = []
        
        for audio_path in audio_paths:
            result = self.transcribe(
                audio_path,
                return_phonemes=return_phonemes
            )
            results.append(result)
        
        return results
    
    @torch.no_grad()
    def transcribe_features(
        self,
        features: torch.Tensor,
        return_phonemes: bool = False
    ) -> Dict[str, any]:
        """
        Transcribe from pre-extracted features.
        
        Args:
            features: Feature tensor [batch, time, n_mels]
            return_phonemes: Whether to return phoneme predictions
            
        Returns:
            Transcription results
        """
        features = features.to(self.device)
        
        outputs = self.model(features)
        
        text_tokens = self.model.text_head.decode_greedy(outputs['text_logits'])
        
        if self.tokenizer is not None:
            texts = [self.tokenizer.decode(tokens) for tokens in text_tokens]
        else:
            texts = text_tokens
        
        result = {'texts': texts}
        
        if return_phonemes:
            phonemes = self.model.phoneme_head.decode_greedy(outputs['phoneme_logits'])
            result['phonemes'] = phonemes
        
        return result
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model."""
        return {
            'device': self.device,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'config': self.model.config if hasattr(self.model, 'config') else None
        }


class StreamingInferenceEngine(InferenceEngine):
    """
    Streaming inference for real-time transcription.
    Processes audio in chunks.
    """
    
    def __init__(
        self,
        chunk_duration: float = 5.0,
        overlap_duration: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.chunk_samples = int(chunk_duration * 16000)
        self.overlap_samples = int(overlap_duration * 16000)
    
    @torch.no_grad()
    def transcribe_streaming(
        self,
        audio_path: Union[str, Path]
    ) -> List[Dict[str, any]]:
        """
        Transcribe audio in streaming fashion.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of chunk transcriptions
        """
        # Load full audio without padding
        waveform = self.audio_processor.process(audio_path, trim_pad=False)
        
        total_samples = waveform.shape[1]
        results = []
        
        start = 0
        chunk_idx = 0
        
        while start < total_samples:
            end = min(start + self.chunk_samples, total_samples)
            chunk = waveform[:, start:end]
            
            # Pad if needed
            if chunk.shape[1] < self.chunk_samples:
                padding = self.chunk_samples - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, padding))
            
            # Extract features
            features = self.feature_extractor.extract(chunk)
            features = features.transpose(0, 1).unsqueeze(0).to(self.device)
            
            # Transcribe
            outputs = self.model(features)
            text_tokens = self.model.text_head.decode_greedy(outputs['text_logits'])
            
            if self.tokenizer is not None:
                text = self.tokenizer.decode(text_tokens[0])
            else:
                text = text_tokens[0]
            
            results.append({
                'chunk': chunk_idx,
                'start_time': start / 16000,
                'end_time': end / 16000,
                'text': text
            })
            
            # Move to next chunk (with overlap)
            start += self.chunk_samples - self.overlap_samples
            chunk_idx += 1
        
        return results


if __name__ == "__main__":
    print("✅ InferenceEngine module loaded successfully!")
    print("")
    
    # Test with dummy model
    engine = InferenceEngine(device='cpu')
    
    info = engine.get_model_info()
    print(f"   Device: {info['device']}")
    print(f"   Parameters: {info['parameters']:,}")
    print("")
    
    # Test with dummy features
    dummy_features = torch.randn(1, 100, 80)
    result = engine.transcribe_features(dummy_features, return_phonemes=True)
    
    print(f"   Text tokens: {len(result['texts'][0])} tokens")
    print(f"   Phonemes: {len(result['phonemes'][0])} phonemes")
