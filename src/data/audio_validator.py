"""
Audio validation utilities for rap transcription.
Validates audio files for training suitability.
"""
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class ValidationResult:
    """Result of audio validation."""
    is_valid: bool
    audio_path: str
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class AudioValidator:
    """
    Validates audio files for training.
    
    Checks:
        - File exists and is readable
        - Duration within acceptable range
        - Sample rate is sufficient
        - Audio is not silent
        - Audio is not clipped
    """
    
    def __init__(
        self,
        min_duration: float = 1.0,
        max_duration: float = 60.0,
        min_sample_rate: int = 16000,
        target_sample_rate: int = 16000,
        silence_threshold: float = 0.01,
        clipping_threshold: float = 0.99,
        max_silence_ratio: float = 0.5
    ):
        """
        Initialize validator.
        
        Args:
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
            min_sample_rate: Minimum acceptable sample rate
            target_sample_rate: Target sample rate for training
            silence_threshold: RMS below this is considered silence
            clipping_threshold: Amplitude above this is considered clipped
            max_silence_ratio: Maximum ratio of silent frames
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_sample_rate = min_sample_rate
        self.target_sample_rate = target_sample_rate
        self.silence_threshold = silence_threshold
        self.clipping_threshold = clipping_threshold
        self.max_silence_ratio = max_silence_ratio
    
    def validate(self, audio_path: Union[str, Path]) -> ValidationResult:
        """
        Validate a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            ValidationResult with status and issues
        """
        audio_path = Path(audio_path)
        issues = []
        
        # Check file exists
        if not audio_path.exists():
            return ValidationResult(
                is_valid=False,
                audio_path=str(audio_path),
                issues=["File not found"]
            )
        
        # Try to load audio info
        try:
            info = torchaudio.info(str(audio_path))
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                audio_path=str(audio_path),
                issues=[f"Cannot read file: {e}"]
            )
        
        duration = info.num_frames / info.sample_rate
        sample_rate = info.sample_rate
        channels = info.num_channels
        
        # Check duration
        if duration < self.min_duration:
            issues.append(f"Too short: {duration:.2f}s < {self.min_duration}s")
        
        if duration > self.max_duration:
            issues.append(f"Too long: {duration:.2f}s > {self.max_duration}s")
        
        # Check sample rate
        if sample_rate < self.min_sample_rate:
            issues.append(f"Low sample rate: {sample_rate}Hz < {self.min_sample_rate}Hz")
        
        # Load audio for quality checks (only if no issues so far)
        if not issues:
            try:
                waveform, sr = torchaudio.load(str(audio_path))
                
                # Check for silence
                silence_issues = self._check_silence(waveform)
                issues.extend(silence_issues)
                
                # Check for clipping
                clipping_issues = self._check_clipping(waveform)
                issues.extend(clipping_issues)
                
            except Exception as e:
                issues.append(f"Cannot load audio data: {e}")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            audio_path=str(audio_path),
            duration=duration,
            sample_rate=sample_rate,
            channels=channels,
            issues=issues
        )
    
    def _check_silence(self, waveform: torch.Tensor) -> List[str]:
        """Check for excessive silence."""
        issues = []
        
        # Compute RMS in windows
        window_size = 1600  # 100ms at 16kHz
        
        # Flatten to mono for analysis
        if waveform.dim() > 1:
            audio = waveform.mean(dim=0)
        else:
            audio = waveform
        
        # Compute RMS per window
        num_windows = len(audio) // window_size
        if num_windows == 0:
            return issues
        
        silent_windows = 0
        for i in range(num_windows):
            window = audio[i * window_size:(i + 1) * window_size]
            rms = torch.sqrt(torch.mean(window ** 2))
            if rms < self.silence_threshold:
                silent_windows += 1
        
        silence_ratio = silent_windows / num_windows
        
        if silence_ratio > self.max_silence_ratio:
            issues.append(f"Too much silence: {silence_ratio:.1%} silent")
        
        # Check if completely silent
        total_rms = torch.sqrt(torch.mean(audio ** 2))
        if total_rms < self.silence_threshold:
            issues.append("Audio is silent")
        
        return issues
    
    def _check_clipping(self, waveform: torch.Tensor) -> List[str]:
        """Check for audio clipping."""
        issues = []
        
        max_amplitude = waveform.abs().max().item()
        
        if max_amplitude > self.clipping_threshold:
            # Count clipped samples
            clipped = (waveform.abs() > self.clipping_threshold).sum().item()
            total = waveform.numel()
            clip_ratio = clipped / total
            
            if clip_ratio > 0.001:  # More than 0.1% clipped
                issues.append(f"Audio clipping detected: {clip_ratio:.2%} clipped")
        
        return issues
    
    def validate_batch(
        self,
        audio_paths: List[Union[str, Path]],
        verbose: bool = True
    ) -> Tuple[List[ValidationResult], Dict]:
        """
        Validate multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            verbose: Whether to print progress
            
        Returns:
            Tuple of (results list, summary dict)
        """
        results = []
        valid_count = 0
        invalid_count = 0
        issue_counts = {}
        
        for i, path in enumerate(audio_paths):
            if verbose and (i + 1) % 100 == 0:
                print(f"  Validated {i + 1}/{len(audio_paths)} files...")
            
            result = self.validate(path)
            results.append(result)
            
            if result.is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                for issue in result.issues:
                    # Extract issue type (first word)
                    issue_type = issue.split(':')[0]
                    issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        summary = {
            'total': len(audio_paths),
            'valid': valid_count,
            'invalid': invalid_count,
            'valid_ratio': valid_count / len(audio_paths) if audio_paths else 0,
            'issue_counts': issue_counts
        }
        
        return results, summary


class AudioQualityAnalyzer:
    """
    Analyzes audio quality metrics for reporting.
    """
    
    @staticmethod
    def analyze(audio_path: Union[str, Path]) -> Dict:
        """
        Analyze audio quality metrics.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of quality metrics
        """
        audio_path = Path(audio_path)
        
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
        except Exception as e:
            return {'error': str(e)}
        
        # Convert to mono for analysis
        if waveform.shape[0] > 1:
            audio = waveform.mean(dim=0)
        else:
            audio = waveform.squeeze(0)
        
        duration = len(audio) / sample_rate
        
        # Compute metrics
        rms = torch.sqrt(torch.mean(audio ** 2)).item()
        peak = audio.abs().max().item()
        
        # Dynamic range (difference between peak and RMS in dB)
        if rms > 0:
            dynamic_range_db = 20 * torch.log10(torch.tensor(peak / rms)).item()
        else:
            dynamic_range_db = 0
        
        # Crest factor
        crest_factor = peak / rms if rms > 0 else 0
        
        # Zero crossing rate (indicator of noisiness/high frequency content)
        zero_crossings = ((audio[:-1] * audio[1:]) < 0).sum().item()
        zcr = zero_crossings / len(audio)
        
        return {
            'duration_seconds': duration,
            'sample_rate': sample_rate,
            'channels': waveform.shape[0],
            'rms': rms,
            'peak_amplitude': peak,
            'dynamic_range_db': dynamic_range_db,
            'crest_factor': crest_factor,
            'zero_crossing_rate': zcr
        }


if __name__ == "__main__":
    print("✅ AudioValidator module loaded successfully!")
    print("")
    
    # Test validator
    print("Testing AudioValidator...")
    validator = AudioValidator(
        min_duration=1.0,
        max_duration=60.0
    )
    
    # Test with sample file if exists
    test_file = Path("data/raw/test/sample.mp3")
    
    if test_file.exists():
        result = validator.validate(test_file)
        print(f"   File: {result.audio_path}")
        print(f"   Valid: {result.is_valid}")
        print(f"   Duration: {result.duration:.2f}s")
        print(f"   Sample rate: {result.sample_rate}Hz")
        print(f"   Channels: {result.channels}")
        if result.issues:
            print(f"   Issues: {result.issues}")
        else:
            print(f"   ✅ No issues found")
        print("")
        
        # Test quality analyzer
        print("Testing AudioQualityAnalyzer...")
        metrics = AudioQualityAnalyzer.analyze(test_file)
        print(f"   RMS: {metrics['rms']:.4f}")
        print(f"   Peak: {metrics['peak_amplitude']:.4f}")
        print(f"   Dynamic range: {metrics['dynamic_range_db']:.1f}dB")
    else:
        print(f"   ⚠️  No test file found at {test_file}")
        print(f"   Creating dummy validation test...")
        
        # Test with non-existent file
        result = validator.validate("nonexistent.mp3")
        print(f"   ✅ Correctly detected missing file: {result.issues}")
