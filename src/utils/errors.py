"""
Custom exceptions and error handling for rap transcription.
"""
from typing import Optional, Dict, Any
from functools import wraps
import traceback


# ============== Custom Exceptions ==============

class RapTranscriberError(Exception):
    """Base exception for rap transcriber."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict:
        return {
            'error': self.__class__.__name__,
            'message': self.message,
            'details': self.details
        }


class AudioProcessingError(RapTranscriberError):
    """Error during audio processing."""
    pass


class ModelError(RapTranscriberError):
    """Error with model operations."""
    pass


class InferenceError(RapTranscriberError):
    """Error during inference."""
    pass


class ConfigError(RapTranscriberError):
    """Error with configuration."""
    pass


class DataError(RapTranscriberError):
    """Error with data loading or processing."""
    pass


class ValidationError(RapTranscriberError):
    """Error with input validation."""
    pass


class CheckpointError(RapTranscriberError):
    """Error loading or saving checkpoints."""
    pass


# ============== Error Handlers ==============

def handle_audio_error(func):
    """Decorator to handle audio processing errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise AudioProcessingError(
                f"Audio file not found: {e}",
                {'original_error': str(e)}
            )
        except PermissionError as e:
            raise AudioProcessingError(
                f"Permission denied accessing audio file: {e}",
                {'original_error': str(e)}
            )
        except Exception as e:
            if "audio" in str(e).lower() or "sound" in str(e).lower():
                raise AudioProcessingError(
                    f"Audio processing failed: {e}",
                    {'original_error': str(e)}
                )
            raise
    return wrapper


def handle_model_error(func):
    """Decorator to handle model errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                raise ModelError(
                    f"GPU memory error: {e}",
                    {'suggestion': 'Try reducing batch size or using CPU'}
                )
            raise ModelError(f"Model error: {e}")
        except Exception as e:
            raise ModelError(f"Model operation failed: {e}")
    return wrapper


def handle_inference_error(func):
    """Decorator to handle inference errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AudioProcessingError:
            raise
        except ModelError:
            raise
        except Exception as e:
            raise InferenceError(
                f"Inference failed: {e}",
                {'traceback': traceback.format_exc()}
            )
    return wrapper


def safe_execute(func, *args, default=None, **kwargs):
    """
    Safely execute a function, returning default on error.
    
    Usage:
        result = safe_execute(risky_function, arg1, arg2, default="fallback")
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Warning: {func.__name__} failed: {e}")
        return default


class ErrorHandler:
    """
    Context manager for error handling with cleanup.
    
    Usage:
        with ErrorHandler("Processing audio") as handler:
            # risky operations
            handler.set_result(result)
        
        if handler.success:
            print(handler.result)
        else:
            print(handler.error)
    """
    
    def __init__(self, operation_name: str, reraise: bool = False):
        self.operation_name = operation_name
        self.reraise = reraise
        self.success = False
        self.result = None
        self.error = None
        self.error_type = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.success = True
        else:
            self.success = False
            self.error = str(exc_val)
            self.error_type = exc_type.__name__
            
            if not self.reraise:
                print(f"Error in {self.operation_name}: {self.error}")
                return True  # Suppress exception
        
        return False
    
    def set_result(self, result):
        """Set the result of the operation."""
        self.result = result


def validate_audio_path(path) -> str:
    """Validate audio file path."""
    from pathlib import Path
    
    path = Path(path)
    
    if not path.exists():
        raise ValidationError(f"File not found: {path}")
    
    valid_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
    if path.suffix.lower() not in valid_extensions:
        raise ValidationError(
            f"Unsupported audio format: {path.suffix}",
            {'supported': list(valid_extensions)}
        )
    
    return str(path)


def validate_config(config: Dict) -> Dict:
    """Validate configuration dictionary."""
    required_keys = ['model', 'training', 'audio']
    
    for key in required_keys:
        if key not in config:
            raise ConfigError(f"Missing required config key: {key}")
    
    return config


def format_error_message(error: Exception) -> str:
    """Format error message for user display."""
    if isinstance(error, RapTranscriberError):
        msg = f"[{error.__class__.__name__}] {error.message}"
        if error.details:
            msg += f"\nDetails: {error.details}"
        return msg
    else:
        return f"[Error] {str(error)}"


if __name__ == "__main__":
    print("✅ Error handling module loaded successfully!")
    print("")
    
    # Test custom exceptions
    print("Testing custom exceptions...")
    
    try:
        raise AudioProcessingError("Test audio error", {'file': 'test.mp3'})
    except RapTranscriberError as e:
        print(f"   ✅ Caught: {e.to_dict()}")
    
    # Test error handler context manager
    print("\nTesting ErrorHandler...")
    
    with ErrorHandler("test operation") as handler:
        result = 1 + 1
        handler.set_result(result)
    
    print(f"   ✅ Success: {handler.success}, Result: {handler.result}")
    
    # Test with error
    with ErrorHandler("failing operation") as handler:
        raise ValueError("test error")
    
    print(f"   ✅ Handled error: {handler.error_type}")
    
    # Test validation
    print("\nTesting validation...")
    
    try:
        validate_audio_path("nonexistent.mp3")
    except ValidationError as e:
        print(f"   ✅ Validation caught: {e.message}")
    
    print("\n✅ Error handling ready!")
