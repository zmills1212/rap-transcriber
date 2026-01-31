"""
Logging configuration for rap transcription system.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


# ============== Log Formatters ==============

SIMPLE_FORMAT = "%(levelname)s: %(message)s"
DETAILED_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEBUG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"


# ============== Logger Setup ==============

def setup_logger(
    name: str = "rap_transcriber",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    detailed: bool = False
) -> logging.Logger:
    """
    Setup and configure a logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
        console: Whether to log to console
        detailed: Use detailed format
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Choose format
    if level == logging.DEBUG:
        fmt = DEBUG_FORMAT
    elif detailed:
        fmt = DETAILED_FORMAT
    else:
        fmt = SIMPLE_FORMAT
    
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(DETAILED_FORMAT))
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "rap_transcriber") -> logging.Logger:
    """Get an existing logger or create a new one."""
    logger = logging.getLogger(name)
    
    # If no handlers, set up default
    if not logger.handlers:
        setup_logger(name)
    
    return logger


# ============== Specialized Loggers ==============

class TrainingLogger:
    """Logger specialized for training progress."""
    
    def __init__(self, log_dir: str = "outputs/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"training_{timestamp}.log"
        
        self.logger = setup_logger(
            name="training",
            level=logging.INFO,
            log_file=str(log_file),
            detailed=True
        )
        
        self.epoch = 0
        self.step = 0
    
    def log_epoch_start(self, epoch: int):
        """Log epoch start."""
        self.epoch = epoch
        self.logger.info(f"{'='*50}")
        self.logger.info(f"EPOCH {epoch} STARTED")
        self.logger.info(f"{'='*50}")
    
    def log_epoch_end(self, epoch: int, metrics: dict):
        """Log epoch end with metrics."""
        self.logger.info(f"EPOCH {epoch} COMPLETED")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_step(self, step: int, loss: float, lr: float):
        """Log training step."""
        self.step = step
        self.logger.debug(f"Step {step}: loss={loss:.4f}, lr={lr:.2e}")
    
    def log_validation(self, metrics: dict):
        """Log validation results."""
        self.logger.info("Validation Results:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_checkpoint(self, path: str):
        """Log checkpoint save."""
        self.logger.info(f"Checkpoint saved: {path}")
    
    def log_error(self, message: str, exc_info: bool = True):
        """Log error."""
        self.logger.error(message, exc_info=exc_info)


class InferenceLogger:
    """Logger specialized for inference."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.logger = setup_logger(
            name="inference",
            level=logging.INFO,
            log_file=log_file
        )
    
    def log_start(self, audio_path: str):
        """Log inference start."""
        self.logger.info(f"Transcribing: {audio_path}")
    
    def log_complete(self, audio_path: str, duration: float):
        """Log inference completion."""
        self.logger.info(f"Completed: {audio_path} ({duration:.2f}s)")
    
    def log_result(self, text: str, truncate: int = 100):
        """Log transcription result."""
        display = text[:truncate] + "..." if len(text) > truncate else text
        self.logger.info(f"Result: {display}")
    
    def log_timing(self, preprocess: float, inference: float, decode: float):
        """Log timing breakdown."""
        self.logger.debug(f"Timing - preprocess: {preprocess:.3f}s, "
                         f"inference: {inference:.3f}s, decode: {decode:.3f}s")


class APILogger:
    """Logger specialized for API requests."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.logger = setup_logger(
            name="api",
            level=logging.INFO,
            log_file=log_file
        )
    
    def log_request(self, method: str, path: str, client: str = "unknown"):
        """Log incoming request."""
        self.logger.info(f"{method} {path} from {client}")
    
    def log_response(self, status: int, duration: float):
        """Log response."""
        self.logger.info(f"Response: {status} ({duration:.3f}s)")
    
    def log_error(self, error: str, status: int = 500):
        """Log error response."""
        self.logger.error(f"Error {status}: {error}")


# ============== Utility Functions ==============

def log_system_info():
    """Log system information."""
    import torch
    
    logger = get_logger()
    
    logger.info("System Information:")
    logger.info(f"  Python: {sys.version.split()[0]}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    logger.info(f"  MPS available: {torch.backends.mps.is_available()}")


def log_model_info(model):
    """Log model information."""
    logger = get_logger()
    
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("Model Information:")
    logger.info(f"  Total parameters: {params:,}")
    logger.info(f"  Trainable parameters: {trainable:,}")


if __name__ == "__main__":
    print("✅ Logging module loaded successfully!")
    print("")
    
    # Test basic logger
    print("Testing basic logger...")
    logger = setup_logger("test", level=logging.DEBUG)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    print("")
    
    # Test training logger
    print("Testing TrainingLogger...")
    train_logger = TrainingLogger()
    train_logger.log_epoch_start(1)
    train_logger.log_epoch_end(1, {'loss': 0.5, 'accuracy': 0.85})
    print("")
    
    # Test inference logger
    print("Testing InferenceLogger...")
    inf_logger = InferenceLogger()
    inf_logger.log_start("test.mp3")
    inf_logger.log_complete("test.mp3", 1.5)
    inf_logger.log_result("i'm finna get this bread yeah")
    print("")
    
    # Test system info
    print("Testing system info logging...")
    log_system_info()
    
    print("\n✅ Logging system ready!")
