"""
Performance optimization utilities for rap transcription.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict
import time
from functools import wraps


def get_optimal_device() -> str:
    """Get the optimal device for inference."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def optimize_model_for_inference(model: nn.Module) -> nn.Module:
    """
    Optimize model for inference.
    
    Applies:
        - Evaluation mode
        - Gradient disabling
        - Optional: torch.compile (PyTorch 2.0+)
    """
    model.eval()
    
    # Disable gradients
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def compile_model(model: nn.Module, backend: str = 'inductor') -> nn.Module:
    """
    Compile model with torch.compile for faster inference.
    
    Only works with PyTorch 2.0+
    """
    if hasattr(torch, 'compile'):
        try:
            compiled = torch.compile(model, backend=backend)
            print(f"Model compiled with backend: {backend}")
            return compiled
        except Exception as e:
            print(f"Compilation failed: {e}")
            return model
    else:
        print("torch.compile not available (requires PyTorch 2.0+)")
        return model


@torch.inference_mode()
def fast_inference(model: nn.Module, inputs: torch.Tensor) -> Dict:
    """
    Perform fast inference with optimizations.
    
    Uses inference_mode for maximum speed.
    """
    return model(inputs)


class InferenceOptimizer:
    """
    Manages inference optimization settings.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        use_compile: bool = False,
        use_half_precision: bool = False
    ):
        self.device = device or get_optimal_device()
        self.use_half = use_half_precision and self.device != 'cpu'
        
        # Prepare model
        self.model = optimize_model_for_inference(model)
        self.model = self.model.to(self.device)
        
        # Half precision
        if self.use_half:
            self.model = self.model.half()
            print("Using half precision (FP16)")
        
        # Compile
        if use_compile:
            self.model = compile_model(self.model)
        
        # Warmup
        self._warmup()
    
    def _warmup(self, iterations: int = 3):
        """Warmup model to optimize memory allocation."""
        dummy = torch.randn(1, 100, 80).to(self.device)
        if self.use_half:
            dummy = dummy.half()
        
        for _ in range(iterations):
            with torch.inference_mode():
                _ = self.model(dummy)
        
        # Clear cache
        if self.device == 'cuda':
            torch.cuda.empty_cache()
    
    @torch.inference_mode()
    def __call__(self, inputs: torch.Tensor) -> Dict:
        """Run optimized inference."""
        inputs = inputs.to(self.device)
        if self.use_half:
            inputs = inputs.half()
        return self.model(inputs)
    
    def benchmark(self, input_shape: tuple = (1, 500, 80), iterations: int = 10) -> Dict:
        """Benchmark inference speed."""
        dummy = torch.randn(*input_shape).to(self.device)
        if self.use_half:
            dummy = dummy.half()
        
        # Warmup
        for _ in range(3):
            _ = self(dummy)
        
        # Timed runs
        times = []
        for _ in range(iterations):
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = self(dummy)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            times.append(time.perf_counter() - start)
        
        return {
            'avg_ms': sum(times) / len(times) * 1000,
            'min_ms': min(times) * 1000,
            'max_ms': max(times) * 1000,
            'device': self.device,
            'half_precision': self.use_half
        }


def timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed*1000:.1f}ms")
        return result
    return wrapper


class BatchProcessor:
    """
    Process multiple inputs efficiently in batches.
    """
    
    def __init__(self, model: nn.Module, batch_size: int = 8, device: str = 'cpu'):
        self.model = optimize_model_for_inference(model).to(device)
        self.batch_size = batch_size
        self.device = device
    
    @torch.inference_mode()
    def process(self, inputs: list) -> list:
        """Process list of inputs in batches."""
        results = []
        
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            
            # Stack into tensor
            batch_tensor = torch.stack(batch).to(self.device)
            
            # Process
            outputs = self.model(batch_tensor)
            
            # Unpack results
            for j in range(len(batch)):
                results.append({
                    k: v[j] for k, v in outputs.items()
                })
        
        return results


if __name__ == "__main__":
    print("✅ Optimization module loaded successfully!")
    print("")
    
    # Test optimal device
    device = get_optimal_device()
    print(f"Optimal device: {device}")
    print("")
    
    # Test with model
    from src.models.rap_transcriber import create_model
    
    print("Testing InferenceOptimizer...")
    model = create_model({
        'encoder_dim': 128,
        'encoder_layers': 2,
        'encoder_heads': 4
    })
    
    optimizer = InferenceOptimizer(model, device='cpu')
    
    # Benchmark
    results = optimizer.benchmark(iterations=5)
    print(f"   Avg inference: {results['avg_ms']:.1f}ms")
    print(f"   Device: {results['device']}")
    print("")
    
    print("✅ Optimization utilities ready!")
