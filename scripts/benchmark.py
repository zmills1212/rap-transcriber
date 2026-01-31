"""
Benchmark suite for rap transcription model.
Provides standardized performance testing.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --checkpoint outputs/checkpoints/best.pt
    python scripts/benchmark.py --speed-only
"""
import sys
sys.path.insert(0, '.')

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import time
import json
from pathlib import Path
from datetime import datetime
import torch

from src.models.rap_transcriber import create_model
from src.inference.engine import InferenceEngine
from src.data.feature_extractor import FeatureExtractor
from src.utils.metrics import compute_batch_metrics
from src.utils.slang_metrics import SlangAccuracyEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark rap transcription model')
    
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Model checkpoint path')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu, cuda, mps)')
    parser.add_argument('--output', '-o', type=str, default='outputs/results/benchmark.json',
                        help='Output file for results')
    parser.add_argument('--speed-only', action='store_true',
                        help='Only run speed benchmarks')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations for speed tests')
    
    return parser.parse_args()


def get_device(requested: str = None) -> str:
    """Get best available device."""
    if requested:
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def benchmark_model_speed(model, device: str, iterations: int = 10) -> dict:
    """
    Benchmark model inference speed.
    
    Returns timing statistics for various input sizes.
    """
    print("\n--- Speed Benchmark ---")
    
    model.eval()
    results = {}
    
    # Test different input lengths (simulating different audio durations)
    test_configs = [
        ("5s audio", 500, 80),    # ~5 seconds
        ("10s audio", 1000, 80),  # ~10 seconds
        ("30s audio", 3000, 80),  # ~30 seconds
    ]
    
    for name, time_frames, n_mels in test_configs:
        print(f"\n   Testing {name} ({time_frames} frames)...")
        
        # Create dummy input
        dummy_input = torch.randn(1, time_frames, n_mels).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)
        
        # Timed runs
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = model(dummy_input)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                elapsed = time.perf_counter() - start
                times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Calculate real-time factor (RTF)
        # RTF < 1 means faster than real-time
        audio_duration = time_frames / 100  # Approximate seconds
        rtf = avg_time / audio_duration
        
        results[name] = {
            'avg_time_ms': avg_time * 1000,
            'min_time_ms': min_time * 1000,
            'max_time_ms': max_time * 1000,
            'rtf': rtf,
            'faster_than_realtime': rtf < 1.0
        }
        
        print(f"      Avg: {avg_time*1000:.1f}ms")
        print(f"      RTF: {rtf:.3f} ({'✅ faster' if rtf < 1 else '⚠️ slower'} than real-time)")
    
    return results


def benchmark_memory(model, device: str) -> dict:
    """Benchmark memory usage."""
    print("\n--- Memory Benchmark ---")
    
    results = {
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    }
    
    print(f"   Parameters: {results['model_parameters']:,}")
    print(f"   Model size: {results['model_size_mb']:.1f} MB")
    
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
        # Run inference to measure peak memory
        dummy_input = torch.randn(1, 1000, 80).to(device)
        with torch.no_grad():
            _ = model(dummy_input)
        
        results['peak_gpu_memory_mb'] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"   Peak GPU memory: {results['peak_gpu_memory_mb']:.1f} MB")
    
    return results


def benchmark_accuracy(checkpoint_path: str = None, device: str = 'cpu') -> dict:
    """
    Benchmark accuracy on synthetic test cases.
    
    Note: Real accuracy requires trained model and real test data.
    """
    print("\n--- Accuracy Benchmark (Synthetic) ---")
    
    # Synthetic test cases
    # In real usage, these would come from actual model predictions
    test_cases = [
        {
            'reference': "i'm finna get this bread yeah",
            'hypothesis': "i'm gonna get this bread yeah"
        },
        {
            'reference': "she bussin it down skrt skrt",
            'hypothesis': "she busting it down skirt skirt"
        },
        {
            'reference': "no cap this fire bruh",
            'hypothesis': "no cap this fire bro"
        },
        {
            'reference': "we getting money every day",
            'hypothesis': "we getting money every day"
        },
        {
            'reference': "ayy let's go yeah",
            'hypothesis': "hey let's go yeah"
        }
    ]
    
    refs = [tc['reference'] for tc in test_cases]
    hyps = [tc['hypothesis'] for tc in test_cases]
    
    # Standard metrics
    standard = compute_batch_metrics(refs, hyps)
    
    # Slang metrics
    slang_eval = SlangAccuracyEvaluator()
    slang = slang_eval.compute_batch_slang_accuracy(refs, hyps)
    
    results = {
        'wer': standard['micro_wer'],
        'cer': standard['micro_cer'],
        'word_accuracy': standard['word_accuracy'],
        'slang_recall': slang['slang_recall'],
        'slang_f1': slang['slang_f1'],
        'test_samples': len(test_cases),
        'note': 'Synthetic test data - train model for real accuracy'
    }
    
    print(f"   WER: {results['wer']:.1%}")
    print(f"   CER: {results['cer']:.1%}")
    print(f"   Word Accuracy: {results['word_accuracy']:.1%}")
    print(f"   Slang Recall: {results['slang_recall']:.1%}")
    print(f"   Slang F1: {results['slang_f1']:.1%}")
    print(f"   (Note: {results['note']})")
    
    return results


def benchmark_feature_extraction(iterations: int = 10) -> dict:
    """Benchmark feature extraction speed."""
    print("\n--- Feature Extraction Benchmark ---")
    
    extractor = FeatureExtractor()
    
    # Simulate different audio lengths
    test_configs = [
        ("5s audio", 80000),   # 5 seconds at 16kHz
        ("10s audio", 160000), # 10 seconds
        ("30s audio", 480000), # 30 seconds
    ]
    
    results = {}
    
    for name, num_samples in test_configs:
        dummy_waveform = torch.randn(1, num_samples)
        
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = extractor.extract(dummy_waveform)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        results[name] = {
            'avg_time_ms': avg_time * 1000
        }
        
        print(f"   {name}: {avg_time*1000:.1f}ms")
    
    return results


def run_full_benchmark(args) -> dict:
    """Run complete benchmark suite."""
    
    device = get_device(args.device)
    
    print("=" * 50)
    print("RAP TRANSCRIPTION BENCHMARK SUITE")
    print("=" * 50)
    print(f"\nDevice: {device}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Load or create model
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint: {args.checkpoint}")
        engine = InferenceEngine(checkpoint_path=args.checkpoint, device=device)
        model = engine.model
    else:
        print("Using default model (untrained)")
        model = create_model().to(device)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'checkpoint': args.checkpoint
    }
    
    # Memory benchmark
    results['memory'] = benchmark_memory(model, device)
    
    # Speed benchmark
    results['speed'] = benchmark_model_speed(model, device, args.iterations)
    
    # Feature extraction benchmark
    results['feature_extraction'] = benchmark_feature_extraction(args.iterations)
    
    # Accuracy benchmark (unless speed-only)
    if not args.speed_only:
        results['accuracy'] = benchmark_accuracy(args.checkpoint, device)
    
    return results


def print_summary(results: dict):
    """Print benchmark summary."""
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    
    print(f"\n📊 Model:")
    print(f"   Parameters: {results['memory']['model_parameters']:,}")
    print(f"   Size: {results['memory']['model_size_mb']:.1f} MB")
    
    print(f"\n⚡ Speed (10s audio):")
    if '10s audio' in results['speed']:
        speed = results['speed']['10s audio']
        print(f"   Inference: {speed['avg_time_ms']:.1f}ms")
        print(f"   Real-time factor: {speed['rtf']:.3f}x")
        if speed['faster_than_realtime']:
            print(f"   ✅ Faster than real-time!")
    
    if 'accuracy' in results:
        print(f"\n🎯 Accuracy (synthetic):")
        acc = results['accuracy']
        print(f"   WER: {acc['wer']:.1%}")
        print(f"   Slang F1: {acc['slang_f1']:.1%}")
    
    print("\n" + "=" * 50)


def save_results(results: dict, output_path: str):
    """Save benchmark results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    args = parse_args()
    
    results = run_full_benchmark(args)
    
    print_summary(results)
    
    save_results(results, args.output)
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
