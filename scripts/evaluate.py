"""
Evaluation script for rap transcription model.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt --test-manifest data/processed/test_manifest.json
"""
import sys
sys.path.insert(0, '.')

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import json
from pathlib import Path
from typing import Dict, List
import torch
from tqdm import tqdm

from src.data.manifest import DataManifest
from src.data.audio_processor import AudioProcessor
from src.data.feature_extractor import FeatureExtractor
from src.inference.engine import InferenceEngine
from src.inference.postprocessor import TranscriptionCleaner
from src.utils.metrics import compute_metrics, compute_batch_metrics, TranscriptionMetrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate rap transcription model')
    
    # Model
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Model checkpoint path')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu, cuda, mps)')
    
    # Data
    parser.add_argument('--test-manifest', type=str, 
                        default='data/processed/test_manifest.json',
                        help='Test manifest path')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to evaluate')
    
    # Output
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file for detailed results')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed results')
    
    return parser.parse_args()


def evaluate_model(
    engine: InferenceEngine,
    manifest: DataManifest,
    cleaner: TranscriptionCleaner,
    max_samples: int = None,
    verbose: bool = False
) -> Dict:
    """
    Evaluate model on test set.
    
    Args:
        engine: Inference engine
        manifest: Test data manifest
        cleaner: Text post-processor
        max_samples: Maximum samples to evaluate
        verbose: Whether to print per-sample results
        
    Returns:
        Dictionary with evaluation results
    """
    references = []
    hypotheses = []
    sample_results = []
    
    samples = manifest.entries[:max_samples] if max_samples else manifest.entries
    
    print(f"\nEvaluating {len(samples)} samples...")
    
    for entry in tqdm(samples, desc="Evaluating"):
        try:
            # Get reference text
            reference = entry.text
            
            # Transcribe
            result = engine.transcribe(entry.audio_path)
            
            # Get hypothesis (might be token IDs)
            hypothesis = result['text']
            if isinstance(hypothesis, list):
                # Convert token IDs to string for now
                hypothesis = ' '.join(map(str, hypothesis))
            
            # Clean hypothesis
            hypothesis = cleaner.clean(hypothesis, format_as_lyrics=False)
            
            # Compute per-sample metrics
            metrics = compute_metrics(reference, hypothesis)
            
            references.append(reference)
            hypotheses.append(hypothesis)
            
            sample_result = {
                'audio_path': entry.audio_path,
                'reference': reference,
                'hypothesis': hypothesis,
                'wer': metrics.wer,
                'cer': metrics.cer
            }
            sample_results.append(sample_result)
            
            if verbose:
                print(f"\n  File: {Path(entry.audio_path).name}")
                print(f"  Ref: {reference[:50]}...")
                print(f"  Hyp: {hypothesis[:50]}...")
                print(f"  WER: {metrics.wer:.2%}")
                
        except Exception as e:
            print(f"\n  Error processing {entry.audio_path}: {e}")
            continue
    
    # Compute aggregate metrics
    if references:
        batch_metrics = compute_batch_metrics(references, hypotheses)
    else:
        batch_metrics = {
            'micro_wer': 1.0,
            'micro_cer': 1.0,
            'macro_wer': 1.0,
            'macro_cer': 1.0,
            'total_samples': 0,
            'word_accuracy': 0.0
        }
    
    return {
        'metrics': batch_metrics,
        'samples': sample_results
    }


def print_results(results: Dict):
    """Print evaluation results."""
    metrics = results['metrics']
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    
    print(f"\nSamples evaluated: {metrics['total_samples']}")
    
    print("\n--- Word Error Rate (WER) ---")
    print(f"  Micro WER: {metrics['micro_wer']:.2%}")
    print(f"  Macro WER: {metrics['macro_wer']:.2%}")
    
    print("\n--- Character Error Rate (CER) ---")
    print(f"  Micro CER: {metrics['micro_cer']:.2%}")
    print(f"  Macro CER: {metrics['macro_cer']:.2%}")
    
    print("\n--- Accuracy ---")
    print(f"  Word Accuracy: {metrics['word_accuracy']:.2%}")
    print(f"  Char Accuracy: {metrics['char_accuracy']:.2%}")
    
    # Show best and worst samples
    samples = results['samples']
    if samples:
        sorted_by_wer = sorted(samples, key=lambda x: x['wer'])
        
        print("\n--- Best Samples (Lowest WER) ---")
        for s in sorted_by_wer[:3]:
            print(f"  {Path(s['audio_path']).name}: WER={s['wer']:.2%}")
        
        print("\n--- Worst Samples (Highest WER) ---")
        for s in sorted_by_wer[-3:]:
            print(f"  {Path(s['audio_path']).name}: WER={s['wer']:.2%}")
    
    print("\n" + "=" * 50)


def save_results(results: Dict, output_path: str):
    """Save detailed results to JSON."""
    # Convert metrics objects to dicts
    output = {
        'metrics': {
            k: v for k, v in results['metrics'].items()
            if k != 'sample_metrics'  # Exclude detailed per-sample metrics objects
        },
        'samples': results['samples']
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("=" * 50)
    print("RAP TRANSCRIPTION MODEL EVALUATION")
    print("=" * 50)
    
    # Load test manifest
    manifest_path = Path(args.test_manifest)
    if not manifest_path.exists():
        print(f"\n❌ Test manifest not found: {manifest_path}")
        print("\nRun 'python scripts/create_sample_dataset.py' first.")
        return
    
    print(f"\nTest manifest: {manifest_path}")
    manifest = DataManifest.load(manifest_path)
    print(f"Test samples: {len(manifest)}")
    
    # Initialize engine
    print("\nInitializing model...")
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    info = engine.get_model_info()
    print(f"  Device: {info['device']}")
    print(f"  Parameters: {info['parameters']:,}")
    
    # Initialize cleaner
    cleaner = TranscriptionCleaner()
    
    # Evaluate
    results = evaluate_model(
        engine=engine,
        manifest=manifest,
        cleaner=cleaner,
        max_samples=args.max_samples,
        verbose=args.verbose
    )
    
    # Print results
    print_results(results)
    
    # Save results if output specified
    if args.output:
        save_results(results, args.output)
    
    # Interpretation
    wer = results['metrics']['micro_wer']
    print("\n--- Interpretation ---")
    if wer < 0.1:
        print("  🏆 Excellent! WER < 10%")
    elif wer < 0.2:
        print("  ✅ Good. WER < 20%")
    elif wer < 0.3:
        print("  📈 Fair. WER < 30% - Room for improvement")
    elif wer < 0.5:
        print("  ⚠️  Needs work. WER < 50%")
    else:
        print("  🔧 Model needs training. WER >= 50%")
        print("     (This is expected for an untrained model)")


if __name__ == "__main__":
    main()
