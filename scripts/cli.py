#!/usr/bin/env python3
"""
Rap Transcription CLI Tool

Usage:
    rap-transcribe audio.mp3
    rap-transcribe audio.mp3 --lyrics
    rap-transcribe --server
    rap-transcribe --help
"""
import sys
sys.path.insert(0, '.')

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
from pathlib import Path


def transcribe_command(args):
    """Handle transcribe command."""
    from src.inference.engine import InferenceEngine
    from src.inference.postprocessor import TranscriptionCleaner
    
    audio_path = Path(args.input)
    
    if not audio_path.exists():
        print(f"❌ File not found: {audio_path}")
        sys.exit(1)
    
    print(f"🎤 Transcribing: {audio_path.name}")
    print("")
    
    # Initialize
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    cleaner = TranscriptionCleaner()
    
    # Transcribe
    result = engine.transcribe(
        audio_path,
        return_phonemes=args.phonemes,
        return_timing=True
    )
    
    # Get text
    text = result['text']
    if isinstance(text, list):
        text = ' '.join(map(str, text))
    
    # Clean
    cleaned = cleaner.clean(text, format_as_lyrics=args.lyrics)
    
    # Output
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            f.write(cleaned)
        print(f"✅ Saved to: {output_path}")
    else:
        print("─" * 40)
        print(cleaned)
        print("─" * 40)
    
    # Timing
    if args.timing and 'timing' in result:
        t = result['timing']
        print(f"\n⏱️  Processing time: {t['total']:.2f}s")


def server_command(args):
    """Handle server command."""
    import uvicorn
    
    print("🚀 Starting Rap Transcription API Server")
    print(f"   URL: http://{args.host}:{args.port}")
    print(f"   Docs: http://{args.host}:{args.port}/docs")
    print("")
    
    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


def evaluate_command(args):
    """Handle evaluate command."""
    from scripts.evaluate import main as eval_main
    sys.argv = ['evaluate.py', '--test-manifest', args.manifest]
    if args.checkpoint:
        sys.argv.extend(['--checkpoint', args.checkpoint])
    eval_main()


def benchmark_command(args):
    """Handle benchmark command."""
    from scripts.benchmark import main as bench_main
    sys.argv = ['benchmark.py', '--iterations', str(args.iterations)]
    bench_main()


def info_command(args):
    """Show system info."""
    import torch
    
    print("=" * 40)
    print("🎤 RAP TRANSCRIPTION SYSTEM")
    print("=" * 40)
    print("")
    
    # PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Model info
    try:
        from src.models.rap_transcriber import create_model
        model = create_model()
        params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {params:,}")
    except Exception as e:
        print(f"Model: Error loading - {e}")
    
    # Check data
    data_dir = Path("data/processed")
    if data_dir.exists():
        manifests = list(data_dir.glob("*_manifest.json"))
        print(f"Data manifests: {len(manifests)}")
    
    # Check checkpoints
    ckpt_dir = Path("outputs/checkpoints")
    if ckpt_dir.exists():
        checkpoints = list(ckpt_dir.glob("*.pt"))
        print(f"Checkpoints: {len(checkpoints)}")
    
    print("")


def main():
    parser = argparse.ArgumentParser(
        prog='rap-transcribe',
        description='🎤 AI-powered rap transcription with slang recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.mp3                  Transcribe audio file
  %(prog)s audio.mp3 --lyrics         Format as lyrics
  %(prog)s audio.mp3 -o output.txt    Save to file
  %(prog)s --server                   Start API server
  %(prog)s --info                     Show system info
        """
    )
    
    # Global options
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    
    # Subcommands via mutually exclusive args
    parser.add_argument('input', nargs='?', help='Audio file to transcribe')
    parser.add_argument('--server', action='store_true', help='Start API server')
    parser.add_argument('--info', action='store_true', help='Show system info')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')
    
    # Transcription options
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-l', '--lyrics', action='store_true', help='Format as lyrics')
    parser.add_argument('-p', '--phonemes', action='store_true', help='Show phonemes')
    parser.add_argument('-t', '--timing', action='store_true', help='Show timing')
    parser.add_argument('-c', '--checkpoint', help='Model checkpoint path')
    parser.add_argument('-d', '--device', help='Device (cpu, cuda, mps)')
    
    # Server options
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    # Evaluation options
    parser.add_argument('--manifest', default='data/processed/test_manifest.json',
                        help='Test manifest for evaluation')
    
    # Benchmark options
    parser.add_argument('--iterations', type=int, default=10,
                        help='Benchmark iterations')
    
    args = parser.parse_args()
    
    # Route to appropriate command
    if args.server:
        server_command(args)
    elif args.info:
        info_command(args)
    elif args.evaluate:
        evaluate_command(args)
    elif args.benchmark:
        benchmark_command(args)
    elif args.input:
        transcribe_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
