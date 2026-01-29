"""
Main inference script for rap transcription.

Usage:
    python scripts/transcribe.py audio.mp3
    python scripts/transcribe.py audio.mp3 --output transcript.txt
    python scripts/transcribe.py audio.mp3 --format-lyrics
    python scripts/transcribe.py audio_folder/ --batch
"""
import sys
sys.path.insert(0, '.')

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
from pathlib import Path
from typing import List
import time

from src.inference.engine import InferenceEngine
from src.inference.postprocessor import TranscriptionCleaner, TextPostProcessor, LyricsFormatter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Transcribe rap audio files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Transcribe single file
    python scripts/transcribe.py my_song.mp3
    
    # Save output to file
    python scripts/transcribe.py my_song.mp3 --output transcript.txt
    
    # Format as lyrics
    python scripts/transcribe.py my_song.mp3 --format-lyrics
    
    # Batch process folder
    python scripts/transcribe.py music_folder/ --batch
    
    # Use specific checkpoint
    python scripts/transcribe.py my_song.mp3 --checkpoint outputs/checkpoints/best.pt
        """
    )
    
    # Input
    parser.add_argument('input', type=str,
                        help='Audio file or folder to transcribe')
    
    # Output
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (prints to console if not specified)')
    parser.add_argument('--output-dir', type=str, default='outputs/results',
                        help='Output directory for batch mode')
    
    # Model
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Model checkpoint path')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu, cuda, mps)')
    
    # Processing options
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Batch process all audio files in folder')
    parser.add_argument('--format-lyrics', '-l', action='store_true',
                        help='Format output as lyrics with line breaks')
    parser.add_argument('--normalize-slang', action='store_true',
                        help='Expand slang to standard English')
    parser.add_argument('--censor', action='store_true',
                        help='Censor profanity in output')
    
    # Decoding options
    parser.add_argument('--beam-size', type=int, default=10,
                        help='Beam size for decoding')
    
    # Output options
    parser.add_argument('--show-phonemes', action='store_true',
                        help='Also show phoneme predictions')
    parser.add_argument('--show-timing', action='store_true',
                        help='Show timing information')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')
    
    return parser.parse_args()


def find_audio_files(folder: Path) -> List[Path]:
    """Find all audio files in folder."""
    extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.wma'}
    files = []
    
    for ext in extensions:
        files.extend(folder.glob(f'*{ext}'))
        files.extend(folder.glob(f'*{ext.upper()}'))
    
    return sorted(files)


def transcribe_single(
    engine: InferenceEngine,
    cleaner: TranscriptionCleaner,
    audio_path: Path,
    args
) -> dict:
    """Transcribe a single audio file."""
    
    # Transcribe
    result = engine.transcribe(
        audio_path,
        return_phonemes=args.show_phonemes,
        return_timing=args.show_timing
    )
    
    # Get raw text (might be token IDs without tokenizer)
    raw_text = result['text']
    
    # If it's a list of token IDs, join them (placeholder)
    if isinstance(raw_text, list):
        raw_text = ' '.join(map(str, raw_text))
    
    # Post-process
    cleaned_text = cleaner.clean(
        raw_text,
        format_as_lyrics=args.format_lyrics
    )
    
    result['cleaned_text'] = cleaned_text
    return result


def print_result(result: dict, args):
    """Print transcription result."""
    print("")
    print("=" * 50)
    print(f"File: {result['audio_path']}")
    print("=" * 50)
    print("")
    print(result['cleaned_text'])
    print("")
    
    if args.show_phonemes and 'phonemes' in result:
        print("-" * 50)
        print("Phonemes:")
        print(' '.join(result['phonemes'][:50]), "...")
        print("")
    
    if args.show_timing and 'timing' in result:
        timing = result['timing']
        print("-" * 50)
        print(f"Timing:")
        print(f"  Preprocess: {timing['preprocess']:.3f}s")
        print(f"  Inference:  {timing['inference']:.3f}s")
        print(f"  Decode:     {timing['decode']:.3f}s")
        print(f"  Total:      {timing['total']:.3f}s")
        print("")


def save_result(result: dict, output_path: Path, args):
    """Save transcription result to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"# Transcription: {result['audio_path']}\n")
        f.write(f"# Generated by Rap Transcriber\n")
        f.write("\n")
        f.write(result['cleaned_text'])
        f.write("\n")
        
        if args.show_phonemes and 'phonemes' in result:
            f.write("\n## Phonemes\n")
            f.write(' '.join(result['phonemes']))
            f.write("\n")
    
    print(f"Saved: {output_path}")


def main():
    """Main transcription function."""
    args = parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)
    
    # Initialize engine
    if not args.quiet:
        print("Initializing transcription engine...")
    
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    if not args.quiet:
        info = engine.get_model_info()
        print(f"  Device: {info['device']}")
        print(f"  Parameters: {info['parameters']:,}")
    
    # Initialize cleaner
    postprocessor = TextPostProcessor(
        normalize_slang=args.normalize_slang,
        censor_profanity=args.censor
    )
    formatter = LyricsFormatter() if args.format_lyrics else None
    cleaner = TranscriptionCleaner(postprocessor, formatter)
    
    # Process files
    if args.batch or input_path.is_dir():
        # Batch mode
        if not input_path.is_dir():
            print(f"Error: {input_path} is not a directory")
            sys.exit(1)
        
        audio_files = find_audio_files(input_path)
        
        if not audio_files:
            print(f"No audio files found in: {input_path}")
            sys.exit(1)
        
        if not args.quiet:
            print(f"Found {len(audio_files)} audio files")
        
        output_dir = Path(args.output_dir)
        
        for i, audio_file in enumerate(audio_files):
            if not args.quiet:
                print(f"\n[{i+1}/{len(audio_files)}] Processing: {audio_file.name}")
            
            try:
                result = transcribe_single(engine, cleaner, audio_file, args)
                
                # Save to output directory
                output_path = output_dir / f"{audio_file.stem}.txt"
                save_result(result, output_path, args)
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        print(f"\nCompleted! Results saved to: {output_dir}")
    
    else:
        # Single file mode
        if not args.quiet:
            print(f"Transcribing: {input_path}")
        
        result = transcribe_single(engine, cleaner, input_path, args)
        
        # Output
        if args.output:
            save_result(result, Path(args.output), args)
        else:
            print_result(result, args)


if __name__ == "__main__":
    main()
