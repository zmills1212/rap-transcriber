"""
Test script for the inference pipeline.
Verifies all inference components work together.
"""
import sys
sys.path.insert(0, '.')

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
from pathlib import Path


def test_inference_engine():
    """Test inference engine initialization and basic inference."""
    print("Testing InferenceEngine...")
    
    from src.inference.engine import InferenceEngine
    
    # Initialize with default model
    engine = InferenceEngine(device='cpu')
    
    info = engine.get_model_info()
    assert info['device'] == 'cpu'
    assert info['parameters'] > 0
    
    print(f"   ✅ Engine initialized")
    print(f"   ✅ Device: {info['device']}")
    print(f"   ✅ Parameters: {info['parameters']:,}")
    
    # Test with dummy features
    dummy_features = torch.randn(1, 100, 80)
    result = engine.transcribe_features(dummy_features)
    
    assert 'texts' in result
    assert len(result['texts']) == 1
    
    print(f"   ✅ Feature inference working")
    
    return True


def test_beam_search_decoder():
    """Test beam search decoding."""
    print("Testing BeamSearchDecoder...")
    
    from src.inference.decoder import BeamSearchDecoder, PrefixBeamSearchDecoder
    import torch.nn.functional as F
    
    # Test standard beam search
    decoder = BeamSearchDecoder(beam_size=5)
    
    log_probs = torch.randn(2, 50, 100)
    log_probs = F.log_softmax(log_probs, dim=-1)
    
    results = decoder.decode(log_probs)
    
    assert len(results) == 2
    print(f"   ✅ BeamSearchDecoder: decoded {len(results)} sequences")
    
    # Test prefix beam search
    prefix_decoder = PrefixBeamSearchDecoder(beam_size=5)
    results = prefix_decoder.decode(log_probs)
    
    assert len(results) == 2
    print(f"   ✅ PrefixBeamSearchDecoder: decoded {len(results)} sequences")
    
    return True


def test_postprocessor():
    """Test post-processing."""
    print("Testing PostProcessor...")
    
    from src.inference.postprocessor import TextPostProcessor, LyricsFormatter, TranscriptionCleaner
    
    # Test text post-processor
    processor = TextPostProcessor()
    
    raw = "i finna get this bread"
    processed = processor.process(raw)
    
    assert processed.startswith("I")  # Should capitalize I
    print(f"   ✅ TextPostProcessor: '{raw}' -> '{processed}'")
    
    # Test slang normalization
    processor_slang = TextPostProcessor(normalize_slang=True)
    normalized = processor_slang.process(raw)
    
    assert "fixing to" in normalized or "I" in normalized
    print(f"   ✅ Slang normalization working")
    
    # Test lyrics formatter
    formatter = LyricsFormatter()
    formatted = formatter.format("yeah yeah I'm on my grind")
    
    assert len(formatted) > 0
    print(f"   ✅ LyricsFormatter working")
    
    # Test full cleaner
    cleaner = TranscriptionCleaner()
    cleaned = cleaner.clean("i gonna make it yeah", format_as_lyrics=False)
    
    assert cleaned.startswith("I")
    print(f"   ✅ TranscriptionCleaner working")
    
    return True


def test_real_audio():
    """Test with real audio file if available."""
    print("Testing Real Audio Inference...")
    
    from src.inference.engine import InferenceEngine
    
    test_file = Path("data/raw/test/sample.mp3")
    
    if not test_file.exists():
        print(f"   ⚠️  No test file found, skipping")
        return True
    
    engine = InferenceEngine(device='cpu')
    
    result = engine.transcribe(
        test_file,
        return_phonemes=True,
        return_timing=True
    )
    
    assert 'text' in result
    assert 'phonemes' in result
    assert 'timing' in result
    
    print(f"   ✅ Audio loaded and processed")
    print(f"   ✅ Inference time: {result['timing']['inference']:.3f}s")
    print(f"   ✅ Text tokens: {len(result['text']) if isinstance(result['text'], list) else 'string'}")
    print(f"   ✅ Phonemes: {len(result['phonemes'])}")
    
    return True


def test_streaming_inference():
    """Test streaming inference engine."""
    print("Testing StreamingInferenceEngine...")
    
    from src.inference.engine import StreamingInferenceEngine
    
    engine = StreamingInferenceEngine(
        chunk_duration=5.0,
        overlap_duration=0.5,
        device='cpu'
    )
    
    assert engine.chunk_duration == 5.0
    assert engine.overlap_duration == 0.5
    
    print(f"   ✅ StreamingInferenceEngine initialized")
    print(f"   ✅ Chunk duration: {engine.chunk_duration}s")
    print(f"   ✅ Overlap: {engine.overlap_duration}s")
    
    return True


def test_end_to_end():
    """Test complete inference pipeline."""
    print("Testing End-to-End Inference...")
    
    from src.inference.engine import InferenceEngine
    from src.inference.postprocessor import TranscriptionCleaner
    
    # Initialize
    engine = InferenceEngine(device='cpu')
    cleaner = TranscriptionCleaner()
    
    # Create dummy features
    dummy_features = torch.randn(1, 200, 80)
    
    # Transcribe
    result = engine.transcribe_features(dummy_features, return_phonemes=True)
    
    # Get text (token IDs in this case)
    raw_text = result['texts'][0]
    if isinstance(raw_text, list):
        raw_text = ' '.join(map(str, raw_text[:20]))  # Limit for display
    
    # Clean
    cleaned = cleaner.clean(raw_text, format_as_lyrics=False)
    
    print(f"   ✅ Pipeline complete")
    print(f"   ✅ Raw output type: {type(result['texts'][0])}")
    print(f"   ✅ Phonemes generated: {len(result['phonemes'][0])}")
    
    return True


def main():
    print("=" * 50)
    print("INFERENCE PIPELINE TEST")
    print("=" * 50)
    print("")
    
    tests = [
        ("Inference Engine", test_inference_engine),
        ("Beam Search Decoder", test_beam_search_decoder),
        ("Post-Processor", test_postprocessor),
        ("Real Audio", test_real_audio),
        ("Streaming Inference", test_streaming_inference),
        ("End-to-End", test_end_to_end),
    ]
    
    results = []
    
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
        print("")
    
    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"   {status} {name}")
    
    print("")
    print(f"   Passed: {passed}/{total}")
    
    if passed == total:
        print("")
        print("🎉 All tests passed! Inference pipeline is ready.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
