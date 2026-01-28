"""
Test script for the data pipeline.
Verifies all components work together.
"""
import sys
sys.path.insert(0, '.')

import torch
from pathlib import Path

def test_audio_processor():
    """Test audio loading and preprocessing."""
    print("Testing AudioProcessor...")
    
    from src.data.audio_processor import AudioProcessor
    
    processor = AudioProcessor()
    test_file = Path("data/raw/test/sample.mp3")
    
    if not test_file.exists():
        print("   ⚠️  No test file found, skipping audio test")
        return False
    
    waveform = processor.process(test_file)
    
    assert waveform.dim() == 2, "Waveform should be 2D"
    assert waveform.shape[0] == 1, "Should be mono"
    
    print(f"   ✅ Audio loaded: {waveform.shape}")
    print(f"   ✅ Duration: {waveform.shape[1] / 16000:.2f} sec")
    return True


def test_feature_extractor():
    """Test mel spectrogram extraction."""
    print("Testing FeatureExtractor...")
    
    from src.data.feature_extractor import FeatureExtractor
    
    extractor = FeatureExtractor()
    
    # Create dummy audio (5 seconds)
    dummy_audio = torch.randn(1, 16000 * 5)
    
    features = extractor.extract(dummy_audio)
    
    assert features.dim() == 2, "Features should be 2D"
    assert features.shape[0] == 80, "Should have 80 mel bins"
    
    print(f"   ✅ Features extracted: {features.shape}")
    return True


def test_full_pipeline():
    """Test audio -> features pipeline."""
    print("Testing Full Pipeline...")
    
    from src.data.audio_processor import AudioProcessor
    from src.data.feature_extractor import FeatureExtractor
    
    processor = AudioProcessor()
    extractor = FeatureExtractor()
    
    test_file = Path("data/raw/test/sample.mp3")
    
    if not test_file.exists():
        print("   ⚠️  No test file found, using dummy audio")
        waveform = torch.randn(1, 16000 * 5)
    else:
        waveform = processor.process(test_file)
    
    features = extractor.extract(waveform)
    
    print(f"   ✅ Pipeline complete!")
    print(f"   ✅ Input:  {waveform.shape} (waveform)")
    print(f"   ✅ Output: {features.shape} (mel spectrogram)")
    return True


def test_slang_lexicon():
    """Test slang lexicon functionality."""
    print("Testing SlangLexicon...")
    
    from src.data.slang_lexicon import SlangLexicon
    
    lexicon = SlangLexicon()
    
    # Test lookup
    assert "finna" in lexicon, "Should contain 'finna'"
    assert lexicon.is_slang("bussin"), "Should recognize 'bussin'"
    
    # Test pronunciations
    prons = lexicon.get_pronunciations("finna")
    assert len(prons) > 0, "Should have pronunciations"
    
    # Test fuzzy lookup
    matches = lexicon.fuzzy_lookup("bussing", max_distance=2)
    
    print(f"   ✅ Lexicon loaded: {len(lexicon)} words")
    print(f"   ✅ Lookup working")
    print(f"   ✅ Fuzzy match for 'bussing': {matches[:2]}")
    return True


def test_config():
    """Test configuration loading."""
    print("Testing Config...")
    
    from src.utils.config import load_config
    
    config = load_config()
    
    assert "project" in config, "Should have project section"
    assert "model" in config, "Should have model section"
    
    print(f"   ✅ Config loaded: {config['project']['name']}")
    return True


def main():
    print("=" * 50)
    print("DATA PIPELINE TEST")
    print("=" * 50)
    print("")
    
    tests = [
        ("Config", test_config),
        ("Audio Processor", test_audio_processor),
        ("Feature Extractor", test_feature_extractor),
        ("Full Pipeline", test_full_pipeline),
        ("Slang Lexicon", test_slang_lexicon),
    ]
    
    results = []
    
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"   ❌ Error: {e}")
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
        print("🎉 All tests passed! Data pipeline is ready.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
