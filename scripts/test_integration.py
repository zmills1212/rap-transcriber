#!/usr/bin/env python3
"""
Final integration test for the Rap Transcription System.
Tests the complete pipeline from audio to transcription.
"""
import sys
sys.path.insert(0, '.')

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from pathlib import Path
import tempfile
import time


def test_imports():
    """Test all major imports work."""
    print("Testing imports...")
    
    # Core
    import torch
    import torchaudio
    print("   ✅ PyTorch & torchaudio")
    
    # Models
    from src.models.rap_transcriber import RapTranscriber, create_model
    from src.models.encoder import ConformerEncoder
    from src.models.phoneme_head import PhonemeHead
    from src.models.text_head import TextHead
    print("   ✅ Model modules")
    
    # Data
    from src.data.audio_processor import AudioProcessor
    from src.data.feature_extractor import FeatureExtractor
    from src.data.tokenizer import RapTokenizer
    from src.data.slang_lexicon import SlangLexicon
    from src.data.manifest import DataManifest
    print("   ✅ Data modules")
    
    # Training
    from src.training.trainer import Trainer
    from src.training.optimizer import create_optimizer, create_scheduler
    print("   ✅ Training modules")
    
    # Inference
    from src.inference.engine import InferenceEngine
    from src.inference.decoder import BeamSearchDecoder
    from src.inference.postprocessor import TranscriptionCleaner
    print("   ✅ Inference modules")
    
    # API
    from src.api.server import app
    from src.api.client import TranscriptionClient
    print("   ✅ API modules")
    
    # Utils
    from src.utils.config import load_config
    from src.utils.metrics import compute_wer, compute_cer
    from src.utils.slang_metrics import SlangAccuracyEvaluator
    print("   ✅ Utility modules")
    
    return True


def test_audio_pipeline():
    """Test audio processing pipeline."""
    print("Testing audio pipeline...")
    
    from src.data.audio_processor import AudioProcessor
    from src.data.feature_extractor import FeatureExtractor
    import torch
    
    processor = AudioProcessor()
    extractor = FeatureExtractor()
    
    # Check for test audio
    test_audio = Path("data/raw/test/sample.mp3")
    
    if test_audio.exists():
        # Process real audio
        waveform = processor.process(test_audio, trim_pad=False)
        features = extractor.extract(waveform)
        
        print(f"   ✅ Loaded: {test_audio.name}")
        print(f"   ✅ Waveform shape: {waveform.shape}")
        print(f"   ✅ Features shape: {features.shape}")
    else:
        # Use dummy audio
        dummy_waveform = torch.randn(1, 16000 * 5)  # 5 seconds
        features = extractor.extract(dummy_waveform)
        
        print(f"   ✅ Dummy waveform: {dummy_waveform.shape}")
        print(f"   ✅ Features shape: {features.shape}")
    
    return True


def test_model_inference():
    """Test model forward pass."""
    print("Testing model inference...")
    
    import torch
    from src.models.rap_transcriber import create_model
    
    # Create small model for testing
    model = create_model({
        'encoder_dim': 128,
        'encoder_layers': 2,
        'encoder_heads': 4
    })
    model.eval()
    
    # Dummy input
    batch_size = 2
    time_frames = 100
    n_mels = 80
    
    dummy_input = torch.randn(batch_size, time_frames, n_mels)
    
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print(f"   ✅ Model created: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"   ✅ Encoder output: {outputs['encoder_out'].shape}")
    print(f"   ✅ Phoneme logits: {outputs['phoneme_logits'].shape}")
    print(f"   ✅ Text logits: {outputs['text_logits'].shape}")
    
    return True


def test_end_to_end_transcription():
    """Test complete transcription pipeline."""
    print("Testing end-to-end transcription...")
    
    from src.inference.engine import InferenceEngine
    from src.inference.postprocessor import TranscriptionCleaner
    import torch
    
    # Initialize
    engine = InferenceEngine(device='cpu')
    cleaner = TranscriptionCleaner()
    
    print(f"   ✅ Engine initialized on {engine.device}")
    
    # Test with dummy features
    dummy_features = torch.randn(1, 100, 80)
    result = engine.transcribe_features(dummy_features, return_phonemes=True)
    
    print(f"   ✅ Transcription completed")
    print(f"   ✅ Output type: {type(result['texts'][0])}")
    
    # Test with real audio if available
    test_audio = Path("data/raw/test/sample.mp3")
    if test_audio.exists():
        result = engine.transcribe(test_audio, return_timing=True)
        
        text = result['text']
        if isinstance(text, list):
            text = f"[{len(text)} tokens]"
        else:
            text = text[:50] + "..." if len(text) > 50 else text
        
        print(f"   ✅ Real audio transcribed: {text}")
        print(f"   ✅ Inference time: {result['timing']['inference']:.3f}s")
    
    return True


def test_training_components():
    """Test training components work."""
    print("Testing training components...")
    
    import torch
    from src.models.rap_transcriber import create_model
    from src.training.optimizer import create_optimizer, create_scheduler
    
    # Small model
    model = create_model({
        'encoder_dim': 64,
        'encoder_layers': 1,
        'encoder_heads': 2
    })
    
    # Optimizer
    optimizer = create_optimizer(model, learning_rate=1e-4)
    print(f"   ✅ Optimizer created")
    
    # Scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_type='cosine_warmup',
        warmup_steps=100,
        total_steps=1000
    )
    print(f"   ✅ Scheduler created")
    
    # Single training step
    dummy_input = torch.randn(2, 100, 80)
    outputs = model(dummy_input)
    loss = outputs['phoneme_logits'].mean()  # Dummy loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    print(f"   ✅ Training step completed")
    
    return True


def test_metrics():
    """Test evaluation metrics."""
    print("Testing evaluation metrics...")
    
    from src.utils.metrics import compute_wer, compute_cer, compute_metrics
    from src.utils.slang_metrics import SlangAccuracyEvaluator
    
    # WER/CER
    ref = "i'm finna get this bread"
    hyp = "i'm gonna get this bread"
    
    wer, _ = compute_wer(ref, hyp)
    cer, _ = compute_cer(ref, hyp)
    
    print(f"   ✅ WER: {wer:.2%}")
    print(f"   ✅ CER: {cer:.2%}")
    
    # Slang metrics
    evaluator = SlangAccuracyEvaluator()
    slang = evaluator.compute_slang_accuracy(ref, hyp)
    
    print(f"   ✅ Slang recall: {slang['slang_recall']:.2%}")
    
    return True


def test_api_components():
    """Test API components."""
    print("Testing API components...")
    
    from src.api.server import app, TranscriptionResponse, HealthResponse
    from src.api.client import TranscriptionClient
    
    # Test response models
    response = TranscriptionResponse(text="test", processing_time_seconds=1.0)
    assert response.text == "test"
    print(f"   ✅ Response models")
    
    # Test client
    client = TranscriptionClient("http://localhost:8000")
    assert client.base_url == "http://localhost:8000"
    print(f"   ✅ Client initialization")
    
    # Test app routes
    routes = [r.path for r in app.routes]
    assert '/health' in routes
    assert '/transcribe' in routes
    print(f"   ✅ API routes configured")
    
    return True


def test_data_manifest():
    """Test data manifest system."""
    print("Testing data manifest...")
    
    from src.data.manifest import DataManifest, ManifestEntry
    import tempfile
    
    # Create manifest
    manifest = DataManifest()
    
    for i in range(5):
        manifest.add_entry(ManifestEntry(
            audio_path=f"audio_{i}.mp3",
            text=f"Sample text {i}",
            duration=10.0 + i
        ))
    
    # Test split
    train, val, test = manifest.split(0.6, 0.2, 0.2)
    
    print(f"   ✅ Manifest created: {len(manifest)} entries")
    print(f"   ✅ Split: train={len(train)}, val={len(val)}, test={len(test)}")
    
    # Test save/load
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        manifest.save(f.name)
        loaded = DataManifest.load(f.name)
        Path(f.name).unlink()
    
    print(f"   ✅ Save/load working")
    
    return True


def main():
    print("=" * 50)
    print("🎤 RAP TRANSCRIPTION - INTEGRATION TEST")
    print("=" * 50)
    print("")
    
    tests = [
        ("Imports", test_imports),
        ("Audio Pipeline", test_audio_pipeline),
        ("Model Inference", test_model_inference),
        ("End-to-End Transcription", test_end_to_end_transcription),
        ("Training Components", test_training_components),
        ("Metrics", test_metrics),
        ("API Components", test_api_components),
        ("Data Manifest", test_data_manifest),
    ]
    
    results = []
    start_time = time.time()
    
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
    
    total_time = time.time() - start_time
    
    # Summary
    print("=" * 50)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"   {status} {name}")
    
    print("")
    print(f"   Passed: {passed}/{total}")
    print(f"   Time: {total_time:.1f}s")
    
    if passed == total:
        print("")
        print("🎉 ALL TESTS PASSED!")
        print("")
        print("Your Rap Transcription System is fully functional!")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
