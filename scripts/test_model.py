"""
Test script for the full model architecture.
Verifies end-to-end forward pass from audio to predictions.
"""
import sys
sys.path.insert(0, '.')

import torch
from pathlib import Path


def test_encoder():
    """Test Conformer encoder."""
    print("Testing Conformer Encoder...")
    
    from src.models.encoder import ConformerEncoder
    
    encoder = ConformerEncoder(
        input_dim=80,
        d_model=512,
        num_layers=12,
        num_heads=8
    )
    
    # Dummy input
    x = torch.randn(2, 500, 80)
    
    output, mask = encoder(x)
    
    assert output.shape == (2, 125, 512), f"Unexpected shape: {output.shape}"
    
    params = sum(p.numel() for p in encoder.parameters())
    
    print(f"   ✅ Input:  {x.shape}")
    print(f"   ✅ Output: {output.shape}")
    print(f"   ✅ Params: {params:,}")
    return True


def test_phoneme_head():
    """Test phoneme prediction head."""
    print("Testing Phoneme Head...")
    
    from src.models.phoneme_head import PhonemeHead
    
    head = PhonemeHead(encoder_dim=512, vocab_size=84)
    
    # Dummy encoder output
    encoder_out = torch.randn(2, 125, 512)
    
    logits = head(encoder_out)
    decoded = head.decode_greedy(logits)
    
    assert logits.shape == (2, 125, 84), f"Unexpected shape: {logits.shape}"
    assert len(decoded) == 2, "Should decode 2 sequences"
    
    print(f"   ✅ Input:   {encoder_out.shape}")
    print(f"   ✅ Logits:  {logits.shape}")
    print(f"   ✅ Decoded: {len(decoded[0])} phonemes")
    return True


def test_text_head():
    """Test text prediction head."""
    print("Testing Text Head...")
    
    from src.models.text_head import TextHead
    
    head = TextHead(encoder_dim=512, vocab_size=8192)
    
    # Dummy encoder output
    encoder_out = torch.randn(2, 125, 512)
    
    logits = head(encoder_out)
    decoded = head.decode_greedy(logits)
    
    assert logits.shape == (2, 125, 8192), f"Unexpected shape: {logits.shape}"
    assert len(decoded) == 2, "Should decode 2 sequences"
    
    print(f"   ✅ Input:   {encoder_out.shape}")
    print(f"   ✅ Logits:  {logits.shape}")
    print(f"   ✅ Decoded: {len(decoded[0])} tokens")
    return True


def test_full_model():
    """Test complete RapTranscriber model."""
    print("Testing Full RapTranscriber Model...")
    
    from src.models.rap_transcriber import RapTranscriber, create_model
    
    model = create_model()
    
    # Dummy mel spectrogram
    features = torch.randn(2, 500, 80)
    
    # Forward pass
    outputs = model(features)
    
    assert 'encoder_out' in outputs
    assert 'phoneme_logits' in outputs
    assert 'text_logits' in outputs
    
    # Test decode
    decoded = model.decode(features)
    
    assert 'phonemes' in decoded
    assert 'text_tokens' in decoded
    
    params = model.parameter_breakdown()
    
    print(f"   ✅ Input:          {features.shape}")
    print(f"   ✅ Encoder out:    {outputs['encoder_out'].shape}")
    print(f"   ✅ Phoneme logits: {outputs['phoneme_logits'].shape}")
    print(f"   ✅ Text logits:    {outputs['text_logits'].shape}")
    print(f"   ✅ Total params:   {params['total']:,}")
    return True


def test_end_to_end():
    """Test complete pipeline: Audio -> Features -> Model -> Predictions."""
    print("Testing End-to-End Pipeline...")
    
    from src.data.audio_processor import AudioProcessor
    from src.data.feature_extractor import FeatureExtractor
    from src.models.rap_transcriber import create_model
    
    # Initialize components
    processor = AudioProcessor()
    extractor = FeatureExtractor()
    model = create_model()
    
    # Check for test audio file
    test_file = Path("data/raw/test/sample.mp3")
    
    if test_file.exists():
        # Use real audio
        waveform = processor.process(test_file)
        print(f"   ✅ Loaded audio: {test_file.name}")
    else:
        # Use dummy audio
        waveform = torch.randn(1, 16000 * 5)  # 5 seconds
        print(f"   ⚠️  Using dummy audio (no test file)")
    
    # Extract features
    features = extractor.extract(waveform)
    print(f"   ✅ Extracted features: {features.shape}")
    
    # Add batch dimension
    features = features.unsqueeze(0)  # [1, n_mels, time]
    
    # Run model
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        decoded = model.decode(features)
    
    print(f"   ✅ Model output shapes:")
    print(f"      Encoder:  {outputs['encoder_out'].shape}")
    print(f"      Phonemes: {outputs['phoneme_logits'].shape}")
    print(f"      Text:     {outputs['text_logits'].shape}")
    print(f"   ✅ Decoded {len(decoded['phonemes'][0])} phonemes")
    print(f"   ✅ Decoded {len(decoded['text_tokens'][0])} text tokens")
    
    return True


def test_model_save_load():
    """Test model save and load."""
    print("Testing Model Save/Load...")
    
    from src.models.rap_transcriber import create_model
    import tempfile
    import os
    
    # Create model
    model = create_model()
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    
    try:
        model.save(temp_path)
        print(f"   ✅ Model saved to temp file")
        
        # Load model
        from src.models.rap_transcriber import RapTranscriber
        loaded_model = RapTranscriber.load(temp_path)
        print(f"   ✅ Model loaded from temp file")
        
        # Verify same config
        assert model.config == loaded_model.config
        print(f"   ✅ Config matches")
        
        # Verify same output
        x = torch.randn(1, 100, 80)
        model.eval()
        loaded_model.eval()
        
        with torch.no_grad():
            out1 = model(x)['encoder_out']
            out2 = loaded_model(x)['encoder_out']
        
        assert torch.allclose(out1, out2, atol=1e-5)
        print(f"   ✅ Outputs match")
        
    finally:
        os.unlink(temp_path)
    
    return True


def main():
    print("=" * 50)
    print("MODEL ARCHITECTURE TEST")
    print("=" * 50)
    print("")
    
    tests = [
        ("Conformer Encoder", test_encoder),
        ("Phoneme Head", test_phoneme_head),
        ("Text Head", test_text_head),
        ("Full Model", test_full_model),
        ("End-to-End Pipeline", test_end_to_end),
        ("Model Save/Load", test_model_save_load),
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
        print("🎉 All tests passed! Model architecture is ready.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
