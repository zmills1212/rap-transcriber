"""
Test script for evaluation pipeline.
Verifies all evaluation components work together.
"""
import sys
sys.path.insert(0, '.')

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


def test_wer_cer_metrics():
    """Test WER and CER computation."""
    print("Testing WER/CER Metrics...")
    
    from src.utils.metrics import compute_wer, compute_cer, compute_metrics, compute_batch_metrics
    
    # Test perfect match
    wer, _ = compute_wer("hello world", "hello world")
    assert wer == 0.0, f"Perfect match should have WER=0, got {wer}"
    print("   ✅ Perfect match: WER=0%")
    
    # Test complete mismatch
    wer, _ = compute_wer("hello world", "goodbye moon")
    assert wer == 1.0, f"Complete mismatch should have WER=1, got {wer}"
    print("   ✅ Complete mismatch: WER=100%")
    
    # Test partial match
    wer, details = compute_wer("i am here", "i am there")
    assert 0 < wer < 1, f"Partial match should have 0<WER<1, got {wer}"
    print(f"   ✅ Partial match: WER={wer:.1%}")
    
    # Test CER
    cer, _ = compute_cer("hello", "hallo")
    assert cer == 0.2, f"One char diff in 5 should be CER=0.2, got {cer}"
    print(f"   ✅ CER computation: CER={cer:.1%}")
    
    # Test combined metrics
    metrics = compute_metrics("i'm finna get bread", "i'm gonna get bread")
    assert hasattr(metrics, 'wer')
    assert hasattr(metrics, 'cer')
    print(f"   ✅ Combined metrics: WER={metrics.wer:.1%}, CER={metrics.cer:.1%}")
    
    # Test batch metrics
    refs = ["hello world", "foo bar", "test case"]
    hyps = ["hello world", "foo baz", "test"]
    batch = compute_batch_metrics(refs, hyps)
    assert 'micro_wer' in batch
    assert 'macro_wer' in batch
    print(f"   ✅ Batch metrics: Micro WER={batch['micro_wer']:.1%}")
    
    return True


def test_slang_metrics():
    """Test slang-specific metrics."""
    print("Testing Slang Metrics...")
    
    from src.utils.slang_metrics import SlangAccuracyEvaluator, AdLibEvaluator
    
    # Test slang evaluator
    evaluator = SlangAccuracyEvaluator()
    
    # Test slang extraction
    slang = evaluator.extract_slang("i'm finna get this bread yeah")
    assert "finna" in slang
    assert "bread" in slang
    assert "yeah" in slang
    print(f"   ✅ Slang extraction: found {len(slang)} terms")
    
    # Test perfect slang match
    metrics = evaluator.compute_slang_accuracy(
        "finna get bread",
        "finna get bread"
    )
    assert metrics['slang_recall'] == 1.0
    print("   ✅ Perfect slang match: Recall=100%")
    
    # Test slang mismatch
    metrics = evaluator.compute_slang_accuracy(
        "she bussin it",
        "she busting it"
    )
    assert metrics['slang_recall'] < 1.0
    print(f"   ✅ Slang mismatch detected: Recall={metrics['slang_recall']:.0%}")
    
    # Test ad-lib evaluator
    adlib_eval = AdLibEvaluator()
    adlibs = adlib_eval.extract_adlibs("yeah yeah skrt skrt ayy")
    assert len(adlibs) > 0
    print(f"   ✅ Ad-lib extraction: found {len(adlibs)} ad-libs")
    
    # Test batch slang metrics
    refs = ["finna do it yeah", "no cap bruh", "bussin fr"]
    hyps = ["gonna do it yeah", "no cap bro", "busting for real"]
    batch = evaluator.compute_batch_slang_accuracy(refs, hyps)
    assert 'slang_recall' in batch
    assert 'most_missed_slang' in batch
    print(f"   ✅ Batch slang metrics: Recall={batch['slang_recall']:.1%}")
    
    return True


def test_metrics_edge_cases():
    """Test edge cases in metrics computation."""
    print("Testing Edge Cases...")
    
    from src.utils.metrics import compute_wer, compute_cer, compute_metrics
    
    # Empty reference
    wer, _ = compute_wer("", "hello")
    assert wer > 0
    print("   ✅ Empty reference handled")
    
    # Empty hypothesis
    wer, _ = compute_wer("hello", "")
    assert wer > 0
    print("   ✅ Empty hypothesis handled")
    
    # Both empty
    wer, _ = compute_wer("", "")
    assert wer == 0.0
    print("   ✅ Both empty handled")
    
    # Single word
    metrics = compute_metrics("word", "word")
    assert metrics.wer == 0.0
    print("   ✅ Single word handled")
    
    # Special characters
    metrics = compute_metrics("i'm here!", "i'm here!")
    assert metrics.wer == 0.0
    print("   ✅ Special characters handled")
    
    # Case insensitivity
    wer, _ = compute_wer("HELLO WORLD", "hello world")
    assert wer == 0.0
    print("   ✅ Case insensitivity working")
    
    return True


def test_metrics_tracker():
    """Test metrics tracker for running evaluation."""
    print("Testing MetricsTracker...")
    
    from src.utils.metrics import MetricsTracker
    
    tracker = MetricsTracker()
    
    # Add samples
    tracker.update("hello world", "hello world")
    tracker.update("foo bar", "foo baz")
    tracker.update("test case", "test")
    
    metrics = tracker.get_current_metrics()
    
    assert metrics['samples'] == 3
    assert 'wer' in metrics
    assert 'cer' in metrics
    print(f"   ✅ Tracked {metrics['samples']} samples")
    print(f"   ✅ Running WER: {metrics['wer']:.1%}")
    
    # Test reset
    tracker.reset()
    metrics = tracker.get_current_metrics()
    assert metrics['samples'] == 0
    print("   ✅ Reset working")
    
    return True


def test_integration():
    """Test full evaluation integration."""
    print("Testing Full Integration...")
    
    from src.utils.metrics import compute_batch_metrics
    from src.utils.slang_metrics import SlangAccuracyEvaluator
    
    # Simulated transcription results
    references = [
        "i'm finna get this bread yeah",
        "she bussin it down skrt",
        "no cap this fire bruh"
    ]
    
    hypotheses = [
        "i'm gonna get this bread yeah",
        "she busting it down skirt",
        "no cap this fire bro"
    ]
    
    # Standard metrics
    standard = compute_batch_metrics(references, hypotheses)
    
    # Slang metrics
    slang_eval = SlangAccuracyEvaluator()
    slang = slang_eval.compute_batch_slang_accuracy(references, hypotheses)
    
    # Combined report
    print(f"   Standard WER: {standard['micro_wer']:.1%}")
    print(f"   Slang Recall: {slang['slang_recall']:.1%}")
    print(f"   Slang F1: {slang['slang_f1']:.1%}")
    
    assert standard['micro_wer'] >= 0
    assert slang['slang_recall'] >= 0
    
    print("   ✅ Integration complete")
    
    return True


def main():
    print("=" * 50)
    print("EVALUATION PIPELINE TEST")
    print("=" * 50)
    print("")
    
    tests = [
        ("WER/CER Metrics", test_wer_cer_metrics),
        ("Slang Metrics", test_slang_metrics),
        ("Edge Cases", test_metrics_edge_cases),
        ("Metrics Tracker", test_metrics_tracker),
        ("Full Integration", test_integration),
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
        print("🎉 All tests passed! Evaluation pipeline is ready.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
