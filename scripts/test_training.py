"""
Test script for the training pipeline.
Verifies all training components work together.
"""
import sys
sys.path.insert(0, '.')

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
from pathlib import Path
import tempfile
import shutil


def test_tokenizer():
    """Test tokenizer encoding/decoding."""
    print("Testing Tokenizer...")
    
    from src.data.tokenizer import RapTokenizer, PhonemeTokenizer
    
    # Text tokenizer
    text_tokenizer = RapTokenizer(vocab_size=1000)
    sample_texts = [
        "I'm finna get this bread",
        "She bussin it down yeah",
        "No cap this fire"
    ]
    text_tokenizer.build_vocab(sample_texts)
    
    test = "finna get bread"
    encoded = text_tokenizer.encode(test)
    decoded = text_tokenizer.decode(encoded)
    
    assert len(encoded) > 0, "Encoding failed"
    assert len(decoded) > 0, "Decoding failed"
    
    print(f"   ✅ Text tokenizer: '{test}' -> {encoded[:5]}... -> '{decoded}'")
    
    # Phoneme tokenizer
    phoneme_tokenizer = PhonemeTokenizer()
    phonemes = ['F', 'IH1', 'N', 'AH0']
    encoded_p = phoneme_tokenizer.encode(phonemes)
    decoded_p = phoneme_tokenizer.decode(encoded_p)
    
    assert len(encoded_p) > 0, "Phoneme encoding failed"
    assert decoded_p == phonemes, "Phoneme decoding mismatch"
    
    print(f"   ✅ Phoneme tokenizer: {phonemes} -> {encoded_p} -> {decoded_p}")
    
    return True


def test_optimizer_creation():
    """Test optimizer and scheduler creation."""
    print("Testing Optimizer & Scheduler...")
    
    from src.models.rap_transcriber import create_model
    from src.training.optimizer import create_optimizer, create_scheduler
    
    # Small model for testing
    model = create_model({
        'encoder_dim': 64,
        'encoder_layers': 1,
        'encoder_heads': 2
    })
    
    # Test optimizer
    optimizer = create_optimizer(model, learning_rate=1e-4)
    assert optimizer.param_groups[0]['lr'] == 1e-4
    print(f"   ✅ Optimizer created: lr={optimizer.param_groups[0]['lr']}")
    
    # Test schedulers
    for sched_type in ['cosine_warmup', 'linear_warmup', 'transformer']:
        scheduler = create_scheduler(
            optimizer,
            scheduler_type=sched_type,
            warmup_steps=100,
            total_steps=1000
        )
        assert scheduler is not None
        print(f"   ✅ {sched_type} scheduler created")
    
    return True


def test_single_train_step():
    """Test a single training step."""
    print("Testing Single Training Step...")
    
    from src.models.rap_transcriber import create_model
    from src.training.trainer import Trainer
    from src.training.optimizer import create_optimizer
    
    # Small model
    model = create_model({
        'encoder_dim': 64,
        'encoder_layers': 1,
        'encoder_heads': 2
    })
    
    optimizer = create_optimizer(model, learning_rate=1e-4)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device='cpu'  # Use CPU for testing
    )
    
    # Create dummy batch
    batch_size = 2
    time_frames = 100
    n_mels = 80
    
    batch = {
        'features': torch.randn(batch_size, time_frames, n_mels),
        'text_targets': torch.randint(0, 1000, (batch_size, 50)),
        'text_lengths': torch.tensor([50, 45]),
        'phoneme_targets': torch.randint(0, 84, (batch_size, 100)),
        'phoneme_lengths': torch.tensor([100, 90]),
        'input_lengths': torch.tensor([time_frames // 4, time_frames // 4]),
    }
    
    # Run training step
    losses = trainer.train_step(batch)
    
    assert 'total_loss' in losses
    assert losses['total_loss'] > 0
    assert trainer.global_step == 1
    
    print(f"   ✅ Training step completed")
    print(f"   ✅ Total loss: {losses['total_loss']:.4f}")
    print(f"   ✅ Global step: {trainer.global_step}")
    
    return True


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    print("Testing Checkpoint Save/Load...")
    
    from src.models.rap_transcriber import create_model
    from src.training.trainer import Trainer
    from src.training.optimizer import create_optimizer
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Small model
        model = create_model({
            'encoder_dim': 64,
            'encoder_layers': 1,
            'encoder_heads': 2
        })
        
        optimizer = create_optimizer(model, learning_rate=1e-4)
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device='cpu',
            checkpoint_dir=temp_dir
        )
        
        # Simulate some training
        trainer.global_step = 100
        trainer.current_epoch = 5
        trainer.best_loss = 1.5
        
        # Save checkpoint
        trainer.save_checkpoint('test.pt')
        
        checkpoint_path = Path(temp_dir) / 'test.pt'
        assert checkpoint_path.exists(), "Checkpoint not saved"
        print(f"   ✅ Checkpoint saved")
        
        # Create new trainer and load
        model2 = create_model({
            'encoder_dim': 64,
            'encoder_layers': 1,
            'encoder_heads': 2
        })
        optimizer2 = create_optimizer(model2, learning_rate=1e-4)
        
        trainer2 = Trainer(
            model=model2,
            optimizer=optimizer2,
            device='cpu',
            checkpoint_dir=temp_dir
        )
        
        trainer2.load_checkpoint('test.pt')
        
        assert trainer2.global_step == 100
        assert trainer2.current_epoch == 5
        assert trainer2.best_loss == 1.5
        
        print(f"   ✅ Checkpoint loaded")
        print(f"   ✅ State restored: step={trainer2.global_step}, epoch={trainer2.current_epoch}")
        
    finally:
        shutil.rmtree(temp_dir)
    
    return True


def test_end_to_end_training():
    """Test end-to-end training for a few steps."""
    print("Testing End-to-End Training (2 epochs)...")
    
    from torch.utils.data import DataLoader, TensorDataset
    from src.models.rap_transcriber import create_model
    from src.training.trainer import Trainer
    from src.training.optimizer import create_optimizer, create_scheduler
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Small model
        model = create_model({
            'encoder_dim': 64,
            'encoder_layers': 1,
            'encoder_heads': 2
        })
        
        optimizer = create_optimizer(model, learning_rate=1e-3)
        scheduler = create_scheduler(
            optimizer,
            scheduler_type='cosine_warmup',
            warmup_steps=5,
            total_steps=20
        )
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device='cpu',
            checkpoint_dir=temp_dir,
            log_dir=temp_dir
        )
        
        # Create tiny dataset
        batch_size = 2
        num_samples = 6
        
        features = torch.randn(num_samples, 100, 80)
        text_targets = torch.randint(0, 1000, (num_samples, 50))
        text_lengths = torch.full((num_samples,), 50)
        phoneme_targets = torch.randint(0, 84, (num_samples, 100))
        phoneme_lengths = torch.full((num_samples,), 100)
        input_lengths = torch.full((num_samples,), 25)
        
        dataset = TensorDataset(
            features, text_targets, text_lengths,
            phoneme_targets, phoneme_lengths, input_lengths
        )
        
        def collate_fn(batch):
            f, tt, tl, pt, pl, il = zip(*batch)
            return {
                'features': torch.stack(f),
                'text_targets': torch.stack(tt),
                'text_lengths': torch.stack(tl),
                'phoneme_targets': torch.stack(pt),
                'phoneme_lengths': torch.stack(pl),
                'input_lengths': torch.stack(il),
            }
        
        train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        
        # Train for 2 epochs
        history = trainer.train(
            train_loader=train_loader,
            num_epochs=2,
            save_every=1,
            early_stopping_patience=10
        )
        
        assert len(history['train_loss']) == 2
        assert trainer.current_epoch == 2
        
        # Check checkpoints exist
        assert (Path(temp_dir) / 'best.pt').exists()
        assert (Path(temp_dir) / 'final.pt').exists()
        
        print(f"   ✅ Training completed: 2 epochs")
        print(f"   ✅ Final loss: {history['train_loss'][-1]:.4f}")
        print(f"   ✅ Checkpoints saved")
        
    finally:
        shutil.rmtree(temp_dir)
    
    return True


def main():
    print("=" * 50)
    print("TRAINING PIPELINE TEST")
    print("=" * 50)
    print("")
    
    tests = [
        ("Tokenizer", test_tokenizer),
        ("Optimizer & Scheduler", test_optimizer_creation),
        ("Single Train Step", test_single_train_step),
        ("Checkpoint Save/Load", test_checkpoint_save_load),
        ("End-to-End Training", test_end_to_end_training),
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
        print("🎉 All tests passed! Training pipeline is ready.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
