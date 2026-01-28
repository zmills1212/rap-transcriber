"""
Main training script for rap transcription model.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --epochs 10 --batch_size 8
"""
import sys
sys.path.insert(0, '.')

import argparse
import torch
from pathlib import Path

from src.utils.config import load_config
from src.models.rap_transcriber import create_model, RapTranscriber
from src.training.trainer import Trainer
from src.training.optimizer import create_optimizer, create_scheduler


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train rap transcription model')
    
    # Config
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    
    # Training params (override config)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=None,
                        help='Warmup steps')
    
    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='outputs/logs',
                        help='Log directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu, cuda, mps)')
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (small model, few steps)')
    
    return parser.parse_args()


def get_device(requested_device: str = None) -> str:
    """Determine best available device."""
    if requested_device:
        return requested_device
    
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def create_dummy_dataloader(batch_size: int, num_batches: int = 10):
    """
    Create dummy data loader for testing.
    Replace with real data loader when you have training data.
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    # Dummy data dimensions
    time_frames = 500
    n_mels = 80
    max_text_len = 100
    max_phoneme_len = 200
    
    # Create dummy tensors
    features = torch.randn(num_batches * batch_size, time_frames, n_mels)
    text_targets = torch.randint(0, 1000, (num_batches * batch_size, max_text_len))
    text_lengths = torch.randint(10, max_text_len, (num_batches * batch_size,))
    phoneme_targets = torch.randint(0, 84, (num_batches * batch_size, max_phoneme_len))
    phoneme_lengths = torch.randint(10, max_phoneme_len, (num_batches * batch_size,))
    input_lengths = torch.full((num_batches * batch_size,), time_frames // 4)
    
    dataset = TensorDataset(
        features, text_targets, text_lengths,
        phoneme_targets, phoneme_lengths, input_lengths
    )
    
    def collate_fn(batch):
        features, text_targets, text_lengths, phoneme_targets, phoneme_lengths, input_lengths = zip(*batch)
        return {
            'features': torch.stack(features),
            'text_targets': torch.stack(text_targets),
            'text_lengths': torch.stack(text_lengths),
            'phoneme_targets': torch.stack(phoneme_targets),
            'phoneme_lengths': torch.stack(phoneme_lengths),
            'input_lengths': torch.stack(input_lengths),
        }
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Override config with command line args
    epochs = args.epochs or config['training'].get('max_epochs', 10)
    batch_size = args.batch_size or config['training'].get('batch_size', 8)
    learning_rate = args.learning_rate or config['training'].get('learning_rate', 1e-4)
    warmup_steps = args.warmup_steps or config['training'].get('warmup_steps', 5000)
    
    # Debug mode: smaller model and fewer steps
    if args.debug:
        print("🐛 DEBUG MODE: Using smaller model")
        epochs = 2
        batch_size = 2
        model_config = {
            'encoder_dim': 128,
            'encoder_layers': 2,
            'encoder_heads': 4,
        }
    else:
        model_config = {
            'encoder_dim': config['model']['encoder'].get('dim', 512),
            'encoder_layers': config['model']['encoder'].get('layers', 12),
            'encoder_heads': config['model']['encoder'].get('heads', 8),
        }
    
    # Device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = create_model(model_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    print("Creating optimizer...")
    optimizer = create_optimizer(
        model,
        optimizer_type='adamw',
        learning_rate=learning_rate,
        weight_decay=0.01
    )
    
    # Create scheduler
    total_steps = epochs * 100  # Approximate, adjust based on dataset
    scheduler = create_scheduler(
        optimizer,
        scheduler_type='cosine_warmup',
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        grad_clip=config['training'].get('grad_clip', 1.0)
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Create data loaders
    # TODO: Replace with real data loaders when training data is available
    print("Creating data loaders (dummy data for now)...")
    train_loader = create_dummy_dataloader(batch_size, num_batches=10)
    val_loader = create_dummy_dataloader(batch_size, num_batches=3)
    
    # Train
    print("")
    print("=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")
    print("=" * 50)
    print("")
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        save_every=1,
        early_stopping_patience=5
    )
    
    print("")
    print("=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Best loss: {trainer.best_loss:.4f}")
    print(f"  Checkpoints saved to: {args.checkpoint_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
