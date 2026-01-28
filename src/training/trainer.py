"""
Training loop for rap transcription model.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Callable
from tqdm import tqdm
import time
import json


class Trainer:
    """
    Handles model training, validation, and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cpu',
        checkpoint_dir: str = 'outputs/checkpoints',
        log_dir: str = 'outputs/logs',
        grad_clip: float = 1.0,
        phoneme_weight: float = 0.3,
        text_weight: float = 0.7
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.grad_clip = grad_clip
        self.phoneme_weight = phoneme_weight
        self.text_weight = text_weight
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Dictionary with 'features', 'text_targets', 'phoneme_targets', etc.
            
        Returns:
            Dictionary with loss values
        """
        self.model.train()
        
        # Move batch to device
        features = batch['features'].to(self.device)
        
        # Forward pass
        outputs = self.model(features)
        
        # Compute losses (simplified - assumes targets are available)
        losses = {}
        total_loss = 0.0
        
        # Phoneme loss (if targets available)
        if 'phoneme_targets' in batch:
            phoneme_targets = batch['phoneme_targets'].to(self.device)
            input_lengths = batch['input_lengths'].to(self.device)
            phoneme_lengths = batch['phoneme_lengths'].to(self.device)
            
            phoneme_loss = self.model.phoneme_head.compute_loss(
                outputs['phoneme_logits'],
                phoneme_targets,
                input_lengths,
                phoneme_lengths
            )
            losses['phoneme_loss'] = phoneme_loss.item()
            total_loss += self.phoneme_weight * phoneme_loss
        
        # Text loss (if targets available)
        if 'text_targets' in batch:
            text_targets = batch['text_targets'].to(self.device)
            input_lengths = batch.get('input_lengths', 
                torch.full((features.size(0),), outputs['text_logits'].size(1), device=self.device))
            text_lengths = batch['text_lengths'].to(self.device)
            
            text_loss = self.model.text_head.compute_loss(
                outputs['text_logits'],
                text_targets,
                input_lengths,
                text_lengths
            )
            losses['text_loss'] = text_loss.item()
            total_loss += self.text_weight * text_loss
        
        # If no targets, use dummy loss for testing
        if total_loss == 0.0:
            total_loss = outputs['phoneme_logits'].mean() * 0.0  # Zero loss
        
        losses['total_loss'] = total_loss.item()
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        # Optimizer step
        self.optimizer.step()
        
        # Scheduler step (if per-step scheduler)
        if self.scheduler is not None and hasattr(self.scheduler, 'step_per_batch'):
            self.scheduler.step()
        
        self.global_step += 1
        
        return losses
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Run validation.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with average loss values
        """
        self.model.eval()
        
        total_losses = {}
        num_batches = 0
        
        for batch in val_loader:
            features = batch['features'].to(self.device)
            outputs = self.model(features)
            
            # Accumulate losses
            if 'phoneme_targets' in batch:
                phoneme_targets = batch['phoneme_targets'].to(self.device)
                input_lengths = batch['input_lengths'].to(self.device)
                phoneme_lengths = batch['phoneme_lengths'].to(self.device)
                
                phoneme_loss = self.model.phoneme_head.compute_loss(
                    outputs['phoneme_logits'],
                    phoneme_targets,
                    input_lengths,
                    phoneme_lengths
                )
                total_losses['phoneme_loss'] = total_losses.get('phoneme_loss', 0) + phoneme_loss.item()
            
            if 'text_targets' in batch:
                text_targets = batch['text_targets'].to(self.device)
                input_lengths = batch.get('input_lengths',
                    torch.full((features.size(0),), outputs['text_logits'].size(1), device=self.device))
                text_lengths = batch['text_lengths'].to(self.device)
                
                text_loss = self.model.text_head.compute_loss(
                    outputs['text_logits'],
                    text_targets,
                    input_lengths,
                    text_lengths
                )
                total_losses['text_loss'] = total_losses.get('text_loss', 0) + text_loss.item()
            
            num_batches += 1
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        # Compute total
        avg_losses['total_loss'] = (
            self.phoneme_weight * avg_losses.get('phoneme_loss', 0) +
            self.text_weight * avg_losses.get('text_loss', 0)
        )
        
        return avg_losses
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        log_interval: int = 100
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            log_interval: How often to log progress
            
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        epoch_losses = {}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Training step
            losses = self.train_step(batch)
            
            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total_loss']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log interval
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = epoch_losses['total_loss'] / num_batches
                print(f"  Step {self.global_step}: loss={avg_loss:.4f}")
        
        # Average epoch losses
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        
        # Validation
        if val_loader is not None:
            val_losses = self.validate(val_loader)
            avg_losses['val_loss'] = val_losses['total_loss']
            print(f"  Validation loss: {val_losses['total_loss']:.4f}")
        
        # Update scheduler (if per-epoch scheduler)
        if self.scheduler is not None and not hasattr(self.scheduler, 'step_per_batch'):
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_losses.get('val_loss', avg_losses['total_loss']))
            else:
                self.scheduler.step()
        
        # Update history
        self.history['train_loss'].append(avg_losses['total_loss'])
        if 'val_loss' in avg_losses:
            self.history['val_loss'].append(avg_losses['val_loss'])
        self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        
        self.current_epoch += 1
        
        return avg_losses
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        save_every: int = 1,
        early_stopping_patience: int = 5
    ) -> Dict[str, list]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Stop if no improvement for N epochs
            
        Returns:
            Training history
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("")
        
        no_improve_count = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train epoch
            metrics = self.train_epoch(train_loader, val_loader)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {self.current_epoch} completed in {epoch_time:.1f}s")
            print(f"  Train loss: {metrics['total_loss']:.4f}")
            if 'val_loss' in metrics:
                print(f"  Val loss:   {metrics['val_loss']:.4f}")
            print("")
            
            # Check for improvement
            current_loss = metrics.get('val_loss', metrics['total_loss'])
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                no_improve_count = 0
                self.save_checkpoint('best.pt')
                print(f"  New best model saved!")
            else:
                no_improve_count += 1
            
            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'epoch_{self.current_epoch}.pt')
            
            # Early stopping
            if no_improve_count >= early_stopping_patience:
                print(f"Early stopping after {early_stopping_patience} epochs without improvement")
                break
        
        # Save final model
        self.save_checkpoint('final.pt')
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'history': self.history,
            'config': self.model.config if hasattr(self.model, 'config') else None
        }
        
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint['history']
        
        print(f"  Checkpoint loaded: {path}")
        print(f"  Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    def save_history(self):
        """Save training history to JSON."""
        path = self.log_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


if __name__ == "__main__":
    print("✅ Trainer module loaded successfully!")
    print("")
    
    # Quick test with dummy model
    from src.models.rap_transcriber import create_model
    
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device='cpu'
    )
    
    print(f"   Trainer initialized")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Checkpoint dir: {trainer.checkpoint_dir}")
    print(f"   Log dir: {trainer.log_dir}")
