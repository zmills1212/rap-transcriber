"""
Optimizer and learning rate scheduler for rap transcription model.
"""
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import (
    LambdaLR, 
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR
)
import math
from typing import Optional, Dict, Any


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adamw',
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.98),
    eps: float = 1e-9
) -> torch.optim.Optimizer:
    """
    Create optimizer for model.
    
    Args:
        model: The model to optimize
        optimizer_type: 'adam' or 'adamw'
        learning_rate: Initial learning rate
        weight_decay: Weight decay (L2 regularization)
        betas: Adam beta parameters
        eps: Adam epsilon
        
    Returns:
        Configured optimizer
    """
    # Separate parameters for weight decay
    # Don't apply weight decay to bias and layer norm
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters() 
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            'weight_decay': weight_decay
        },
        {
            'params': [
                p for n, p in model.named_parameters() 
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            'weight_decay': 0.0
        }
    ]
    
    if optimizer_type.lower() == 'adamw':
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps
        )
    elif optimizer_type.lower() == 'adam':
        optimizer = Adam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def get_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int
) -> LambdaLR:
    """
    Linear warmup followed by linear decay.
    
    Args:
        optimizer: The optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Linear decay
            return max(
                0.0,
                float(total_steps - current_step) / 
                float(max(1, total_steps - warmup_steps))
            )
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    scheduler.step_per_batch = True  # Flag for trainer
    return scheduler


def get_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1
) -> LambdaLR:
    """
    Linear warmup followed by cosine decay.
    
    Args:
        optimizer: The optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr_ratio: Minimum LR as ratio of initial LR
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    scheduler.step_per_batch = True
    return scheduler


def get_transformer_scheduler(
    optimizer: torch.optim.Optimizer,
    d_model: int = 512,
    warmup_steps: int = 4000
) -> LambdaLR:
    """
    Transformer-style learning rate schedule.
    From "Attention Is All You Need" paper.
    
    LR = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    
    Args:
        optimizer: The optimizer
        d_model: Model dimension
        warmup_steps: Number of warmup steps
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int) -> float:
        current_step = max(1, current_step)  # Avoid division by zero
        return (d_model ** -0.5) * min(
            current_step ** -0.5,
            current_step * (warmup_steps ** -1.5)
        )
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    scheduler.step_per_batch = True
    return scheduler


def get_one_cycle_scheduler(
    optimizer: torch.optim.Optimizer,
    max_lr: float,
    total_steps: int,
    pct_start: float = 0.3
) -> OneCycleLR:
    """
    One-cycle learning rate policy.
    Good for fast training.
    
    Args:
        optimizer: The optimizer
        max_lr: Maximum learning rate
        total_steps: Total number of training steps
        pct_start: Percentage of cycle spent increasing LR
        
    Returns:
        Learning rate scheduler
    """
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy='cos'
    )
    scheduler.step_per_batch = True
    return scheduler


def get_reduce_on_plateau_scheduler(
    optimizer: torch.optim.Optimizer,
    factor: float = 0.5,
    patience: int = 5,
    min_lr: float = 1e-7
) -> ReduceLROnPlateau:
    """
    Reduce LR when validation loss plateaus.
    
    Args:
        optimizer: The optimizer
        factor: Factor to reduce LR by
        patience: Number of epochs with no improvement
        min_lr: Minimum learning rate
        
    Returns:
        Learning rate scheduler
    """
    return ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine_warmup',
    warmup_steps: int = 5000,
    total_steps: int = 100000,
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Factory function to create learning rate scheduler.
    
    Args:
        optimizer: The optimizer
        scheduler_type: Type of scheduler
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        **kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'linear_warmup':
        return get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)
    
    elif scheduler_type == 'cosine_warmup':
        return get_cosine_warmup_scheduler(
            optimizer, warmup_steps, total_steps,
            min_lr_ratio=kwargs.get('min_lr_ratio', 0.1)
        )
    
    elif scheduler_type == 'transformer':
        return get_transformer_scheduler(
            optimizer,
            d_model=kwargs.get('d_model', 512),
            warmup_steps=warmup_steps
        )
    
    elif scheduler_type == 'one_cycle':
        return get_one_cycle_scheduler(
            optimizer,
            max_lr=kwargs.get('max_lr', 1e-3),
            total_steps=total_steps
        )
    
    elif scheduler_type == 'plateau':
        return get_reduce_on_plateau_scheduler(
            optimizer,
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 5)
        )
    
    elif scheduler_type == 'none' or scheduler_type is None:
        return None
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


if __name__ == "__main__":
    print("✅ Optimizer module loaded successfully!")
    print("")
    
    # Test with dummy model
    from src.models.rap_transcriber import create_model
    
    model = create_model()
    
    # Test optimizer creation
    print("Testing optimizer creation...")
    optimizer = create_optimizer(
        model,
        optimizer_type='adamw',
        learning_rate=1e-4,
        weight_decay=0.01
    )
    print(f"   ✅ AdamW optimizer created")
    print(f"   ✅ Learning rate: {optimizer.param_groups[0]['lr']}")
    print("")
    
    # Test different schedulers
    print("Testing scheduler creation...")
    
    schedulers_to_test = [
        ('linear_warmup', {}),
        ('cosine_warmup', {'min_lr_ratio': 0.1}),
        ('transformer', {'d_model': 512}),
        ('plateau', {'patience': 5}),
    ]
    
    for sched_type, kwargs in schedulers_to_test:
        scheduler = create_scheduler(
            optimizer,
            scheduler_type=sched_type,
            warmup_steps=1000,
            total_steps=10000,
            **kwargs
        )
        print(f"   ✅ {sched_type} scheduler created")
    
    print("")
    print("   All schedulers working!")
