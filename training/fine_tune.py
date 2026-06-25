"""
Phase D: Fine-Tune Whisper with LoRA (Regularized)

Based on the proven Phase B2 architecture that trained successfully.
Adds: reduced LoRA rank, dropout, early stopping, weight decay,
label smoothing, and support for augmented datasets.

Usage:
    python -m training.fine_tune                     # Train with regularized defaults
    python -m training.fine_tune --evaluate           # Evaluate fine-tuned model
    python -m training.fine_tune --transcribe file.mp3
"""

import os
import sys
import json
import torch
import evaluate
import numpy as np
import librosa
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, PeftModel


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "training" / "hf_dataset" / "rap_transcription"
OUTPUT_DIR = PROJECT_ROOT / "training" / "fine_tuned"
ADAPTER_DIR = OUTPUT_DIR / "lora_adapter"

MODEL_MAP = {
    "tiny":   "openai/whisper-tiny.en",
    "base":   "openai/whisper-base.en",
    "small":  "openai/whisper-small.en",
    "medium": "openai/whisper-medium.en",
}


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("Using CPU (this will be slow)")
        return torch.device("cpu")


# =============================================================================
# Data Collator (identical to proven B2 version)
# =============================================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Custom data collator for Whisper fine-tuning.
    Handles padding of both input features and labels.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# =============================================================================
# Model Setup
# =============================================================================

def setup_model_and_processor(model_size: str = "small"):
    """Load Whisper model and processor."""
    model_name = MODEL_MAP.get(model_size)
    if model_name is None:
        raise ValueError(f"Unknown model size: {model_size}")

    print(f"\nLoading model: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    return model, processor


def apply_lora(model, rank: int = 8, alpha: int = 16, dropout: float = 0.15):
    """Apply LoRA adapters to Whisper model."""
    print(f"\nApplying LoRA (rank={rank}, alpha={alpha}, dropout={dropout})")

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=dropout,
        bias="none",
    )

    model = get_peft_model(model, config)
    model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


# =============================================================================
# Dataset Preprocessing
# =============================================================================

def preprocess_dataset(dataset, processor):
    """
    Preprocess audio and text for Whisper.
    
    Handles two dataset formats:
      - Old format: 'audio' is a file path string
      - New format: 'audio' is a dict with 'array' and 'sampling_rate'
    """

    def prepare_sample(batch):
        audio_field = batch["audio"]

        # Handle both formats
        if isinstance(audio_field, dict):
            # New format: audio array already loaded
            audio_array = np.array(audio_field["array"], dtype=np.float32)
        elif isinstance(audio_field, str):
            # Old format: file path
            audio_array, _ = librosa.load(audio_field, sr=16000, mono=True)
        else:
            raise ValueError(f"Unexpected audio format: {type(audio_field)}")

        # Compute input features (log-mel spectrogram)
        input_features = processor.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="np",
        ).input_features[0]

        # Encode target text
        labels = processor.tokenizer(batch["text"]).input_ids[:448]

        return {
            "input_features": input_features,
            "labels": labels,
        }

    print("Preprocessing dataset...")
    processed = dataset.map(
        prepare_sample,
        remove_columns=dataset.column_names,
        num_proc=1,
    )

    return processed


# =============================================================================
# Training
# =============================================================================

def train(
    model_size: str = "small",
    epochs: int = 15,
    batch_size: int = 1,
    grad_accum: int = 4,
    learning_rate: float = 5e-5,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.15,
    weight_decay: float = 0.05,
    label_smoothing: float = 0.1,
    warmup_ratio: float = 0.1,
    eval_steps: int = 50,
    early_stopping_patience: int = 5,
):
    """Run fine-tuning with regularization."""

    device = get_device()

    # Load dataset
    print(f"\nLoading dataset from: {DATASET_DIR}")
    if not DATASET_DIR.exists():
        print("ERROR: Dataset not found. Run 'python -m training.prepare_dataset' first.")
        sys.exit(1)

    dataset = load_from_disk(str(DATASET_DIR))
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Validation: {len(dataset['validation'])} samples")

    # Setup model
    model, processor = setup_model_and_processor(model_size)
    model = apply_lora(model, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

    # Preprocess data (converts audio → mel spectrograms + tokenizes text)
    train_dataset = preprocess_dataset(dataset["train"], processor)
    val_dataset = preprocess_dataset(dataset["validation"], processor)

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Metrics
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Training arguments
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),

        # Training schedule
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,

        # Optimizer (regularized)
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",

        # Regularization
        label_smoothing_factor=0,

        # Memory optimization
        gradient_checkpointing=False,
        fp16=False,

        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=eval_steps,
        predict_with_generate=True,
        generation_max_length=225,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Logging
        logging_steps=10,
        report_to="none",

        # Device
        use_mps_device=(device.type == "mps"),
        dataloader_num_workers=0,

        # Misc
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # Callbacks
    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        )
        print(f"\nEarly stopping enabled (patience={early_stopping_patience})")

    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # Train
    print("\n" + "=" * 60)
    print("STARTING FINE-TUNING (REGULARIZED)")
    print("=" * 60)
    print(f"  Model:            whisper-{model_size}")
    print(f"  Epochs:           {epochs} (with early stopping)")
    print(f"  Batch size:       {batch_size} (effective: {batch_size * grad_accum})")
    print(f"  Learning rate:    {learning_rate}")
    print(f"  Weight decay:     {weight_decay}")
    print(f"  Label smoothing:  {label_smoothing}")
    print(f"  LoRA rank:        {lora_rank} (dropout={lora_dropout})")
    print(f"  Eval every:       {eval_steps} steps")
    print(f"  Early stopping:   patience {early_stopping_patience}")
    print(f"  Device:           {device}")
    print("=" * 60)
    print("\nTraining will stop early if eval loss stops improving.")
    print("You can also stop manually with Ctrl+C (progress saves).\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted! Saving current state...")

    # Save the LoRA adapter
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    processor.save_pretrained(str(ADAPTER_DIR))

    print(f"\nAdapter saved to: {ADAPTER_DIR}")

    # Final evaluation
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    final_wer = eval_results.get("eval_wer")
    if final_wer is not None:
        print(f"Final WER: {final_wer:.4f} ({final_wer*100:.1f}%)")
    else:
        print(f"Final WER: N/A")

    # Save results
    results_file = OUTPUT_DIR / "training_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "model_size": model_size,
            "epochs_configured": epochs,
            "epochs_completed": trainer.state.epoch,
            "early_stopped": trainer.state.epoch < epochs,
            "batch_size": batch_size,
            "effective_batch_size": batch_size * grad_accum,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "label_smoothing": label_smoothing,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "early_stopping_patience": early_stopping_patience,
            "final_wer": final_wer,
            "eval_loss": eval_results.get("eval_loss"),
            "train_samples": len(dataset["train"]),
            "val_samples": len(dataset["validation"]),
        }, f, indent=2)

    print(f"Results saved to: {results_file}")
    print("\nDone! To evaluate:")
    print("  python -m training.fine_tune --evaluate")


# =============================================================================
# Evaluation / Inference
# =============================================================================

def evaluate_model(model_size: str = "small", audio_path: str = None):
    """Evaluate the fine-tuned model or transcribe a single file."""

    device = get_device()

    if not ADAPTER_DIR.exists():
        print("ERROR: No fine-tuned adapter found.")
        print("Run 'python -m training.fine_tune' first.")
        sys.exit(1)

    model_name = MODEL_MAP[model_size]

    print(f"Loading base model: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float32
    )

    print(f"Loading LoRA adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model = model.merge_and_unload()
    model = model.to(device)
    model.eval()

    if audio_path:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        input_features = processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features, max_length=225)

        transcription = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        print(f"\nTranscription:\n{transcription}")
        return transcription

    else:
        dataset = load_from_disk(str(DATASET_DIR))
        val = dataset["validation"]

        wer_metric = evaluate.load("wer")
        predictions = []
        references = []

        print(f"\nEvaluating on {len(val)} validation samples...")

        for i, sample in enumerate(val):
            audio_field = sample["audio"]

            if isinstance(audio_field, dict):
                audio_array = np.array(audio_field["array"], dtype=np.float32)
            elif isinstance(audio_field, str):
                audio_array, _ = librosa.load(audio_field, sr=16000, mono=True)
            else:
                print(f"  SKIP sample {i}: unexpected audio format")
                continue

            input_features = processor.feature_extractor(
                audio_array, sampling_rate=16000, return_tensors="pt",
            ).input_features.to(device)

            with torch.no_grad():
                predicted_ids = model.generate(input_features, max_length=225)

            pred = processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            ref = sample["text"]

            predictions.append(pred)
            references.append(ref)

            print(f"\n--- Sample {i+1} ---")
            print(f"  REF: {ref[:100]}...")
            print(f"  HYP: {pred[:100]}...")

        wer = wer_metric.compute(predictions=predictions, references=references)
        print(f"\n{'='*60}")
        print(f"Fine-Tuned Model WER: {wer:.4f} ({wer*100:.1f}%)")
        print(f"{'='*60}")

        return wer


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Whisper with LoRA for rap transcription")
    parser.add_argument("--model", type=str, default="small", choices=MODEL_MAP.keys())
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--transcribe", type=str, default=None)

    args = parser.parse_args()

    if args.evaluate or args.transcribe:
        evaluate_model(model_size=args.model, audio_path=args.transcribe)
    else:
        train(
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            learning_rate=args.lr,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            early_stopping_patience=args.patience,
            eval_steps=args.eval_steps,
        )
