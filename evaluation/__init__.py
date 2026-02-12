"""Rap transcription evaluation framework."""
from .evaluate import (
    evaluate_sample,
    run_evaluation,
    generate_report,
    compute_wer,
    categorize_errors,
    normalize_text,
    tokenize,
    EvalResult,
)
