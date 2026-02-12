"""Training module for Whisper fine-tuning on rap data."""
from .data_collector import (
    add_training_sample,
    get_status,
    export_for_training,
    load_manifest,
)
