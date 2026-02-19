# 🎤 Rap Transcriber

**AI-powered lyric transcription for rap music — built to handle slang, dialect, and fast delivery where standard speech-to-text fails.**

Existing tools like Genius rely on user-submitted lyrics that are often inaccurate or missing entirely. Standard ASR models (trained on news anchors and podcasts) consistently fail on rap's linguistic complexity — AAVE dialect, regional slang, rapid-fire delivery, heavy ad-libs, and dense vocal mixing.

This system solves that with a three-stage pipeline:

```
Audio → Whisper → Post-Processor → Lyrics Matcher → Accurate Transcription
```

**Each stage measurably improves accuracy:**

| Stage | WER | Improvement |
|-------|-----|-------------|
| Whisper (baseline) | 42.6% | — |
| + Post-Processor | 35.5% | -7.1% |
| + Lyrics Matcher | 1.9% | -33.6% |

> *Benchmarked on rap samples with heavy slang, dialect, and ad-libs*

---

## Quick Start

```bash
# Setup
git clone https://github.com/zmills1212/rap-transcriber.git
cd rap-transcriber
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Transcribe with reference lyrics (highest accuracy)
python rap_transcribe.py song.mp3 --lyrics reference.txt -v

# Transcribe without reference (Whisper + post-processing)
python rap_transcribe.py song.mp3 -v

# Save output
python rap_transcribe.py song.mp3 --lyrics reference.txt -o output.txt

# JSON output for integration
python rap_transcribe.py song.mp3 --lyrics reference.txt --json
```

---

## How It Works

### Stage 1: Whisper ASR
OpenAI's Whisper (small.en, 244M params) provides the raw transcription. It handles general English well but consistently misinterprets rap-specific language:

- `"bands"` → `"pants"` (slang misidentification)
- `"all these hoes gon' flock"` → `"how do you hug and fly"` (complete hallucination)
- `"doin' my thang"` → `"doing my thing"` (dialect normalization)

### Stage 2: Rap Post-Processor
150+ rule-based corrections targeting predictable Whisper errors:

- **Slang recovery:** Restores rap vocabulary Whisper normalizes away
- **Dialect preservation:** `doing` → `doin'`, `thing` → `thang`, `lying` → `lyin'`
- **Ad-lib recognition:** Identifies and preserves rap ad-libs (ayy, skrrt, etc.)
- **Context-aware phrases:** Multi-word corrections using surrounding context

### Stage 3: Lyrics Matcher
When reference lyrics are available, the matcher aligns Whisper's output against the known text using:

- **Sequence alignment** (Needleman-Wunsch variant via `difflib`)
- **Phonetic matching** with rap/AAVE equivalence tables (`bands`↔`pants`, `dat`↔`that`, `thang`↔`thing`)
- **Edit distance scoring** (Levenshtein) for unknown word pairs
- **Per-word confidence scoring** — high when Whisper and reference agree, low when the matcher had to guess

---

## Pipeline Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Audio In  │────▶│  Whisper ASR     │────▶│  Post-Processor │
│  (.mp3/.wav)│     │  (small.en 244M) │     │  (150+ rules)   │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
                    ┌──────────────────┐     ┌─────────────────┐
                    │  Final Output    │◀────│  Lyrics Matcher │
                    │  + Confidence    │     │  (if ref avail) │
                    └──────────────────┘     └─────────────────┘
```

---

## Project Structure

```
rap-transcriber/
├── rap_transcribe.py          # Unified CLI — main entry point
├── lyrics_matcher/            # Phase A: Lyrics alignment engine
│   ├── lyrics_matcher.py      #   Core matching algorithm
│   └── pipeline_eval.py       #   Full pipeline benchmarking
├── postprocessor/             # Rap-specific text correction
│   ├── rap_postprocessor.py   #   150+ correction rules
│   └── eval_postprocessor.py  #   Post-processor benchmarking
├── training/                  # Phase B: Whisper fine-tuning
│   ├── fine_tune.py           #   LoRA fine-tuning on Apple Silicon
│   ├── augment_audio.py       #   Data augmentation pipeline
│   ├── prepare_dataset.py     #   Dataset preparation
│   └── data_collector.py      #   Training data collection CLI
├── evaluation/                # WER evaluation framework
│   └── evaluate.py            #   Systematic error analysis
├── training_data/             # Curated rap audio + transcripts
│   ├── manifest.json          #   51 samples with challenge tags
│   └── transcripts/           #   Ground truth lyrics
└── test_data/                 # Held-out evaluation samples
```

---

## Training Pipeline (Phase B)

Fine-tuning Whisper on rap-specific data using LoRA (Low-Rank Adaptation):

```bash
# Collect training data
python -m training.data_collector add \
    --audio song.mp4 --lyrics lyrics.txt \
    --artist "Artist" --song "Song" \
    --tags heavy_slang fast_flow

# Generate augmented training data (7 variants per sample)
python -m training.augment_audio

# Prepare HuggingFace dataset
python -m training.prepare_dataset

# Fine-tune with regularization + early stopping
python -m training.fine_tune
```

**Training details:**
- 51 base samples → 408 via augmentation (speed, pitch, noise perturbation)
- LoRA rank 8 on attention layers (884K trainable params / 0.36%)
- Weight decay 0.05, cosine LR schedule, early stopping (patience 3)
- Optimized for M1 MacBook Pro 16GB (MPS backend)

**Data categories:** heavy slang, fast flow, melodic/autotune, mumble, ad-libs, accent/dialect, clean delivery, loud beats

---

## Evaluation

```bash
# Batch evaluation on all test samples
python -m lyrics_matcher.pipeline_eval --batch

# Evaluate post-processor independently
python -m postprocessor.eval_postprocessor
```

---

## Technical Stack

- **ASR:** OpenAI Whisper (small.en) via `openai-whisper`
- **Fine-tuning:** HuggingFace Transformers + PEFT (LoRA)
- **Audio processing:** librosa, soundfile
- **Evaluation:** `jiwer` (WER), custom error categorization
- **Runtime:** Python 3.9+, PyTorch 2.8, Apple Silicon MPS

---

## Requirements

- Python 3.9+
- 16GB RAM (for Whisper small model)
- macOS with Apple Silicon recommended (MPS acceleration)
- Works on CPU/CUDA as well

```bash
pip install openai-whisper torch transformers peft librosa jiwer evaluate
```

---

## License

MIT
