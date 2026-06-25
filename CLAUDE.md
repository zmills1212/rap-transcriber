# CLAUDE.md — Rap Transcriber

## Project Overview

Domain-specialized ASR for rap music. Fine-tunes OpenAI Whisper (small, 242M params) with LoRA adapters to transcribe rap lyrics from audio — a domain where all existing ASR fails due to non-standard phonology, slang, rapid delivery, ad-libs, and production effects.

**Long-term vision:** Accurate transcription *without* reference lyrics. Start with rap (where verified lyrics exist via Genius), then generalize to courtrooms, regional dialects, oral history — any non-standard English domain.

**Three-stage pipeline:**
1. LoRA-adapted Whisper → raw transcription
2. Rule-based post-processor → phonetic corrections
3. Phonetic lyrics matcher (Genius) → final output

## Current State (as of June 2026)

- **211 songs** from ~40+ artists collected (audio via yt-dlp, lyrics via Genius API)
- **~192 verified-clean training segments** after filtering (30s chunks with aligned lyrics)
- **Best WER: 70.8%** (on 155 segments), most recent: **86.4%** (192 segments, different val set)
- Model produces near-perfect transcriptions on well-aligned segments
- WER inflated by noisy validation segments; qualitative results are strong
- Currently processing 211 songs through the full pipeline to scale dataset

## Current Task

Scale training data from ~192 to 400+ clean segments by:
1. Running `clean_lyrics --fix` on all lyrics files (strip Genius metadata)
2. Running `segment_audio` (v2 sequential alignment) on all 211 songs
3. Running `model_filter --remove --threshold 0.12` to remove misaligned segments
4. Rebuilding dataset and retraining

After reaching 400+ clean segments, planned upgrades:
- Whisper-medium (769M params) as base model
- LoRA rank 16 (currently rank 8)
- Proper held-out test set (20-30 manually verified segments)

## Project Structure

```
rap-transcriber/
├── CLAUDE.md
├── .env                          # GENIUS_API_TOKEN
├── venv/                         # Python 3.9 virtualenv
│
├── training/                     # Core training pipeline
│   ├── fine_tune.py              # Main training script (LoRA fine-tuning)
│   ├── segment_audio.py          # v2 sequential greedy alignment
│   ├── prepare_dataset.py        # Builds HuggingFace dataset from manifest
│   ├── auto_collect.py           # Batch song collection (yt-dlp + Genius)
│   ├── data_collector.py         # Manifest management
│   ├── clean_lyrics.py           # Genius metadata stripping
│   ├── audit_segments.py         # Whisper-based alignment auditor
│   ├── model_filter.py           # Model-based data quality filter
│   └── hf_dataset/               # Built HuggingFace datasets
│       └── rap_transcription/
│
├── training_data/
│   ├── manifest.json             # Master index of all songs
│   ├── segment_manifest.json     # Index of 30s segments with lyrics
│   ├── audio/                    # Raw audio files (16kHz mono WAV)
│   ├── transcripts/              # Lyrics text files
│   └── segments/                 # 30s audio chunks
│
├── lyrics_matcher/
│   └── genius.py                 # Genius API client (lyrics fetching)
│
├── post_processor/               # Rule-based post-processing
│
└── web_ui.py                     # Web interface (built but untested)
```

## Key Commands

All commands run from project root with venv activated:
```bash
cd ~/Desktop/Projects_2026/rap-transcriber
source venv/bin/activate
```

### Data Collection
```bash
# Add single song
python -m training.auto_collect --song "Song Name" --artist "Artist" --tags tag1 tag2

# Batch collect from file (format: "Artist - Song | tag1 tag2")
python -m training.auto_collect --batch songs.txt

# Check collection status
python -m training.data_collector status
```

### Data Cleaning & Segmentation
```bash
# Strip Genius metadata from lyrics (dry run first, then --fix)
python -m training.clean_lyrics
python -m training.clean_lyrics --fix --verbose

# Segment songs into 30s chunks with aligned lyrics
python -m training.segment_audio

# Audit segment quality (fast metadata check)
python -m training.audit_segments --no-whisper

# Audit with Whisper alignment check
python -m training.audit_segments --n 30
python -m training.audit_segments --all --remove-bad

# Model-based quality filter (best filter — uses fine-tuned model)
python -m training.model_filter --remove --threshold 0.12
```

### Training
```bash
# Build HuggingFace dataset from segment manifest
python -m training.prepare_dataset --segments

# Train (use caffeinate to prevent Mac sleep)
caffeinate -i python -m training.fine_tune

# Evaluate
python -m training.fine_tune --evaluate
```

### Full Pipeline (run in order)
```bash
python -m training.clean_lyrics --fix
python -m training.segment_audio
python -m training.model_filter --remove --threshold 0.12
python -m training.prepare_dataset --segments
caffeinate -i python -m training.fine_tune
```

## Architecture Details

### Model Configuration
- **Base model:** `openai/whisper-small.en` (242M params)
- **LoRA:** rank 8, alpha 16, dropout 0.15, targets q_proj + v_proj
- **Trainable params:** 884,736 / 242,618,880 (0.36%)
- **Training:** batch_size=1, grad_accum=4, lr=5e-5, cosine schedule, weight_decay=0.05
- **Early stopping:** patience 5 on eval_loss
- **Gradient checkpointing:** DISABLED (conflicts with PEFT/LoRA on this transformers version — causes `requires_grad=True` warning that blocks gradient flow)
- **Label smoothing:** 0 (non-zero causes ValueError with Whisper's decoder_input_ids handling in this transformers version)
- **Device:** MPS (Apple Silicon)

### Segmentation Pipeline (v2)
- Splits songs into 30s chunks with 5s overlap
- Transcribes each chunk with baseline Whisper
- **Sequential greedy alignment:** each chunk's lyrics search starts where previous chunk ended (enforces monotonic ordering — the key fix over v1's proportional positioning)
- Scores alignment via combined word-overlap + bigram-overlap
- Stores alignment_score per segment in manifest

### Model Filter
- Uses fine-tuned model to transcribe every training segment
- Scores via word overlap (0.3 weight) + bigram sequence overlap (0.7 weight)
- Repetition detection (any 2-4 word phrase repeating 5+ times → 0.3x penalty)
- Threshold 0.12 for removal (was 0.08, tightened after false negatives)

### Data Quality Issues (Solved)
- **Genius metadata contamination:** Contributor headers, song descriptions, translation tags, "Read More" markers embedded in lyrics files. Fixed by `clean_lyrics.py`.
- **Segment misalignment:** v1 segmentation used proportional positioning which assigned wrong lyrics to wrong chunks. Fixed by v2 sequential greedy alignment.
- **Repetition collapse:** Whisper degenerates into loops ("I'm a bitch, I'm a bitch...") when audio/lyrics don't match. This is a symptom of bad data, not a model bug. Fixed by aggressive filtering.
- **Gradient checkpointing conflict:** `gradient_checkpointing=True` + PEFT LoRA causes `requires_grad=True` warning every step, blocking gradient flow. Fixed by disabling gradient checkpointing.

## Environment Notes

- **Python:** 3.9 (system Python via CommandLineTools — limits yt-dlp version)
- **yt-dlp:** Homebrew version at `/opt/homebrew/bin/yt-dlp` (2026.03.17) — the venv version is old and gets YouTube 403s. `auto_collect.py` is patched to use the Homebrew path.
- **Key packages:** transformers, peft, whisper (openai-whisper), datasets, evaluate, librosa, torch (MPS backend), beautifulsoup4
- **GENIUS_API_TOKEN:** stored in `.env` at project root
- **Training time:** ~30-60 min for 150-250 segments on M1 MacBook

## Known Issues & Gotchas

1. **Don't use `python` — use `python3` or activate venv first** (system has no `python` alias)
2. **yt-dlp in venv is outdated** — always use `/opt/homebrew/bin/yt-dlp` for downloads
3. **Token length warnings** during preprocessing ("sequence length > 1024") — some segments have lyrics longer than Whisper's token limit. They get truncated. Not critical but worth cleaning up.
4. **Eval WER is noisy** — val set still contains some misaligned segments. WER numbers across runs aren't directly comparable due to different val set sizes.
5. **SoundCloud frequently 404s** — YouTube fallback handles most cases
6. **The model filter uses the current fine-tuned adapter** — if the adapter is bad, the filter is less effective. Bootstrap problem: clean enough data to get a decent model, then use that model to clean more data.

## WER History

| Run | Segments | WER | Notes |
|-----|----------|-----|-------|
| 1 | 574 train (dirty) | 127.0% | Genius metadata + misaligned segments |
| 2 | 522 train (partial clean) | 159.6% | Gradient checkpointing still broken |
| 3 | ~400 (filtered) | 142.0% | Gradient flow fixed, still noisy data |
| 4 | ~340 (more filtering) | 126.8% | Better but still high |
| 5 | 155 (aggressively filtered) | 84.1% | Clean data works |
| 6 | 155 (re-filtered) | 70.8% | Best WER achieved |
| 7 | 192 (expanded + filtered) | 86.4% | More data, different val set |
| 8 | 192 (re-filtered 0.12) | 86.4% | Same — plateau on whisper-small |

## Verification Checks

Before any training run, verify:
- [ ] `gradient_checkpointing=False` in fine_tune.py
- [ ] `label_smoothing_factor=0` in fine_tune.py
- [ ] No `requires_grad=True` warning in first 50 steps of training output
- [ ] Segment manifest has 0 metadata-contaminated entries (`audit_segments --no-whisper`)
- [ ] Train loss is decreasing after epoch 1
