# 🎤 Rap Transcription System

**AI-powered transcription for rap music with slang recognition**

A production-grade Automatic Speech Recognition (ASR) system specifically designed for transcribing rap music, featuring:

- 🧠 **85M parameter Conformer encoder** with dual prediction heads
- 🎯 **Slang-aware transcription** with custom rap vocabulary
- ⚡ **Real-time inference** faster than real-time on GPU/MPS
- 🌐 **REST API & WebSocket** for easy integration
- 🐳 **Docker ready** for deployment

---

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd rap-transcriber

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install python-multipart
```

### Transcribe Audio
```bash
# Using the CLI
python scripts/transcribe.py your_audio.mp3

# With lyrics formatting
python scripts/transcribe.py your_audio.mp3 --format-lyrics

# Save to file
python scripts/transcribe.py your_audio.mp3 --output transcript.txt
```

### Start the API Server
```bash
python scripts/run_api.py
```

Then visit: http://localhost:8000/docs

---

## 📁 Project Structure
```
rap-transcriber/
├── src/
│   ├── models/           # Neural network architecture
│   │   ├── encoder.py    # Conformer encoder (75M params)
│   │   ├── phoneme_head.py   # Phoneme prediction head
│   │   ├── text_head.py      # Text prediction head
│   │   └── rap_transcriber.py # Main model wrapper
│   │
│   ├── data/             # Data processing
│   │   ├── audio_processor.py    # Audio loading & preprocessing
│   │   ├── feature_extractor.py  # Mel spectrogram extraction
│   │   ├── dataset.py            # PyTorch dataset classes
│   │   ├── tokenizer.py          # Text tokenization
│   │   ├── slang_lexicon.py      # Rap slang dictionary
│   │   └── manifest.py           # Data manifest management
│   │
│   ├── training/         # Training pipeline
│   │   ├── trainer.py    # Training loop
│   │   └── optimizer.py  # Optimizer & schedulers
│   │
│   ├── inference/        # Inference pipeline
│   │   ├── engine.py     # Inference engine
│   │   ├── decoder.py    # Beam search decoder
│   │   └── postprocessor.py  # Text cleanup
│   │
│   ├── api/              # REST API
│   │   ├── server.py     # FastAPI server
│   │   ├── client.py     # API client
│   │   └── websocket.py  # WebSocket streaming
│   │
│   └── utils/            # Utilities
│       ├── config.py     # Configuration loader
│       ├── metrics.py    # WER/CER metrics
│       └── slang_metrics.py  # Slang accuracy
│
├── scripts/              # Executable scripts
│   ├── train.py          # Training script
│   ├── transcribe.py     # Transcription CLI
│   ├── evaluate.py       # Evaluation script
│   ├── benchmark.py      # Performance benchmarks
│   └── run_api.py        # API server launcher
│
├── configs/              # Configuration files
│   └── config.yaml       # Main config
│
├── data/                 # Data directories
│   ├── raw/              # Raw audio files
│   ├── processed/        # Processed manifests
│   └── lexicon/          # Slang lexicon
│
└── outputs/              # Output directories
    ├── checkpoints/      # Model checkpoints
    ├── logs/             # Training logs
    └── results/          # Evaluation results
```

---

## 🏋️ Training

### Prepare Data

1. Place audio files in `data/raw/`
2. Create matching text files (same name, `.txt` extension)
3. Run data preparation:
```bash
python scripts/prepare_data.py --audio-dir data/raw --output-dir data/processed
```

### Train the Model
```bash
# Debug mode (quick test)
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train.py --debug --epochs 2

# Full training
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train.py --epochs 100
```

### Evaluate
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt
```

---

## 🌐 API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/transcribe` | POST | Transcribe audio file |
| `/transcribe/async` | POST | Start async transcription |
| `/job/{job_id}` | GET | Get async job status |
| `/format` | POST | Format/clean text |

### WebSocket

Connect to `/ws/transcribe` for streaming transcription.

### Example Usage
```python
from src.api.client import TranscriptionClient

client = TranscriptionClient("http://localhost:8000")

# Transcribe
result = client.transcribe("audio.mp3")
print(result['text'])

# Format text
formatted = client.format_text(
    "i'm finna get bread",
    normalize_slang=True
)
```

---

## 🐳 Docker

### Build & Run
```bash
# Build image
docker build -t rap-transcriber .

# Run container
docker run -p 8000:8000 rap-transcriber

# Or use Docker Compose
docker-compose up -d
```

---

## 📊 Model Architecture
```
Input Audio
    ↓
[Mel Spectrogram] (80 bins)
    ↓
[Conv Subsampling] (4x reduction)
    ↓
[Conformer Encoder] (12 layers, 512 dim, 8 heads)
    ↓
    ├──→ [Phoneme Head] → ARPAbet phonemes
    │
    └──→ [Text Head] → BPE tokens → Text
```

**Total Parameters:** ~85 million

---

## 🎯 Slang Support

The system includes a custom slang lexicon with 40+ rap terms:

- **Ad-libs:** yeah, yuh, skrt, ayy, sheesh
- **Slang verbs:** finna, tryna, bussin
- **Slang nouns:** bread, bands, drip, cap
- **Pronouns:** bruh, fam, dawg

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Parameters | 85M |
| Inference (10s audio) | ~100ms |
| Real-time Factor | < 0.1x |
| Supported Formats | mp3, wav, flac, m4a |

---

## 🛠️ Requirements

- Python 3.9+
- PyTorch 2.0+
- 16GB RAM recommended
- GPU/MPS optional but recommended

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/)
- [torchaudio](https://pytorch.org/audio/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Conformer](https://arxiv.org/abs/2005.08100) architecture
