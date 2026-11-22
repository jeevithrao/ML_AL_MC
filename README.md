# ML_AL_MC: bird sound detection on BirdCLEF 2024 (active learning + model compression)

Live demo (runs on the phone; no audio is uploaded): https://jeevithrao.github.io/ML_AL_MC/

## 1. Overview
Recognition of 182 Western Ghats bird species from sound (BirdCLEF 2024) with a mel-spectrogram CNN. The project has four parts:
1. **Baseline**: EfficientNet-B0 trained on Xeno-canto focal recordings.
2. **Active learning**: a simulated study of which recordings to label next (random, entropy, margin, core-set).
3. **Compression**: knowledge distillation into MobileNetV3-small, plus a documented INT8 quantization failure.
4. **Web demo**: the compressed model running in a phone browser with ONNX Runtime Web.

## 2. Repository structure
```
notebooks/   one notebook per phase (run in numeric order)
src/         reusable Python modules called by the notebooks
results/     small outputs: splits, metrics, plots, active learning runs, efficiency tables
docs/        web demo (index.html, app.js, dsp.js, data/, model/) served by GitHub Pages
requirements.txt
```
Data, spectrograms and PyTorch checkpoints are **not** stored in Git (download links in section 11).

## 3. Modules (src/)
| Module | What it does |
|---|---|
| `audio_config.py` | Audio and spectrogram constants shared by training and the web demo: 32 kHz, 5 s windows, 128 mel bands, n_fft 1024, hop 512, 50-14,000 Hz, 80 dB range, up to 3 loudest windows per clip from the first 60 s |
| `preprocess.py` | `clip_to_mels`: reads one recording and returns its log-mel windows (float16), start times and loudness |
| `data_utils.py` | `build_targets` (multi-hot labels from primary and secondary species), `sqrt_inv_weights` (class-balancing sampler weights), `MelDataset` (memory-mapped dataset) |
| `data_prep.py` | `consolidate` (packs all windows into one array), `prepare` (loads index, targets, splits and runs a leakage check) |
| `train_config.py` | Training hyperparameters (batch 64, 10 epochs, AdamW, mixup, SpecAugment) |
| `model.py` | `make_model`: timm model with a 1-channel input and 182 outputs |
| `metrics.py` | `per_class_auc`, `macro_auc` (competition metric, skipping classes without positives) |
| `engine.py` | GPU-side augmentation (time shift, SpecAugment, mixup), data loaders, `predict`, and `fit` (mixed precision, resumable checkpoints) |
| `al_config.py` | Active learning settings (5% seed, +5% per round, 5 rounds) |
| `al.py` | Active learning: class-stratified seed, strategies (random, entropy, margin, core-set), resumable loop |
| `distill.py` | `kd_loss` (sigmoid-based distillation loss) and `fit_kd` |
| `profile_tools.py` | Efficiency measurements: parameters, size, FLOPs, CPU latency, mel time, windows per second |
| `onnx_tools.py` | Scoring and timing of ONNX models with ONNX Runtime |

Web demo (docs/): `dsp.js` (spectrogram, resampler and windowing in JavaScript, mirrors librosa), `app.js` (recording, file input, inference, self-test), `index.html` (page), `data/` (mel filterbank, window, class names, config, self-test fixtures), `model/student_fp32.onnx`.

## 4. Requirements
**Environment used:** Google Colab, Python 3.13, NVIDIA T4 GPU (free-tier GPU availability varies; training code saves resumable checkpoints).

**Accounts:** Google (Colab and Drive), Kaggle (join the BirdCLEF 2024 competition and accept its rules, otherwise the download fails with 403).

**Space:** the competition download is 21.8 GB (local Colab disk); the spectrogram archive is 5 GB on Drive; the free Drive tier is 15 GB.

**Libraries** (pinned versions in `requirements.txt`): torch, timm, numpy, pandas, scikit-learn, scipy, librosa, soundfile, joblib, tqdm, matplotlib, onnx, onnxruntime, kaggle. The web demo loads ONNX Runtime Web 1.20.1 from a CDN. On Colab, torch is preinstalled; install the rest with:
```
!pip install -q timm onnx onnxruntime soundfile librosa scikit-learn kaggle
```

## 5. How to run (Google Colab)
1. Create a Kaggle API token and add `KAGGLE_USERNAME` and `KAGGLE_KEY` as Colab Secrets (key icon in the left sidebar, notebook access on).
2. Open a notebook in Colab (links below). **Replace its first cell** with this public setup cell (the original uses the author's private secrets):
```python
import os, sys
from google.colab import drive, userdata
drive.mount('/content/drive')
REPO_DIR = "/content/ML_AL_MC"
if not os.path.exists(REPO_DIR):
    !git clone https://github.com/jeevithrao/ML_AL_MC.git {REPO_DIR}
%cd {REPO_DIR}
sys.path.append(f"{REPO_DIR}/src"); os.environ["PYTHONPATH"] = f"{REPO_DIR}/src"
os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
!mkdir -p /content/drive/MyDrive/birdclef_project/checkpoints
```
3. Run the notebooks in order. Each restores what it needs from Drive, so a runtime reset costs minutes, not work.

| Notebook | Purpose | Runtime | Main outputs |
|---|---|---|---|
| [01_eda](https://colab.research.google.com/github/jeevithrao/ML_AL_MC/blob/main/notebooks/01_eda.ipynb) | Download data, explore classes, durations, sample rates | CPU | `results/clip_durations.csv` |
| [02_preprocess](https://colab.research.google.com/github/jeevithrao/ML_AL_MC/blob/main/notebooks/02_preprocess.ipynb) | Log-mel windows, recording-level train/val/test split | CPU | Drive `mels.tar`; `results/chunk_index.csv`, `file_splits.csv`, `classes.json` |
| [03_baseline](https://colab.research.google.com/github/jeevithrao/ML_AL_MC/blob/main/notebooks/03_baseline.ipynb) | Train EfficientNet-B0 | GPU | Drive `checkpoints/baseline_best.pt`; `results/baseline_*` |
| [04_active_learning](https://colab.research.google.com/github/jeevithrao/ML_AL_MC/blob/main/notebooks/04_active_learning.ipynb) | Compare labeling strategies over seeds | GPU | `results/al/*.json`, `al_summary.csv`, `al_curves*.png` |
| [05_compression](https://colab.research.google.com/github/jeevithrao/ML_AL_MC/blob/main/notebooks/05_compression.ipynb) | Distillation, ONNX export, INT8 attempts, latency, rare-species check | GPU (training), CPU (rest) | `results/efficiency.csv`, `final_compression_table.csv`, `pareto.png` |
| [06_soundscape_and_web_export](https://colab.research.google.com/github/jeevithrao/ML_AL_MC/blob/main/notebooks/06_soundscape_and_web_export.ipynb) | Real-soundscape sanity test, export web assets | CPU | `results/soundscape_sanity.csv`; `docs/data`, `docs/model` |

Approximate times on a Colab T4 (measured here): baseline 10 epochs about 18 min; one active learning strategy run (6 rounds) about 21-23 min; student training about 7 min (plain) and about 10 min (distilled); data download about 5 min.

Paths are Colab paths (`/content/...` and `/content/drive/MyDrive/birdclef_project/...`). To run elsewhere, edit the constants in `src/data_prep.py` and the path variables at the top of each notebook.

## 6. Web demo
Open https://jeevithrao.github.io/ML_AL_MC/ on a phone (tested: Android Chrome). Tap Record (5-30 s) or choose an audio file; each 5-second window is scored on the device. "Run self-test" (under About) checks the spectrogram and model output against Python.

To run it locally: `cd docs && python -m http.server 8000`, then open http://localhost:8000 (microphone access works on localhost; ONNX Runtime Web is fetched from the internet). To host it yourself: repository Settings, Pages, Deploy from branch `main`, folder `/docs`. Regenerate `docs/data` and `docs/model` with the export cell in notebook 06.

## 7. Results
All metrics are macro ROC-AUC over 182 classes on clean focal recordings, split by recording (no recording appears in two splits).

**Baseline** (EfficientNet-B0, 4.24 M parameters): validation 0.9627, test 0.9653.

**Active learning** (5% seed, +5% per round up to 30% of 17,125 training files; mean paired difference to random over 4 seeds, entropy over 2):

| Strategy | Final macro-AUC vs random | Rare-species AUC vs random | Speed-up factor | Rare-species files labeled vs random |
|---|---|---|---|---|
| Core-set | +0.0097 | +0.030 | 0.94 | 1.70x |
| Margin | +0.0048 | +0.016 | 0.96 | 1.26x |
| Entropy (2 seeds) | +0.0036 | +0.013 | 0.96 | 1.51x |

Gains over the whole learning curve are smaller than the spread between two random runs (0.0087), so the benefit is concentrated at larger label budgets and in rare species.

**Compression** (ONNX Runtime, 2 CPU threads, Colab; timings vary between sessions):

| Model | Size | Parameters | GFLOPs | Val AUC | Test AUC |
|---|---|---|---|---|---|
| B0 FP32 | 16.94 MB | 4.24 M | 0.615 | 0.9627 | 0.9653 |
| MobileNetV3-small, distilled (T=2, alpha=0.5) | 6.82 MB | 1.70 M | 0.085 | 0.9611 | 0.9634 |
| Same student, INT8 (not used) | 2.58 MB | 1.70 M | 0.085 | 0.8130 | 0.8114 |

The student is about 8x faster than B0 at batch 32 and 2.8-6.5x at batch 1. Distillation beat plain training of the same student in two of two seeds (mean test AUC 0.9616 vs 0.9561; teacher 0.9653). Post-training INT8 failed in 9 variants and was slower on the test CPU; 8-bit weights alone are harmless (0.9611), so the loss comes from activation quantization.

**On-device speed:** on an Android phone, about 17 ms for the spectrogram plus about 24 ms for the model per 5-second window.

**Soundscape sanity test** (100 unlabeled recordings, 4,800 windows, no labels): teacher and student agree on the top-1 species in 80.9% of teacher-confident windows; species-set overlap at threshold 0.5 is 0.47. This measures agreement, not accuracy. Informal phone tests of three recordings (Indian Peafowl, Southern Hill Myna, Indian Scimitar-Babbler) were judged correct by the authors; this is an anecdote, not an accuracy estimate.

## 8. Reproducibility
The split is fixed and committed (`results/file_splits.csv`, seed 42). Training seeds are fixed (`SEED = 42`; active learning seeds 0-3). GPU nondeterminism means reruns differ slightly. Long jobs save after every epoch or round and resume if the session drops.

## 9. Limitations
- Training and test data are clean focal recordings; accuracy on noisy field soundscapes was not measured, and published BirdCLEF 2024 scores (about 0.65-0.69) are on a different, harder test.
- Active learning was simulated with free labels on focal recordings, not on expensive soundscape labels.
- Rare-species scores rest on few validation files and are noisy; several results use one or two seeds.
- Recordists were not grouped across splits.
- Each spectrogram window is normalized individually, so near-silent audio can produce false detections.

## 10. License, data and credits
Code license: MIT (see LICENSE). Model weights and any audio derived from Xeno-canto or BirdCLEF recordings follow their own terms (Xeno-canto recordings carry Creative Commons licenses that vary per recording); check them before reuse.

Data: BirdCLEF 2024 (Kahl et al., Overview of BirdCLEF 2024, CEUR-WS Vol. 3740) with recordings from Xeno-canto. Built with PyTorch, timm, librosa, ONNX Runtime and ONNX Runtime Web.

## 11. Resources for evaluators (large files are not in Git)
- GitHub repository: https://github.com/jeevithrao/ML_AL_MC
- Live demo: https://jeevithrao.github.io/ML_AL_MC/
- Google Drive folder (shared with the instructor): https://drive.google.com/drive/folders/1QqTkX97V40NsMkjmBe8fDgNYfWXJ0idQ?usp=sharing
  - mels.tar (5 GB): preprocessed spectrograms, restored automatically by the notebooks
  - checkpoints/: baseline_best.pt (EfficientNet-B0), mnv3s_kd_T2_a05_best.pt (distilled student) and the other student checkpoints
  - al_runs/: active learning run files (also copied to results/al in this repository)
  - export/: ONNX models (b0_fp32.onnx, student_fp32.onnx)
- Dataset: BirdCLEF 2024 on Kaggle, https://www.kaggle.com/c/birdclef-2024 (join the competition and accept its rules to download)
