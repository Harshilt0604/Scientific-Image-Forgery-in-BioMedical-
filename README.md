# Scientific Image Forgery Detection

**Pixel-level copy-move forgery segmentation for biomedical publication images**
Western blots · Microscopy · Gel electrophoresis · Flow cytometry

---

## What This Is

AgenticForensicNet is a three-agent deep learning system that detects manipulated regions in scientific images at the pixel level. Each agent analyses the same image through a forensically distinct lens — raw RGB, SRM noise residuals, and FFT frequency bands — before combining evidence through cross-agent transformer attention and spatial prototype memory retrieval.

Built for the **Recod.AI / LUC Scientific Image Forgery Detection** Kaggle competition dataset.
MOD006567 — Applications of Machine Learning, Anglia Ruskin University.

---

## Architecture

### Three Specialised Agents
- **Agent 1 — RGB Encoder:** detects colour and texture inconsistencies in raw pixel space
- **Agent 2 — SRM Noise Extractor:** applies fixed 5×5 Laplacian high-pass filters to reveal camera noise fingerprint differences between copied and authentic regions. Zero learnable parameters, output range [−1, 1]
- **Agent 3 — FFT Frequency Analyser:** applies 2D Fast Fourier Transform per channel, splits magnitude spectrum into low / mid / high bands to detect resampling artefacts at paste boundaries. Zero learnable parameters, output range [0, 1]

### Shared Encoder
- One **ResNet-18** backbone shared across all three agents (saves ~23M parameters vs three separate encoders)
- Each domain normalised via a **Domain Adapter** — Conv(3→16) → BatchNorm → **GELU** → Conv(16→3) → Sigmoid (~5,000 params each)
- GELU chosen over ReLU to preserve near-zero forensic signals in SRM and FFT channels

### Per-Agent Confidence Heads
- Each agent produces features [B, 128, 16, 16] and a spatial confidence map [B, 1, 16, 16]
- Confidence maps enable spatially adaptive fusion — agents weighted by local certainty rather than fixed equal weights

### Cross-Agent Attention Debate
- Feature maps flattened to 256 tokens per agent → 768 tokens total
- Learnable 128-dim **agent-type embedding** added per agent so attention distinguishes domains
- **Pre-LN Multi-Head Self-Attention** with 8 heads across all 768 tokens
- Attended outputs combined via confidence-gated residual: `feat = feat + conf × debated_feat`

### Spatial Prototype Memory
- **CPU-resident** memory bank (embed_dim=128, bank_size=256, top_k=8)
- Populated with real forgery patch embeddings during training via no-gradient writes
- Registered as PyTorch buffer; `_apply()` overridden to keep bank on CPU after `model.to(device)`
- At inference: cosine similarity query → top-8 prototype retrieval via cross-attention
- Analogous to retrieval-augmented generation (Lewis et al., 2020)

### ForgeryDecoder
- Concatenates three agent maps → 384 channels (3 × 128)
- Two-stage upsampling: 16×16 → 64×64 → 256×256 via learned transposed convolutions
- Skip connection from encoder `layer1` at 64×64 to preserve fine boundary detail
- No sigmoid at output — BCEWithLogitsLoss handles sigmoid numerically

---

## Dataset

**Recod.AI / LUC Scientific Image Forgery Detection** (Kaggle)
- Biomedical publication images with pixel-level copy-move annotations
- Ground truth masks stored as NumPy `.npy` arrays (variable formats: 2D, 3D single-channel, 3D multi-channel)
- Custom loader handles all mask formats

### Data Audit & Preprocessing (Notebook 1)
- Seed = 42 for full reproducibility
- Five folders validated via stem-based filename matching: `train_images/authentic`, `train_images/forged`, `train_masks`, `supplemental_images`, `supplemental_masks`
- Image–mask pairing confirmed; orphan masks detected and logged
- `positive_ratio` computed per mask → four difficulty groups:

| Group | Forged Pixel Fraction |
|-------|----------------------|
| Tiny | < 0.001 |
| Small | < 0.01 |
| Medium | < 0.05 |
| Large | ≥ 0.05 |

- Patch-based training (primary patch size: **512px**) — full-image resize would destroy tiny forgery regions
- Patch sampling: 50% positive · 20% hard negative · 15% random negative · 15% authentic
- Tiny and small difficulty groups **oversampled**
- **80/20 stratified train/validation split** via sklearn `train_test_split`, stratification key = label + difficulty group

---

## Training

| Setting | Value |
|---------|-------|
| Framework | PyTorch |
| GPU | NVIDIA T4 (Kaggle) |
| Epochs | 40 |
| Optimiser | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=40, eta_min=1e-7) |
| Batch size | 4 physical / 16 effective (gradient accumulation ×4) |
| Precision | Mixed (AMP + GradScaler) |
| Memory | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

### Loss Function — CombinedLoss
- **Focal Loss** (α=0.75, γ=2.0) — down-weights easy authentic-pixel majority
- **Dice Loss** (weight=1.0) — directly optimises mask overlap
- Combination chosen because tiny forgery groups have < 0.1% forged pixels

### Augmentation (Albumentations) — conservative to protect forensic signals
- `Resize(256, 256)`
- `HorizontalFlip(p=0.5)` · `VerticalFlip(p=0.5)` · `RandomRotate90(p=0.5)`
- `ShiftScaleRotate(shift=0.05, scale=0.10, rotate=15°, border_mode=0, p=0.70)`
- ❌ No colour jitter — corrupts SRM noise residuals
- ❌ No JPEG compression simulation — distorts FFT agent artefact patterns
- ❌ No Cutout — may erase forged regions and invert training labels

---

## Results

Evaluation with **Test-Time Augmentation (TTA)** — predictions averaged across 4 passes.

| Model | Val IoU | Val F1 | Precision | Recall |
|-------|---------|--------|-----------|--------|
| ResNet-18 U-Net Baseline | 0.412 | 0.583 | 0.601 | 0.566 |
| **AgenticForensicNet (ResNet-18)** | **0.531** | **0.694** | **0.712** | **0.677** |
| AgenticForensicNet (ResNet-34)* | 0.558 | 0.717 | 0.739 | 0.697 |

*ResNet-34 variant by teammate (Notebook 3) — same architecture, larger shared encoder.

**+28.9% relative IoU improvement** over baseline. Gains attributable entirely to architecture — same backbone, training config, data split, and evaluation protocol across both models.

---

## Notebooks

| Notebook | Contributor | Contents |
|----------|-------------|----------|
| `Data_Audit_and_Analysis.ipynb` | Teammate | Dataset audit, pairing validation, mask content analysis, preprocessing decisions, stratified split |
| `scientific-image-forgery-resnet-18.ipynb` | Me | AgenticForensicNet full implementation + ResNet-18 baseline |
| `scientific-image-forgery-withresnet-34.ipynb` | Teammate | ResNet-34 backbone variant |

---

## Stack

Python · PyTorch · segmentation-models-pytorch · Albumentations · NumPy · Pandas · Kaggle T4 GPU

