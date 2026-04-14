# Scientific Image Forgery Detection

Pixel-level forgery segmentation for biomedical publication images.
Western blots · Microscopy · Gel electrophoresis · Flow cytometry

## What this is
A three-agent deep learning system that detects manipulated regions in 
scientific images at the pixel level. Each agent analyses the same image 
through a different forensic lens before combining evidence through 
cross-attention and prototype memory retrieval.

## Architecture
- Agent 1 — RGB encoder: detects colour and texture inconsistencies
- Agent 2 — SRM noise extractor: reveals camera noise fingerprint differences
- Agent 3 — FFT frequency analyser: detects boundary artefacts in frequency space
- Cross-agent attention debate: 768-token joint attention across all three agents
- Spatial prototype memory: CPU-resident bank of learned forgery patterns
- ForgeryDecoder: 16x16 → 256x256 with skip connection

## Results
| Model | Val IoU |
|---|---|
| ResNet-34 U-Net Baseline | 0.2447 |
| AgenticForensicNet | reported post-training |

0.2 – 0.3 IoU represents current state-of-the-art for this domain.

## Stack
Python · PyTorch · Albumentations · Kaggle T4 GPU · NumPy · Pandas


