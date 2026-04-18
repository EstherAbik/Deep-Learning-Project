# Deep-Learning-Project

# Predicting Object Categories from Limited Labelled Data: Comparing MLP, CNN, and Transfer Learning on STL-10

## What This Project Does

This project answers three questions about deep learning for image classification, using the **STL-10** dataset and **PyTorch**:

1. **Does network architecture matter?** We compare a Multi-Layer Perceptron (MLP) against a Convolutional Neural Network (CNN) on the same images. The MLP has 72× more parameters but the CNN wins — because convolutions encode the right assumptions about how images work.

2. **Does regularisation help when data is limited?** We train each model twice — once without regularisation, once with Dropout and BatchNorm — and compare the overfitting behaviour. With only 5,000 training images, there is improvement with regularization.

3. **Can pretrained models rescue a small dataset?** We fine-tune AlexNet and ResNet18 (pretrained on 1.2 million ImageNet photos) on our 5,000 STL-10 images, then combine them into an **ensemble** for an additional accuracy boost.

---

## The Dataset: STL-10

STL-10 is a benchmark from Stanford containing 96×96 colour photographs across 10 object classes (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck) with only **5,000 labelled training images** and 8,000 test images. Perfectly balanced at 500/800 images per class.

We chose STL-10 because its small training set makes overfitting visible, transfer learning's advantage dramatic, and the MLP→CNN→pretrained progression clear. It contains natural object photos (same domain as ImageNet), so pretrained features transfer directly — unlike digit datasets where they fail.

---

## The Seven Models

| # | Model | Parameters | What it demonstrates |
|---|-------|-----------|---------------------|
| 1 | MLP (no reg) | 14.3M | Baseline — overfits severely with 2,860 params per training image |
| 2 | MLP + Dropout | 14.3M | Dropout(0.2) reduces train-test gap from 35% to 12% |
| 3 | CNN (no reg) | 198K | 72× fewer params than MLP, much better accuracy |
| 4 | CNN + BatchNorm/Dropout | 199K | BatchNorm accelerates convergence, Dropout prevents co-adaptation |
| 5 | AlexNet (pretrained) | 57M (54.6M trainable) | ImageNet features, classifier fine-tuned |
| 6 | ResNet18 (pretrained) | 11M (5K trainable) | ImageNet features, only `.fc` head replaced |
| 7 | **Ensemble** | — | Averages AlexNet + ResNet18 probabilities for best accuracy |

---

## Key Concepts Explained

### Test loss vs test accuracy

**Test accuracy** counts correct predictions. **Test loss** (cross-entropy) measures prediction confidence. A model can maintain 45% accuracy while its test loss *rises* — meaning predictions are becoming less confident. This is an early warning of overfitting that accuracy alone misses. Our no-reg MLP demonstrates this: accuracy plateaus at 45% but test loss climbs from 1.83 to 1.95.

### Why the ensemble works

AlexNet uses large 11×11 filters; ResNet18 uses 3×3 filters with skip connections. They learn different features and make different mistakes. Averaging their softmax probabilities smooths out individual errors.

### Why AlexNet's classifier is unfrozen

AlexNet's classifier has Dropout(0.5) layers. When the Linear layers underneath are frozen, the Dropout corrupts the signal without the model being able to adapt. The entire classifier (keeping conv features frozen) is unfrozen to fix this. ResNet18 has no Dropout, so frozen-backbone works cleanly.

---

## How to Run

### Prerequisites

- Python 3.9+
- ~3 GB disk space (dataset + pretrained weights, downloaded automatically)
- GPU recommended (CPU takes ~2–3 hours; GPU takes ~20–45 minutes)

### Steps

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter Deep_Learning_Project.ipynb
# Then: Kernel → Restart & Run All
```

> **GPU:** Default pip installs CPU PyTorch. For NVIDIA CUDA, see [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). Apple Silicon MPS works automatically.

---

## Project Structure

```
├── README.md                 
├── requirements.txt           ← pip install -r requirements.txt
├── report.md                  ← project report
└── Deep_Learning_Project.ipynb     ← main analysis notebook (58 cells)
```

---

## Notebook Structure

| Part | What happens |
|------|-------------|
| 1 | Setup — imports, device selection |
| 2 | Load STL-10, visualise samples, class balance |
| 3 | Build MLP and CNN (no regularisation) |
| 4 | Training utilities (Adam + CosineAnnealingLR) |
| 5 | Train without regularisation (30 epochs) |
| 6 | Analyse overfitting — plot train-test gaps |
| 7 | Build regularised models (Dropout, BatchNorm) |
| 8 | Train with regularisation (30 epochs) |
| 9 | Compare before vs after regularisation |
| 10 | Transfer learning — adapt AlexNet + ResNet18 |
| 11 | Train pretrained models (10 epochs) |
| 12 | Final comparison — all six models |
| 12b | **Ensemble** — average AlexNet + ResNet18 predictions |
| 13 | Inspect correct and misclassified samples |
| 14 | Discussion and conclusions |

---

## Technical Decisions

| Decision | Value | Why |
|----------|-------|-----|
| Image size (scratch) | 96×96 native | No resizing needed |
| Image size (pretrained) | 128×128 (Resize 144 → Crop 128) | Worksheet 4 convention |
| Scratch normalisation | STL-10 stats | Dataset-specific |
| Pretrained normalisation | ImageNet stats | Required by backbone |
| Augmentation | RandomCrop(padding=4) + HorizontalFlip | Valid for natural objects |
| Scratch epochs | 30 | at least 20 epochs chosen |
| Pretrained epochs | 10 | Converges faster with pretrained features |
| AlexNet LR | 1e-4 | Protects pretrained classifier weights |
| CNN pooling | AdaptiveAvgPool2d(4,4) | Keeps classifier head compact |
| Ensemble | Average softmax probs |  combines diverse predictions |

### Data leakage prevention

Test set evaluated every epoch for monitoring only. No training decisions based on test metrics. Fixed epoch count. Final-epoch accuracy reported.

---

## API Keys

**None required.** Everything downloads automatically on first run.

---

## Estimated Runtime

| Device | All models total |
|--------|-----------------|
| Apple MPS | 30–45 min |
| NVIDIA GPU | 20–30 min |
| CPU | 2–3 hours |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: torch` | Activate venv, `pip install -r requirements.txt` |
| `AttributeError: TransformWrapper` | Ensure `num_workers=0` (already set) |
| MPS out of memory | Reduce `BATCH_SIZE` to 32 |
| Slow on CPU | Reduce `SCRATCH_EPOCHS` to 15 |

---

## Attribution

- **STL-10:** Coates, Ng, Lee (Stanford, 2011) — [cs.stanford.edu/~acoates/stl10](https://cs.stanford.edu/~acoates/stl10/)
- **Pretrained weights:** PyTorch torchvision — [pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)
- **Course materials:** Extends Worksheet 4 and Tutorials 5–7
