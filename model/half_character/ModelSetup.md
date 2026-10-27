# Half Character Model Setup Guide

Guide for preparing the dataset and training the specialized Devanagari **Half Character (अर्धाक्षर)** recognition model.

---

## 1. Dataset Generation

Since standard datasets only contain full characters, we generate authentic half-character glyphs using OpenType text shaping (`HarfBuzz` + `FreeType`) across 16 diverse Devanagari fonts (calligraphy, handwriting, serif, sans-serif).

### Step 1: Download Fonts

```bash
python model/half_character/download_fonts.py
```

- Fetches Google Fonts (`Kalam`, `Dekko`, `NotoSans`, `RozhaOne`, etc.) and system fonts into `model/half_character/fonts/`.

### Step 2: Generate Augmented Dataset

```bash
python model/half_character/01_generate_dataset.py
```

- Shapes true half-glyphs (stripping the vertical _pai_ stem).
- Applies pipeline-matching handwriting augmentations (pen thickness dilation/erosion, rotation $\pm 8^\circ$, slant, and top shirorekha-cut clipping).
- Saves:
  - `model/half_character/dataset/train` (160 samples per class)
  - `model/half_character/dataset/test` (40 samples per class)

---

## 2. Training the Model

```bash
python model/half_character/03_train_half_model.py
```

- Model: **ResNet-18** (37 output classes).
- Image Size: 224×224 (normalized).
- Weights output: `model/half_character/half_resnet18.pth`.
