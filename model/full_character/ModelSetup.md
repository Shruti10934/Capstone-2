# Model Setup Guide

Guide for preparing dataset and training Devanagari character recognition model.

## Automatic Setup

### 1. Download Dataset:

- Download from Kaggle: [Devanagari Handwritten Character Dataset](https://www.kaggle.com/datasets/medahmedkrichen/devanagari-handwritten-character-datase)
- Save the ZIP file (`archive.zip`) into the `model/full_character/dataset` folder.

### 2. Run Auto Setup:

```bash
python model/full_character/setup.py
```

- This script will extract, rename, preprocess, and clean up automatically.
- It will also prompt you if you want to start training immediately.

---

## Manual Setup

### 1. Get Dataset

- Download from Kaggle: [Devanagari Handwritten Character Dataset](https://www.kaggle.com/datasets/medahmedkrichen/devanagari-handwritten-character-datase)
- Save ZIP in `model/full_character/dataset` folder.

### 2. Prepare Folders

- Extract ZIP in `model/full_character/dataset`.
- Rename folders:
    - `train` → `training`
    - `test` → `testing`

### 3. Preprocess Data

- Run the preprocessing script:
    ```bash
    python model/full_character/01-prepare-classes.py
    ```
- This script cleans images (shirorekha removal) and automatically creates `train` and `test` folders inside `model/full_character/dataset` from your `training` and `testing` sources.

### 4. Cleanup

- Delete raw folders `model/full_character/dataset/training` and `model/full_character/dataset/testing`.

### 5. Train Model

- **Local**:
    ```bash
    python model/full_character/02-train-model.py
    ```
- **Google Colab**:
    - Zip the `model/full_character/dataset` folder.
    - Upload to Google Drive / Capstone2.
    - Run [Google Colab Notebook](https://colab.research.google.com/drive/1jA5Dw27gYUvwUcF0EjvPbY5d5Xz6Qbl6).
    - Download `resnet18.pth` into `model/full_character/`.

### 6. Verify Model Weights

- Ensure `resnet18.pth` is saved directly inside `model/full_character/`.
- Ensure filename is `resnet18.pth` for classifier and notebook compatibility.
