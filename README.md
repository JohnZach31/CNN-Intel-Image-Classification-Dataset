# 🧠 Intel Image Classification using CNN (PyTorch)

## Overview

This project implements a Convolutional Neural Network (CNN) for multi-class image classification using the **Intel Image Classification dataset**. The model is built with **PyTorch** and trained to classify natural scenes into six categories.

The pipeline covers:

* Data acquisition (Kaggle API)
* Preprocessing & augmentation
* Model design
* Training with validation
* Performance visualization

---

## 📂 Dataset

* Source: [Intel Image Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

* Classes:

  * Buildings
  * Forest
  * Glacier
  * Mountain
  * Sea
  * Street

* Dataset Split:

  * ~14,000 training images
  * ~3,000 test images
  * 15% of training set used for validation

---

## ⚙️ Tech Stack

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Kaggle API

---

## 🧪 Data Preprocessing

### Transformations

* Resize to `150x150`
* Random horizontal flip (training only)
* Normalization:

  ```
  mean = (0.5, 0.5, 0.5)
  std  = (0.5, 0.5, 0.5)
  ```

### Data Loaders

* Batch size: `32`
* Shuffling enabled for training
* Validation and test sets are not shuffled

---

## 🏗️ Model Architecture

Custom CNN with 3 convolutional blocks:

```
Input: 3 x 150 x 150

Conv(3 → 32) → ReLU → MaxPool
Conv(32 → 64) → ReLU → MaxPool
Conv(64 → 128) → ReLU → MaxPool

Flatten
Linear(128 * 18 * 18 → 256) → ReLU → Dropout(0.5)
Linear(256 → 6)
```

---

## 🧠 Training Configuration

* Loss Function: CrossEntropyLoss
* Optimizer: Adam
* Learning Rate: `1e-3`
* Epochs: `12`
* Scheduler:

  * ReduceLROnPlateau
  * Factor: `0.5`
  * Patience: `2`

---

## 📊 Results

### Training Performance

* Final Training Accuracy: ~94%
* Final Validation Accuracy: ~86–87%

### Observations

* Strong convergence within first few epochs
* Learning rate scheduler improves stability
* Slight overfitting observed (train > validation accuracy)

---

## 📈 Visualizations

The project includes:

* Loss vs Epochs
* Accuracy vs Epochs

These help analyze:

* Convergence behavior
* Overfitting trends
* Generalization performance

---

## 🚀 How to Run

### 1. Setup Kaggle API

```bash
pip install kaggle
```

Place your `kaggle.json` in:

```
~/.kaggle/kaggle.json
```

### 2. Download Dataset

```bash
kaggle datasets download -d puneet6060/intel-image-classification
```

### 3. Extract Dataset

```bash
unzip intel-image-classification.zip
```

### 4. Run Training

Execute the notebook or script:

```bash
python train.py
```

*(or run in Google Colab)*

---

## 📌 Future Improvements

* Data augmentation (rotation, color jitter, etc.)
* Transfer learning (ResNet, EfficientNet)
* Hyperparameter tuning
* Early stopping
* Confusion matrix & per-class metrics

---

## 📎 Notes

* Designed to run efficiently on GPU (Colab T4 supported)
* Training time: ~5 minutes for 12 epochs (Colab GPU)

---

## 📄 License

Dataset license belongs to original authors (Kaggle dataset).
Code is free to use for educational and research purposes.
