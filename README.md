# ASL Sign Language Classification

**Student:** Abrham Assefa Habtamu — MAT. VR548223  
**University:** University of Verona

An end-to-end machine learning pipeline for classifying American Sign Language hand signs. Instead of feeding raw pixels into a deep network, we use Google MediaPipe to extract compact hand-landmark features and train classical ML classifiers on top — fast, interpretable, and highly accurate.

---

## What This Project Does

The pipeline takes raw ASL hand-sign images, extracts 63-dimensional feature vectors using MediaPipe hand-landmark detection, trains three classifiers, explains predictions with SHAP, and exports everything needed for real-time inference.

---

## Dataset

**Kaggle ASL Alphabet** — [grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

| Property | Value |
|---|---|
| Total images | 87,000 |
| Image size | 200 × 200 px (JPEG) |
| Classes | 29 |
| Images per class | 3,000 (perfectly balanced) |
| Classes | A–Z + `del` + `nothing` + `space` |

---

## Class Distribution

```
Images per class (each = 3,000)

A   ████████████████████  3,000
B   ████████████████████  3,000
C   ████████████████████  3,000
...
Z   ████████████████████  3,000
del ████████████████████  3,000
spc ████████████████████  3,000

All 29 classes perfectly balanced.
```

---

## Pipeline Overview

```
Raw Images (200×200 px)
        │
        ▼
  ┌─────────────────────┐
  │  MediaPipe          │  Hand Landmark Detection
  │  HandLandmarker     │  min_confidence = 0.3
  └──────────┬──────────┘
             │
             ▼
   21 landmarks × (x, y, z)
             │
      Normalisation:
      coords -= coords[0]          ← translate wrist to origin
      coords /= max(|coords|) + ε  ← scale-invariant
             │
             ▼
      63-dimensional feature vector
             │
             ▼
      StandardScaler → SMOTE → Classifier
```

This approach is far more compute-efficient than CNN-based methods. MediaPipe pre-solves the hard part — localisation, pose normalisation, background removal — so the classifier only sees 63 clean, normalised numbers.

---

## Feature Extraction Results

```
Total attempted   : 5,800 images  (200 per class × 29 classes)
Successful        : 5,030  (86.7%)
Failed detections :   770  (13.3%)

Feature matrix shape : (5,030 × 63)
Unique classes       : 28  (nothing class had 0 detections)
```

---

## Preprocessing

| Step | Tool | Why |
|---|---|---|
| Label encoding | `LabelEncoder` | Models need integer targets |
| Train/test split | `train_test_split` | Stratified 80/20 |
| Feature scaling | `StandardScaler` | KNN & SVM are distance-based |
| Class balancing | `SMOTE` | Synthetic samples for under-represented classes |

```
Train (raw)       : 4,024 samples
Train (balanced)  : 4,480 samples  ← SMOTE added synthetic samples
Test              : 1,006 samples
```

---

## PCA Analysis

```
Principal Components needed to explain 95% variance: 7  (out of 63)

PC1: 54.1%  ████████████████████████████████████████████████████
PC2: 22.1%  ██████████████████████████
PC3–PC7: remaining ~18.8%

Total variance explained by PC1 + PC2: 76.2%
```

Only 7 components capture 95% of the information — the landmark features are highly structured and redundant, which is why classical ML works so well here.

---

## Models & Hyperparameter Tuning

Three classifiers trained with GridSearchCV (5-fold stratified CV):

| Model | Search Space | Best Params |
|---|---|---|
| KNN | k ∈ {3,5,7,11}, weights, metric | k=3, euclidean, distance weights |
| SVM | C ∈ {1,10,100}, gamma, kernel | C=100, RBF, scale |
| Random Forest | 300 trees, max_depth | default |

---

## Results

### Test Accuracy

```
SVM (RBF, C=100)  ██████████████████████████████████████████  99.50%
KNN (k=3)         █████████████████████████████████████████   98.91%
Random Forest     █████████████████████████████████████████   98.91%
                  0%                                    100%
```

**Best model: SVM** with test accuracy of **99.50%**

---

### 5-Fold Cross-Validation — SVM

```
Fold 1  ████████████████████████████████████████  99.22%
Fold 2  █████████████████████████████████████████ 99.55%
Fold 3  █████████████████████████████████████████ 99.67%
Fold 4  ████████████████████████████████████████  99.00%
Fold 5  ██████████████████████████████████████████99.89%

Mean: 99.46%  |  Std: ±0.32%
```

Stable across all folds — the model generalises reliably, not just getting lucky on one split.

---

### Per-Class Performance — SVM (selected classes)

```
Class   Precision  Recall   F1     Support
─────────────────────────────────────────────
A         0.97      1.00    0.99      37
B         1.00      1.00    1.00      36
C         1.00      0.97    0.99      39
D         1.00      1.00    1.00      39
...
W         0.96      1.00    0.98      24
space     0.95      1.00    0.97      36
─────────────────────────────────────────────
Accuracy                    1.00    1,006
Macro avg 0.99      1.00    0.99    1,006
```

---

### Learning Curve

```
Accuracy
  1.00 ┤────────────────────────────────── Train (1.0000)
  0.99 ┤  ·  ·  · ·──────────────────────  CV    (0.9946)
  0.98 ┤
  0.97 ┤
       └──────────────────────────────────────────
        Small     Medium     Large   (training size)

Final train-CV gap: 0.0054  → No overfitting
```

Both curves converge to high accuracy with a tiny gap — the model fits well without memorising noise.

---

## SHAP Feature Importance

Top landmarks by mean |SHAP value| (Random Forest, TreeExplainer):

```
lm04_x  (Thumb tip, x)          ████████████████████  0.00804
lm12_y  (Middle fingertip, y)   ████████████████      0.00618
lm20_y  (Pinky tip, y)          █████████████         0.00533
lm16_y  (Ring fingertip, y)     ████████████          0.00486
lm15_y  (Ring finger PIP, y)    ███████████           0.00440
```

Fingertip positions — especially the thumb tip x-coordinate and fingertip y-positions — are the strongest predictors. This makes intuitive sense since ASL signs are primarily differentiated by finger positions and configurations.

---

## Inference

The saved pipeline can predict any hand image without retraining:

```
image file
    → extract_landmarks()      # MediaPipe → 63-float vector
    → scaler.transform()       # Same StandardScaler from training
    → model.predict()          # Integer class index
    → le.inverse_transform()   # Integer → 'A' / 'B' / ... / 'Z'
    → model.predict_proba()    # Confidence + top-3 alternatives
```

Example outputs:
```
del  → del   (confidence: 96.2%)   ✓
Y    → Y     (confidence: 85.2%)   ✓
T    → T     (confidence: 91.3%)   ✓
```

---

## Saved Artifacts

| File | Size | Description |
|---|---|---|
| `asl_best_model.pkl` | 766.8 KB | Best classifier (SVM) |
| `asl_svm_model.pkl` | 766.8 KB | Tuned SVM |
| `asl_knn_model.pkl` | 1,138.5 KB | Tuned KNN |
| `asl_rf_model.pkl` | 25,391.9 KB | Random Forest (for SHAP) |
| `scaler.pkl` | 2.1 KB | Fitted StandardScaler |
| `label_encoder.pkl` | 0.9 KB | Fitted LabelEncoder |
| `metrics_summary.json` | 0.5 KB | Accuracy + metadata |

> Important: always apply `scaler.transform()` before inference — forgetting this is the most common bug.

---

## Limitations

The ~99.5% accuracy should be read carefully:

- **MediaPipe pre-solves the hard part** — the classifier sees 63 clean numbers, not raw pixels with lighting/background variation.
- **One data source** — all images come from a single Kaggle dataset with consistent studio lighting. The model has not been tested on different cameras, lighting, skin tones, or signer styles.
- **200 images/class** — only ~17% of available data was used (compute trade-off for Colab). Some classes had very low test support due to detection failures.
- **Static signs only** — no motion, no occlusion, single hand, single signer style per class.

---

## Tech Stack

- Python 3.10 · Google Colab (T4 GPU)
- MediaPipe · scikit-learn · imbalanced-learn
- SHAP · OpenCV · Matplotlib / Seaborn
- Kaggle API

---

## How to Run

1. Open the notebook in Google Colab with a T4 GPU runtime
2. Upload your `kaggle.json` API key when prompted
3. Run all cells top to bottom — takes roughly one session
