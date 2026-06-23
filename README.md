# ASL Hand-Sign Classification

An end-to-end machine learning pipeline that takes raw American Sign Language hand-sign images and turns them into a working real-time classifier. Built entirely in Python on Google Colab using MediaPipe for feature extraction and scikit-learn for modeling.

---

## What this project does

The idea was simple: instead of feeding raw pixels into a deep learning model (which would need a lot of compute and data), we extract compact hand-landmark features using Google MediaPipe and train classical ML classifiers on top of those. It's fast, interpretable, and surprisingly accurate.

The full pipeline covers:
- Downloading and auditing the dataset
- Extracting 63-dimensional feature vectors from hand images
- Handling class imbalance with SMOTE
- Training and tuning KNN, SVM, and Random Forest
- Explaining predictions with SHAP
- Exporting everything needed for real-time inference

---

## Dataset

**Kaggle ASL Alphabet** — [grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

```
Total images  : 87,000
Image size    : 200 × 200 px (JPEG)
Total classes : 29
Images/class  : 3,000 (perfectly balanced)

Classes: A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
         + del  + nothing  + space
```

The dataset is already perfectly balanced — every class has exactly 3,000 images — so SMOTE is applied after feature extraction to handle any imbalance introduced during the train/test split.

---

## Class Distribution

Each class has exactly 3,000 images:

```
Class counts (per class = 3,000 images)

A   ████████████████████  3000
B   ████████████████████  3000
C   ████████████████████  3000
D   ████████████████████  3000
E   ████████████████████  3000
F   ████████████████████  3000
G   ████████████████████  3000
H   ████████████████████  3000
I   ████████████████████  3000
J   ████████████████████  3000
K   ████████████████████  3000
L   ████████████████████  3000
M   ████████████████████  3000
N   ████████████████████  3000
O   ████████████████████  3000
P   ████████████████████  3000
Q   ████████████████████  3000
R   ████████████████████  3000
S   ████████████████████  3000
T   ████████████████████  3000
U   ████████████████████  3000
V   ████████████████████  3000
W   ████████████████████  3000
X   ████████████████████  3000
Y   ████████████████████  3000
Z   ████████████████████  3000
del ████████████████████  3000
nth ████████████████████  3000  (nothing)
spc ████████████████████  3000  (space)
```

---

## Feature Extraction

Instead of raw pixels, we use **Google MediaPipe** to detect 21 hand landmarks per image. Each landmark gives us (x, y, z) coordinates, which results in a **63-dimensional feature vector** per image.

```
Raw Image (200×200 px)
        │
        ▼
  ┌─────────────┐
  │  MediaPipe  │   ← Hand Landmark Detection
  │  HandLandmarker │
  └──────┬──────┘
         │
         ▼
  21 landmarks × (x, y, z)
         │
         ▼
  63-dimensional feature vector
         │
         ▼
  StandardScaler → normalized features
         │
         ▼
  ML Classifier (KNN / SVM / RF)
```

This approach is much more compute-efficient than CNN-based methods and produces models that are easy to inspect and explain.

---

## Models & Tuning

Three classifiers were trained and compared:

| Model         | Tuning Strategy                        | Notes                        |
|---------------|----------------------------------------|------------------------------|
| KNN           | GridSearchCV — k, distance metric      | Simple baseline               |
| SVM           | GridSearchCV — C, kernel (RBF, Poly)   | Best generalization           |
| Random Forest | GridSearchCV — n_estimators, max_depth | 300 trees, SHAP-compatible    |

All tuning was done with **5-fold stratified cross-validation** to avoid leaking test data into model selection.

---

## Model Accuracy Comparison

```
Test Accuracy (%)

Random Forest  ████████████████████████████████████████  ~97%
SVM (RBF)      ███████████████████████████████████████▌  ~96%
SVM (Poly)     ██████████████████████████████████████▌   ~94%
KNN            █████████████████████████████████████     ~91%
               0%                                   100%
```

SVM with RBF kernel and Random Forest were the top performers, both crossing the 95% mark on the held-out test set.

---

## Cross-Validation Scores

```
5-Fold CV Mean Accuracy

      100% ┤
       97% ┤   ●───────────────────────────────● RF
       95% ┤       ●─────────────────────● SVM
       92% ┤
       90% ┤           ●─────────● KNN
       88% ┤
           └─────────────────────────────────────
            Fold 1   Fold 2   Fold 3   Fold 4   Fold 5
```

Results were stable across all folds, meaning the models generalize well and aren't just fitting noise.

---

## SHAP Feature Importance

SHAP (SHapley Additive exPlanations) was used on the Random Forest model to figure out which hand landmarks matter most for classification.

```
Top landmarks by mean |SHAP value|

Wrist (z)          ████████████████████  0.42
Index fingertip    ████████████████      0.34
Thumb tip          ██████████████        0.29
Middle fingertip   ████████████          0.25
Ring fingertip     ██████████            0.21
Pinky fingertip    █████████             0.19
Index MCP (y)      ████████              0.17
Middle PIP         ███████               0.14
Thumb IP           ██████                0.12
Wrist (x)          █████                 0.10
```

Fingertip positions and wrist depth (z-axis) are the strongest predictors — which makes intuitive sense since ASL signs are primarily differentiated by finger positions.

---

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────┐
│                    Full ML Pipeline                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. Data Download     Kaggle API → 87,000 images         │
│         ↓                                                │
│  2. EDA               Class counts, sample grid          │
│         ↓                                                │
│  3. Feature Extraction  MediaPipe → 63-dim vectors       │
│         ↓                                                │
│  4. Preprocessing     StandardScaler + LabelEncoder      │
│         ↓                                                │
│  5. Class Balancing   SMOTE oversampling                 │
│         ↓                                                │
│  6. Model Training    KNN, SVM, Random Forest            │
│         ↓                                                │
│  7. Hyperparameter Tuning  GridSearchCV (5-fold CV)      │
│         ↓                                                │
│  8. Evaluation        Accuracy, confusion matrix, report │
│         ↓                                                │
│  9. Explainability    SHAP TreeExplainer                 │
│         ↓                                                │
│  10. Export           .pkl models + scaler + encoder     │
│                       + metrics.json                     │
└──────────────────────────────────────────────────────────┘
```

---

## Tech Stack

- **Python 3.10**
- **Google Colab** (T4 GPU runtime)
- **MediaPipe** — hand landmark detection
- **scikit-learn** — KNN, SVM, Random Forest, GridSearchCV, metrics
- **imbalanced-learn** — SMOTE
- **SHAP** — model explainability
- **OpenCV** — image loading and color conversion
- **Matplotlib / Seaborn** — visualization
- **Kaggle API** — dataset download

---

## How to Run

1. Open `ASL_Sign_Language_Classification_Enhanced.ipynb` in Google Colab
2. Set the runtime to **GPU (T4)**
3. Upload your `kaggle.json` API key when prompted
4. Run all cells top to bottom — each section is well commented

The notebook handles everything: package installation, dataset download, feature extraction, training, evaluation, and export.

---

## Outputs

After the full pipeline runs, you'll have:

```
outputs/
├── knn_model.pkl          ← trained KNN
├── svm_model.pkl          ← trained SVM (best kernel)
├── rf_model.pkl           ← trained Random Forest
├── scaler.pkl             ← fitted StandardScaler
├── label_encoder.pkl      ← fitted LabelEncoder
└── metrics.json           ← accuracy, CV scores, classification report
```

These can be loaded directly for real-time inference — no retraining needed.

---

## Results Summary

| Model         | Test Accuracy | CV Mean (5-fold) |
|---------------|--------------|------------------|
| Random Forest | ~97%         | ~96.8%           |
| SVM (RBF)     | ~96%         | ~95.9%           |
| SVM (Poly)    | ~94%         | ~93.4%           |
| KNN           | ~91%         | ~90.2%           |

Random Forest edges out SVM slightly on the test set and has the added benefit of native SHAP support for per-prediction explanations.

---

## Files in This Repo

| File | Description |
|------|-------------|
| `ASL_Sign_Language_Classification_Enhanced.ipynb` | Main notebook — full pipeline |
| `ASL Hand-Sign Classification_ An End-to-End Machine Learning Pipeline.pdf` | Presentation slides |
| `ASL_Classification_Technical_Report_Final.docx` | Full technical report |

---

## Author

Built as part of an end-to-end ML project showcasing classical machine learning on a real-world computer vision task — using smart feature engineering to avoid the need for deep learning.
