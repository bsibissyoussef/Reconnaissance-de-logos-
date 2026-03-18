# 🔍 Logo Recognition — SIFT + Bag of Visual Words + KNN

> **Computer Vision Project** · Grp.10  
> Mehdi Ghine · Youssef B'sibiss · Anass Bellagrid

---

## 📌 Overview

This project tackles **brand logo recognition in advertising images** using classical computer vision techniques. Given an input image containing a logo, the system identifies the brand it belongs to — without any deep learning.

The pipeline follows the **Bag of Visual Words (BoVW)** paradigm:
1. Extract local keypoint descriptors with **SIFT**
2. Build a **visual vocabulary** via K-Means clustering
3. Encode each image as a **BoW histogram**
4. Classify with **K-Nearest Neighbors (KNN)**

---

## 🗂️ Dataset

**[Logo-2K+](https://github.com/msn199959/Logo-2K-plus-Dataset)**  
A large-scale logo dataset containing **2,000+ brand logos** organized into hierarchical categories.

| Property | Details |
|---|---|
| # of classes | 2,341 brand categories |
| # of images | ~167,000 images |
| Format | JPG / PNG |
| Organization | Category folders |

---

## 🧠 Methods

### 1. Feature Extraction — SIFT
**Scale-Invariant Feature Transform** detects and describes local keypoints that are robust to scale, rotation, and illumination changes.

```python
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(image, None)
```

### 2. Visual Dictionary — K-Means
All SIFT descriptors from the training set are clustered into `K` visual words using **K-Means**. Each cluster center becomes a "visual word".

```python
from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(n_clusters=K, random_state=42)
kmeans.fit(all_descriptors)
```

### 3. Bag of Visual Words Encoding
Each image is represented as a **histogram** of visual word frequencies — a fixed-size feature vector regardless of the number of keypoints.

```python
def compute_bow(descriptors, kmeans, K):
    histogram = np.zeros(K)
    labels = kmeans.predict(descriptors)
    for label in labels:
        histogram[label] += 1
    return histogram / histogram.sum()  # L1 normalization
```

### 4. Classification — KNN
The encoded BoW vectors are classified using **K-Nearest Neighbors**, comparing a query image's histogram against all training histograms.

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
```

---

## 📁 Project Structure

```
logo-recognition-bovw/
│
├── data/
│   └── Logo-2K+/               # Dataset (not included in repo)
│       ├── brand_A/
│       ├── brand_B/
│       └── ...
│
├── src/
│   ├── extract_features.py     # SIFT descriptor extraction
│   ├── build_vocabulary.py     # K-Means visual dictionary
│   ├── encode_bow.py           # BoW histogram encoding
│   ├── train_knn.py            # KNN classifier training
│   └── predict.py              # Inference on new images
│
├── models/
│   ├── kmeans_vocab.pkl        # Saved K-Means model
│   └── knn_classifier.pkl      # Saved KNN model
│
├── notebooks/
│   └── exploration.ipynb       # EDA & pipeline walkthrough
│
├── results/
│   └── evaluation_report.txt   # Accuracy, confusion matrix
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/logo-recognition-bovw.git
cd logo-recognition-bovw

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`
```
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
Pillow>=10.0.0
joblib>=1.3.0
```

---

## 🚀 Usage

### Step 1 — Extract SIFT features
```bash
python src/extract_features.py --data_dir data/Logo-2K+ --output features/
```

### Step 2 — Build the visual vocabulary
```bash
python src/build_vocabulary.py --features_dir features/ --k 500 --output models/kmeans_vocab.pkl
```

### Step 3 — Encode images as BoW vectors
```bash
python src/encode_bow.py --features_dir features/ --vocab models/kmeans_vocab.pkl --output bow_vectors/
```

### Step 4 — Train the KNN classifier
```bash
python src/train_knn.py --bow_dir bow_vectors/ --k_neighbors 5 --output models/knn_classifier.pkl
```

### Step 5 — Predict on a new image
```bash
python src/predict.py --image path/to/logo.jpg \
                      --vocab models/kmeans_vocab.pkl \
                      --model models/knn_classifier.pkl
```

---

## 📊 Pipeline Summary

```
Input Image
    │
    ▼
[SIFT Keypoint Detection & Description]
    │  → N × 128 descriptor matrix
    ▼
[K-Means Visual Vocabulary (K visual words)]
    │  → Assign each descriptor to nearest cluster
    ▼
[Bag of Visual Words Histogram]
    │  → 1 × K normalized frequency vector
    ▼
[KNN Classification]
    │  → Top-K most similar training images
    ▼
Predicted Brand Label
```

---

## 📈 Hyperparameters

| Parameter | Description | Default |
|---|---|---|
| `K` (vocabulary size) | Number of visual words (K-Means clusters) | `500` |
| `k` (KNN neighbors) | Number of nearest neighbors | `5` |
| SIFT `nfeatures` | Max keypoints per image | `200` |
| KNN distance metric | Similarity measure | `euclidean` |

> **Tip:** Larger vocabularies (K = 500–1000) generally yield better discrimination. Experiment with `cosine` distance in KNN for normalized histograms.

---

## 🧪 Evaluation

Performance is measured on a held-out test split using:
- **Top-1 Accuracy**
- **Top-5 Accuracy**
- **Confusion Matrix**

```bash
python src/evaluate.py --bow_dir bow_vectors/ --model models/knn_classifier.pkl
```

---

## 🔬 Discussion & Limitations

| Aspect | Note |
|---|---|
| **Scalability** | KNN becomes slow at inference for large datasets; consider approximate NN (FAISS) |
| **Vocabulary size** | Small K → underfitting; Large K → overfitting / slow training |
| **SIFT limitations** | May struggle with heavily stylized or text-based logos |
| **Possible improvements** | TF-IDF weighting of BoW, spatial pyramid matching (SPM), SVM classifier |

---

## 👥 Authors

| Name | GitHub |
|---|---|
| Mehdi Ghine | [@m3hdiix-h4x](https://github.com/m3hdiix-h4x) |
| Youssef B'sibiss | [@bsibissyoussef](https://github.com/bsibissyoussef) |
| Anass Bellagrid | [@wh0s-ans](https://github.com/wh0s-ans) |

---

## 📄 License

This project is developed for academic purposes as part of a Computer Vision course.  
The Logo-2K+ dataset is subject to its own license — refer to the [original repository](https://github.com/msn199959/Logo-2K-plus-Dataset).

---
