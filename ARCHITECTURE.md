# Architecture Technique — Reconnaissance de Logos

## Vue d'ensemble

Le système repose sur un pipeline classique de vision par ordinateur en 4 étapes principales :

```
Image Brute → Prétraitement → Extraction SIFT → Encodage BoW → Classification → Résultat
```

---

## 1. Prétraitement

Chaque image subit les transformations suivantes avant toute analyse :

- **Redimensionnement** à 256×256 pixels
- **Conversion en niveaux de gris** (BGR → GRAY via OpenCV)
- **Égalisation d'histogramme** (`cv2.equalizeHist`) pour améliorer le contraste

---

## 2. Extraction des Caractéristiques — SIFT

**SIFT** (Scale-Invariant Feature Transform) détecte automatiquement des points clés dans l'image et calcule pour chacun un **descripteur de 128 dimensions** basé sur les gradients locaux.

**Avantages :**
- Robuste aux rotations et changements d'échelle
- Robuste aux variations d'éclairage
- Invariant aux transformations affines

```python
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(image_gray, None)
```

---

## 3. Dictionnaire Visuel — Bag of Visual Words

### 3.1 Clustering K-Means

L'ensemble des descripteurs SIFT extraits des images d'entraînement est regroupé en **K clusters** (défaut : K=2000) via `MiniBatchKMeans` de scikit-learn.

```python
kmeans = MiniBatchKMeans(n_clusters=K, batch_size=5000)
kmeans.fit(all_descriptors)
```

### 3.2 Encodage BoW

Chaque image est représentée comme un **histogramme de fréquences** des K mots visuels, normalisé en L2.

```
Image → descripteurs SIFT → assignation aux clusters → histogramme (K dimensions) → normalisation L2
```

---

## 4. Classification

### 4.1 KNN (K-Nearest Neighbors)
- **Métrique** : cosinus
- **Prédiction** : classe majoritaire parmi les K voisins les plus proches
- **Confiance** : proportion de voisins appartenant à la classe prédite

### 4.2 SVM (Support Vector Machine)
- **Kernel** : RBF (Radial Basis Function)
- **Stratégie multi-classes** : One-vs-One
- **Confiance** : softmax appliqué sur la `decision_function`

---

## 5. Performances

| Modèle | Accuracy |
|--------|----------|
| KNN    | 40.0 %   |
| SVM    | 42.0 %   |

---

## 6. Stack Technique

| Composant     | Technologie              |
|---------------|--------------------------|
| Backend       | Flask (Python)           |
| Vision        | OpenCV (cv2)             |
| ML            | scikit-learn             |
| Calcul        | NumPy                    |
| Frontend      | HTML / CSS / JavaScript  |
| Visualisation | Chart.js                 |
| Modèles       | pickle (.pkl)            |

---

## 7. Structure des Fichiers

```
app/
├── app.py               # Point d'entrée Flask + logique ML
├── models/
│   ├── kmeans.pkl       # Dictionnaire visuel KNN
│   ├── kmeans_svm.pkl   # Dictionnaire visuel SVM
│   ├── knn.pkl          # Classificateur KNN
│   └── svm.pkl          # Classificateur SVM
└── templates/
    └── index.html       # Interface web
```

---

## 8. API Endpoints

| Méthode | Endpoint    | Description                        |
|---------|-------------|------------------------------------|
| GET     | `/`         | Page principale                    |
| POST    | `/train`    | Lancer l'entraînement              |
| GET     | `/progress` | Progression de l'entraînement      |
| POST    | `/predict`  | Prédiction sur une image uploadée  |
| GET     | `/status`   | État des modèles chargés           |

---

*Projet réalisé par Mehdi GHINE, Youssef B'SIBISS et Anass BELLAGRID — encadré par Prof. Abdelillah Semmaa*
