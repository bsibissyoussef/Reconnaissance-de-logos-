from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import pickle
import os
import threading
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_TRAIN = r"C:\Users\joz3ph\Documents\ESTSB\python\projet\datasetcopy\trainandtest\train"
DATASET_TEST  = r"C:\Users\joz3ph\Documents\ESTSB\python\projet\datasetcopy\trainandtest\test"

CLASS_NAMES = [
    'Accessories', 'Clothes', 'Cosmetic', 'Electronic',
    'Food', 'Institution', 'Leisure', 'Medical',
    'Necessities', 'Transportation'
]

sift = cv2.SIFT_create()

state = {
    'training': False,
    'progress': 0,
    'logs': [],
    'done': False,
    'error': None,
    # KNN
    'model_knn': None,
    'kmeans_knn': None,
    'K_knn': None,
    'accuracy_knn': None,
    'f1_knn': None,
    'cm_knn': None,
    # SVM
    'model_svm': None,
    'kmeans_svm': None,
    'K_svm': None,
    'accuracy_svm': None,
    'f1_svm': None,
    'cm_svm': None,
}

# ─────────────────────────────────────────
# Charger modèles sauvegardés au démarrage
# ─────────────────────────────────────────
try:
    state['model_knn']  = pickle.load(open(os.path.join(BASE_DIR, 'models', 'knn.pkl'), 'rb'))
    state['kmeans_knn'] = pickle.load(open(os.path.join(BASE_DIR, 'models', 'kmeans.pkl'), 'rb'))
    state['K_knn']      = 2000
    state['accuracy_knn'] = 40.0
    print("KNN chargé ✓")
except Exception as e:
    print(f"KNN non chargé : {e}")

try:
    state['model_svm']  = pickle.load(open(os.path.join(BASE_DIR, 'models', 'svm.pkl'), 'rb'))
    state['kmeans_svm'] = pickle.load(open(os.path.join(BASE_DIR, 'models', 'kmeans_svm.pkl'), 'rb'))
    state['K_svm']      = 2000
    state['accuracy_svm'] = 42.0
    print("SVM chargé ✓")
except Exception as e:
    print(f"SVM non chargé : {e}")

# ─────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────
def log(msg):
    state['logs'].append(msg)
    print(msg)

def encode_bow(descriptors, kmeans, K):
    histogram = np.zeros(K)
    if descriptors is not None:
        words = kmeans.predict(descriptors)
        for w in words:
            histogram[w] += 1
    return histogram

def get_img_paths(base_path, max_per_class):
    paths = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(base_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        class_imgs = [
            os.path.join(class_dir, subdir, img_name)
            for subdir in os.listdir(class_dir)
            if os.path.isdir(os.path.join(class_dir, subdir))
            for img_name in os.listdir(os.path.join(class_dir, subdir))
            if img_name.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        if max_per_class:
            class_imgs = class_imgs[:max_per_class]
        paths += class_imgs
    return paths

def extract_bow_features(img_paths, kmeans, K, base_path, label=""):
    X, y = [], []
    total = len(img_paths)
    for i, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (256, 256))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        _, descriptors = sift.detectAndCompute(gray, None)
        bow = encode_bow(descriptors, kmeans, K)
        X.append(bow)
        class_name = os.path.relpath(img_path, base_path).split(os.sep)[0]
        y.append(class_name)
        if (i + 1) % 50 == 0:
            log(f"{label} : {i+1}/{total} images traitées...")
    return normalize(np.array(X)), np.array(y)

def predict_single(img_bytes, model, kmeans, K, model_type):
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, descriptors = sift.detectAndCompute(gray, None)

    bow = encode_bow(descriptors, kmeans, K)
    bow = bow.reshape(1, -1)
    bow = bow / (np.linalg.norm(bow) + 1e-6)

    prediction = model.predict(bow)[0]

    if model_type == 'knn':
        probas = model.predict_proba(bow)[0]
        confidence = round(max(probas) * 100, 2)
        all_probas = {CLASS_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(probas)}
    else:
        decision = model.decision_function(bow)[0]
        exp_scores = np.exp(decision - np.max(decision))
        softmax = exp_scores / exp_scores.sum()
        confidence = round(float(max(softmax)) * 100, 2)
        all_probas = {CLASS_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(softmax)}

    return prediction, confidence, all_probas

# ─────────────────────────────────────────
# Entraînement
# ─────────────────────────────────────────
def train_pipeline(max_per_class, K, model_type):
    try:
        state['training'] = True
        state['done'] = False
        state['logs'] = []
        state['progress'] = 0
        state['error'] = None

        log("Collecte des images...")
        train_paths = get_img_paths(DATASET_TRAIN, max_per_class)
        test_paths  = get_img_paths(DATASET_TEST, max_per_class)
        log(f"Train : {len(train_paths)} | Test : {len(test_paths)}")
        state['progress'] = 10

        log("Extraction SIFT...")
        train_descriptors = []
        for i, img_path in enumerate(train_paths):
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (256, 256))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            _, descriptors = sift.detectAndCompute(gray, None)
            if descriptors is not None:
                train_descriptors.append(descriptors)
            if (i + 1) % 100 == 0:
                log(f"SIFT : {i+1}/{len(train_paths)}...")
        train_descriptors = np.vstack(train_descriptors)
        log(f"Descripteurs : {train_descriptors.shape}")
        state['progress'] = 30

        log(f"K-Means K={K}...")
        kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=5000, n_init=3)
        kmeans.fit(train_descriptors)
        log("Dictionnaire visuel ✓")
        state['progress'] = 50

        log("Encodage BoW Train...")
        X_train, y_train = extract_bow_features(train_paths, kmeans, K, DATASET_TRAIN, "Train")
        state['progress'] = 65

        log("Encodage BoW Test...")
        X_test, y_test = extract_bow_features(test_paths, kmeans, K, DATASET_TEST, "Test")
        state['progress'] = 80

        log(f"Entraînement {model_type.upper()}...")
        if model_type == 'knn':
            model = KNeighborsClassifier(n_neighbors=1, metric='cosine', n_jobs=-1)
        else:
            model = SVC(kernel='rbf', C=10, gamma='scale', decision_function_shape='ovr')
        model.fit(X_train, y_train)
        log(f"{model_type.upper()} entraîné ✓")
        state['progress'] = 90

        log("Évaluation...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=None, labels=CLASS_NAMES)
        cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES).tolist()
        log(f"Accuracy : {accuracy * 100:.2f}%")
        state['progress'] = 100

        if model_type == 'knn':
            state['model_knn']    = model
            state['kmeans_knn']   = kmeans
            state['K_knn']        = K
            state['accuracy_knn'] = round(accuracy * 100, 2)
            state['f1_knn']       = f1.tolist()
            state['cm_knn']       = cm
        else:
            state['model_svm']    = model
            state['kmeans_svm']   = kmeans
            state['K_svm']        = K
            state['accuracy_svm'] = round(accuracy * 100, 2)
            state['f1_svm']       = f1.tolist()
            state['cm_svm']       = cm

        state['done'] = True
        state['training'] = False
        log("Entraînement terminé !")

    except Exception as e:
        state['error'] = str(e)
        state['training'] = False
        log(f"Erreur : {str(e)}")

# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    if state['training']:
        return jsonify({'error': 'Entraînement déjà en cours'}), 400
    data = request.json
    max_per_class = data.get('max_per_class')
    K = int(data.get('K', 500))
    model_type = data.get('model_type', 'knn')
    max_per_class = int(max_per_class) if max_per_class else None
    t = threading.Thread(target=train_pipeline, args=(max_per_class, K, model_type))
    t.daemon = True
    t.start()
    return jsonify({'status': 'started'})

@app.route('/progress')
def progress():
    return jsonify({
        'training': state['training'],
        'progress': state['progress'],
        'logs': state['logs'],
        'done': state['done'],
        'error': state['error'],
        'accuracy_knn': state['accuracy_knn'],
        'accuracy_svm': state['accuracy_svm'],
        'f1_knn': state['f1_knn'],
        'f1_svm': state['f1_svm'],
        'cm_knn': state['cm_knn'],
        'cm_svm': state['cm_svm'],
    })

@app.route('/predict', methods=['POST'])
def predict_route():
    img_bytes = request.files['image'].read()
    results = {}

    if state['model_knn'] is not None:
        pred, conf, probas = predict_single(img_bytes, state['model_knn'], state['kmeans_knn'], state['K_knn'], 'knn')
        results['knn'] = {'prediction': pred, 'confidence': conf, 'all_probas': probas}

    if state['model_svm'] is not None:
        pred, conf, probas = predict_single(img_bytes, state['model_svm'], state['kmeans_svm'], state['K_svm'], 'svm')
        results['svm'] = {'prediction': pred, 'confidence': conf, 'all_probas': probas}

    if not results:
        return jsonify({'error': 'Aucun modèle disponible !'}), 400

    return jsonify(results)

@app.route('/status')
def status():
    return jsonify({
        'knn_ready': state['model_knn'] is not None,
        'svm_ready': state['model_svm'] is not None,
        'accuracy_knn': state['accuracy_knn'],
        'accuracy_svm': state['accuracy_svm'],
    })

if __name__ == '__main__':
    app.run(debug=True)