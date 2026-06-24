"""
train_knn.py
============
Pipeline pelatihan model KNN untuk klasifikasi kematangan kelapa sawit.

Arsitektur:
  ┌─────────────────┐     ┌─────────────────┐
  │  Gambar Input    │──►  │  Ekstraksi Fitur │──► 15 fitur (6 GLCM + 9 HSV)
  └─────────────────┘     └─────────────────┘
                                  │
                      ┌───────────┼───────────┐
                      ▼                       ▼
              ┌───────────────┐     ┌─────────────────┐
              │ Validasi HSV  │     │  Scaling +       │
              │ (z-score)     │     │  KNN.predict()   │
              └───────────────┘     └─────────────────┘
                      │                       │
                      ▼                       ▼
              Bukan sawit?              Label + Confidence
              → Tolak                   → Matang/Mengkal/Mentah

Label diperoleh dari nama file:
  - berondol_10_xxx.jpg → Matang  (label 0)
  - berondol_5_xxx.jpg  → Mengkal (label 1)
  - berondol_1_xxx.jpg  → Mentah  (label 2)

Fitur:
  - 6 GLCM: contrast, dissimilarity, homogeneity, energy, correlation, ASM
  - 9 HSV:  mean_h, std_h, skew_h, mean_s, std_s, skew_s, mean_v, std_v, skew_v
"""

import os
import glob
import json
import pickle
import numpy as np
from extract_features import extract_glcm_features
from extract_hsv_features import extract_hsv_features
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Paths relative to this script inside ml_pipeline/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
TEST_DIR = os.path.join(BASE_DIR, 'dataset', 'test')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Label mapping berdasarkan jumlah berondol
LABEL_MAP = {
    'berondol_10': 0,  # Matang
    'berondol_5': 1,   # Mengkal
    'berondol_1': 2,   # Mentah
}

LABEL_NAMES = {
    0: 'Matang',
    1: 'Mengkal',
    2: 'Mentah',
}


def get_image_paths(directory):
    return glob.glob(os.path.join(directory, '*.jpg')) + \
           glob.glob(os.path.join(directory, '*.jpeg')) + \
           glob.glob(os.path.join(directory, '*.png'))


def parse_label_from_filename(filename):
    """
    Mengekstrak label dari nama file berdasarkan prefix berondol_X.
    Contoh: berondol_10_123.jpg → label 0 (Matang)
    """
    basename = os.path.basename(filename)
    for prefix, label in LABEL_MAP.items():
        if basename.startswith(prefix + '_'):
            return label
    return None  # Tidak dikenali


def extract_dataset(directory):
    """
    Mengekstrak fitur GLCM (6) + HSV (9) = 15 fitur dari semua gambar,
    beserta label dari nama file.
    """
    paths = get_image_paths(directory)
    features_list = []
    labels_list = []
    valid_paths = []

    print(f"Mengekstrak fitur GLCM + HSV dari {len(paths)} gambar di: {directory}")
    skipped = 0

    for idx, path in enumerate(paths):
        try:
            # Parse label dari nama file
            label = parse_label_from_filename(path)
            if label is None:
                skipped += 1
                continue

            # Extract GLCM features (6 features)
            glcm_vec, _ = extract_glcm_features(image_path=path)
            # Extract HSV features (9 features)
            hsv_vec, _ = extract_hsv_features(image_path=path)
            # Combine into 15-dimensional feature vector
            combined_vec = np.concatenate([glcm_vec, hsv_vec])

            features_list.append(combined_vec)
            labels_list.append(label)
            valid_paths.append(path)

            if (idx + 1) % 100 == 0:
                print(f"  Memproses {idx+1}/{len(paths)} gambar...")
        except Exception as e:
            print(f"Error pada {path}: {e}")

    if skipped > 0:
        print(f"  ⚠️ {skipped} gambar dilewati (nama file tidak sesuai format berondol_X_Y.jpg)")

    return np.array(features_list), np.array(labels_list), valid_paths


def find_best_k(X_scaled, y, k_range=range(1, 21)):
    """
    Mencari nilai K terbaik menggunakan Stratified K-Fold Cross Validation.
    """
    print("\n🔍 Mencari nilai K terbaik (K=1..20) dengan 5-Fold CV...")
    best_k = 1
    best_score = 0
    cv_results = {}

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='minkowski')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(knn, X_scaled, y, cv=cv, scoring='accuracy')
        mean_score = scores.mean()
        cv_results[k] = {
            'mean_accuracy': float(mean_score),
            'std_accuracy': float(scores.std()),
        }
        marker = ' ← TERBAIK' if mean_score > best_score else ''
        if mean_score > best_score:
            best_score = mean_score
            best_k = k
        print(f"  K={k:2d}: Accuracy = {mean_score:.4f} (±{scores.std():.4f}){marker}")

    print(f"\n✅ K terbaik: {best_k} dengan Accuracy CV: {best_score:.4f}")
    return best_k, cv_results


def train_knn():
    # ============================================================
    # 1. Ekstraksi Fitur & Label dari Dataset
    # ============================================================
    X_train, y_train, train_paths = extract_dataset(TRAIN_DIR)

    if len(X_train) == 0:
        print("Error: Tidak ada data latih")
        return

    print(f"\nTotal data latih: {X_train.shape[0]} gambar, {X_train.shape[1]} fitur (6 GLCM + 9 HSV)")
    for label_id, label_name in LABEL_NAMES.items():
        count = np.sum(y_train == label_id)
        print(f"  {label_name} (label {label_id}): {count} gambar")

    # ============================================================
    # 2. Scaling Fitur
    # ============================================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # ============================================================
    # 3. Mencari K Terbaik (Cross-Validation)
    # ============================================================
    best_k, cv_results = find_best_k(X_scaled, y_train)

    # ============================================================
    # 4. Melatih KNN dengan K Terbaik
    # ============================================================
    print(f"\n🏋️ Melatih KNN final (K={best_k}) pada seluruh data latih...")
    knn = KNeighborsClassifier(
        n_neighbors=best_k,
        weights='distance',
        metric='minkowski',
    )
    knn.fit(X_scaled, y_train)

    # Prediksi pada data latih (untuk metrik training)
    train_pred = knn.predict(X_scaled)
    train_accuracy = float(accuracy_score(y_train, train_pred))
    print(f"  Training Accuracy: {train_accuracy:.4f}")

    # ============================================================
    # 5. PCA untuk Visualisasi (2D) - Bukan untuk klasifikasi
    # ============================================================
    print("\n📊 Menghitung PCA (15D → 2D) untuk visualisasi...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"  Total Variance Captured: {sum(pca.explained_variance_ratio_):.4f}")

    # ============================================================
    # 6. Validasi Warna HSV (Tetap dipertahankan untuk deteksi non-sawit)
    # ============================================================
    print("\n🎨 Menghitung distribusi warna HSV untuk validasi objek...")
    hsv_features_train = X_train[:, 6:]  # 9 fitur HSV
    hsv_feature_names = ['mean_h', 'std_h', 'skew_h', 'mean_s', 'std_s', 'skew_s', 'mean_v', 'std_v', 'skew_v']

    hsv_mean = np.mean(hsv_features_train, axis=0)
    hsv_std = np.std(hsv_features_train, axis=0)
    hsv_std_safe = np.where(hsv_std == 0, 1e-6, hsv_std)

    z_scores_train = np.abs(hsv_features_train - hsv_mean) / hsv_std_safe
    mean_z_per_sample = np.mean(z_scores_train, axis=1)
    hsv_distance_threshold = float(np.percentile(mean_z_per_sample, 99))

    hsv_ranges = {}
    for i, name in enumerate(hsv_feature_names):
        hsv_ranges[name] = {
            'mean': float(hsv_mean[i]),
            'std': float(hsv_std[i]),
            'p1': float(np.percentile(hsv_features_train[:, i], 1)),
            'p99': float(np.percentile(hsv_features_train[:, i], 99)),
        }

    hsv_validation = {
        'hsv_mean': hsv_mean.tolist(),
        'hsv_std': hsv_std_safe.tolist(),
        'hsv_distance_threshold': hsv_distance_threshold,
        'hsv_ranges': hsv_ranges,
    }
    print(f"  HSV Distance Threshold: {hsv_distance_threshold:.4f}")

    # ============================================================
    # 7. Evaluasi pada Data Test
    # ============================================================
    print("\n📋 Mengevaluasi pada data test...")
    X_test, y_test, test_paths = extract_dataset(TEST_DIR)

    test_metrics = {}
    test_predictions = []
    test_dist = {}
    test_confusion = []

    if len(X_test) > 0:
        X_test_scaled = scaler.transform(X_test)
        test_pred = knn.predict(X_test_scaled)
        test_proba = knn.predict_proba(X_test_scaled)

        test_accuracy = float(accuracy_score(y_test, test_pred))
        test_precision = float(precision_score(y_test, test_pred, average='weighted', zero_division=0))
        test_recall = float(recall_score(y_test, test_pred, average='weighted', zero_division=0))
        test_f1 = float(f1_score(y_test, test_pred, average='weighted', zero_division=0))

        # Per-class metrics
        per_class_precision = precision_score(y_test, test_pred, average=None, zero_division=0).tolist()
        per_class_recall = recall_score(y_test, test_pred, average=None, zero_division=0).tolist()
        per_class_f1 = f1_score(y_test, test_pred, average=None, zero_division=0).tolist()

        test_metrics = {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'per_class': {
                LABEL_NAMES[i]: {
                    'precision': float(per_class_precision[i]) if i < len(per_class_precision) else 0,
                    'recall': float(per_class_recall[i]) if i < len(per_class_recall) else 0,
                    'f1_score': float(per_class_f1[i]) if i < len(per_class_f1) else 0,
                }
                for i in range(3)
            }
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, test_pred, labels=[0, 1, 2])
        test_confusion = cm.tolist()

        # Distribution
        test_dist = {str(k): int(v) for k, v in zip(*np.unique(test_pred, return_counts=True))}

        # Per-image predictions
        for i, path in enumerate(test_paths):
            filename = os.path.basename(path)
            pred_label = int(test_pred[i])
            true_label = int(y_test[i])
            confidence = float(np.max(test_proba[i]))
            test_predictions.append({
                "filename": filename,
                "predicted_label": pred_label,
                "predicted_name": LABEL_NAMES.get(pred_label, "?"),
                "true_label": true_label,
                "true_name": LABEL_NAMES.get(true_label, "?"),
                "confidence": confidence,
                "correct": pred_label == true_label,
            })

        print(f"\n  📊 Test Results:")
        print(f"     Accuracy:  {test_accuracy:.4f}")
        print(f"     Precision: {test_precision:.4f}")
        print(f"     Recall:    {test_recall:.4f}")
        print(f"     F1 Score:  {test_f1:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"     {['Matang', 'Mengkal', 'Mentah']}")
        for i, row in enumerate(cm):
            print(f"     {LABEL_NAMES[i]:8s}: {row}")

        print(f"\n  Classification Report:")
        print(classification_report(y_test, test_pred, target_names=['Matang', 'Mengkal', 'Mentah']))
    else:
        print("  ⚠️ Tidak ada data test.")

    # ============================================================
    # 8. PCA Scatter Data untuk Visualisasi Frontend
    # ============================================================
    sample_indices = np.random.choice(len(X_pca), min(200, len(X_pca)), replace=False)
    pca_data = []
    for idx in sample_indices:
        pca_data.append({
            'x': float(X_pca[idx, 0]),
            'y': float(X_pca[idx, 1]),
            'label': int(y_train[idx]),
        })

    # ============================================================
    # 9. Train Distribution
    # ============================================================
    train_dist = {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))}

    # ============================================================
    # 10. Compile Evaluation Metrics
    # ============================================================
    evaluation_metrics = {
        'model_type': 'KNN',
        'best_k': best_k,
        'cv_results': cv_results,
        'train_accuracy': train_accuracy,
        'train_distribution': train_dist,
        'test_metrics': test_metrics,
        'test_distribution': test_dist,
        'test_confusion_matrix': test_confusion,
        'test_predictions': test_predictions,
        'pca_scatter': pca_data,
        'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
        'feature_info': {
            'total_features': 15,
            'glcm_features': ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'],
            'hsv_features': ['mean_h', 'std_h', 'skew_h', 'mean_s', 'std_s', 'skew_s', 'mean_v', 'std_v', 'skew_v'],
        },
        'label_mapping': {
            '0': 'Matang (berondol_10: buah matang, warna merah jingga)',
            '1': 'Mengkal (berondol_5: buah setengah matang, warna oranye kehijauan)',
            '2': 'Mentah (berondol_1: buah mentah, warna kehitaman)',
        },
    }

    # ============================================================
    # 11. Menyimpan Model & Metrik
    # ============================================================
    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(os.path.join(MODELS_DIR, 'knn_model.pkl'), 'wb') as f:
        pickle.dump(knn, f)
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, 'pca_model.pkl'), 'wb') as f:
        pickle.dump(pca, f)
    with open(os.path.join(MODELS_DIR, 'hsv_validation.json'), 'w') as f:
        json.dump(hsv_validation, f)
    with open(os.path.join(MODELS_DIR, 'evaluation_metrics.json'), 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)

    print(f"\n✅ Model & Metrik Evaluasi berhasil disimpan ke {MODELS_DIR}/")
    print(f"   - knn_model.pkl (K={best_k})")
    print(f"   - scaler.pkl")
    print(f"   - pca_model.pkl (untuk visualisasi)")
    print(f"   - hsv_validation.json")
    print(f"   - evaluation_metrics.json")


if __name__ == '__main__':
    train_knn()
