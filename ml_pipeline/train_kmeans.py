"""
train_kmeans.py
===============
Pipeline pelatihan model untuk deteksi & klasifikasi kematangan kelapa sawit.

Arsitektur validasi (AND-gate):
  ┌─────────────┐     ┌─────────────┐
  │  GLCM (6)   │──►  │  K-Means    │──► GLCM_valid  (tekstur ≈ sawit?)
  └─────────────┘     │  threshold  │
                      └─────────────┘
  ┌─────────────┐     ┌─────────────┐
  │  HSV  (9)   │──►  │  z-score    │──► HSV_valid   (warna  ≈ sawit?)
  └─────────────┘     │  threshold  │
                      └─────────────┘

  GLCM_valid AND HSV_valid  →  Kelapa Sawit  (+ label Matang/Mengkal/Mentah)
  salah satu False           →  Bukan kelapa sawit

K-Means dilatih **hanya pada fitur GLCM** agar clustering murni
menangkap pola tekstur.  Label (Matang/Mengkal/Mentah) ditentukan
dari mean_h centroid masing-masing cluster setelah pelatihan.
"""

import os
import glob
import json
import pickle
import numpy as np
from extract_features import extract_glcm_features
from extract_hsv_features import extract_hsv_features
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Paths relative to this script inside ml_pipeline/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
TEST_DIR = os.path.join(BASE_DIR, 'dataset', 'test')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def get_image_paths(directory):
    return glob.glob(os.path.join(directory, '*.jpg')) + \
           glob.glob(os.path.join(directory, '*.jpeg')) + \
           glob.glob(os.path.join(directory, '*.png'))

def extract_dataset(directory):
    """
    Extracts combined GLCM (6) + HSV (9) = 15 features from all images in a directory.
    """
    paths = get_image_paths(directory)
    features_list = []
    print(f"Mengekstrak fitur GLCM + HSV dari {len(paths)} gambar di: {directory}")
    for idx, path in enumerate(paths):
        try:
            # Extract GLCM features (6 features)
            glcm_vec, _ = extract_glcm_features(image_path=path)
            # Extract HSV features (9 features)
            hsv_vec, _ = extract_hsv_features(image_path=path)
            # Combine into 15-dimensional feature vector
            combined_vec = np.concatenate([glcm_vec, hsv_vec])
            features_list.append(combined_vec)
            if (idx + 1) % 100 == 0:
                print(f"  Memproses {idx+1}/{len(paths)} gambar...")
        except Exception as e:
            print(f"Error pada {path}: {e}")
    return np.array(features_list), paths

def train_kmeans():
    X_train, train_paths = extract_dataset(TRAIN_DIR)
    
    if len(X_train) == 0:
        print("Error: Tidak ada data latih")
        return
        
    print(f"Total data latih diproses: {X_train.shape} (6 GLCM + 9 HSV = 15 fitur)")
    
    # 1. Scaling Fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # 2. PCA to 2D
    print("Melakukan reduksi dimensi dengan PCA (15D → 2D)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 3. Melatih K-Means di ruang PCA
    print("Melatih model K-Means (K=3) di ruang PCA...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    
    # 4. Pengurutan Centroid Dinamis berdasarkan Mean Hue (HSV fitur index 6)
    # Proyeksikan kembali cluster centers di ruang PCA ke ruang fitur asli untuk dianalisis
    centroids_scaled = pca.inverse_transform(kmeans.cluster_centers_)
    centroids_original = scaler.inverse_transform(centroids_scaled)
    
    # Urutkan berdasarkan mean_hue secara menaik (ascending)
    # Index 6 = mean_h (fitur pertama dari HSV setelah 6 fitur GLCM)
    # Sawit matang: Hue rendah (merah/oranye ≈ 0-20 di OpenCV HSV)
    # Sawit mentah: Hue tinggi (hijau ≈ 35-85 di OpenCV HSV)
    # index 0: Matang (hue terendah), 1: Mengkal, 2: Mentah (hue tertinggi)
    sort_order = np.argsort(centroids_original[:, 6])  # index 6 = mean_h
    
    print(f"  Mean Hue per centroid (sebelum sort): {centroids_original[:, 6]}")
    print(f"  Urutan sort (matang→mentah): {sort_order}")
    
    # Update cluster centers di K-Means sesuai urutan baru
    kmeans.cluster_centers_ = kmeans.cluster_centers_[sort_order]
    
    # Hitung label training berdasarkan centroid yang sudah diurutkan
    distances = kmeans.transform(X_pca)
    train_labels = np.argmin(distances, axis=1)
    
    # 5. Menghitung Threshold di ruang PCA
    closest_distances = np.min(distances, axis=1)
    threshold = float(np.percentile(closest_distances, 99))
    print(f"Threshold Jarak Validasi Objek (PCA): {threshold:.4f}")
    
    # 5b. Menghitung Validasi Warna HSV (terpisah dari PCA)
    # Ekstrak hanya fitur HSV (index 6-14) dari data latih asli (belum di-scale)
    print("Menghitung distribusi warna HSV untuk validasi...")
    hsv_features_train = X_train[:, 6:]  # 9 fitur HSV
    hsv_feature_names = ['mean_h', 'std_h', 'skew_h', 'mean_s', 'std_s', 'skew_s', 'mean_v', 'std_v', 'skew_v']
    
    hsv_mean = np.mean(hsv_features_train, axis=0)
    hsv_std = np.std(hsv_features_train, axis=0)
    # Hindari pembagian nol
    hsv_std_safe = np.where(hsv_std == 0, 1e-6, hsv_std)
    
    # Hitung z-score jarak untuk setiap sampel latih
    # Z-score = abs(x - mean) / std, lalu rata-ratakan per sampel
    z_scores_train = np.abs(hsv_features_train - hsv_mean) / hsv_std_safe
    mean_z_per_sample = np.mean(z_scores_train, axis=1)
    hsv_distance_threshold = float(np.percentile(mean_z_per_sample, 99))
    
    # Simpan juga range per fitur (percentile 1-99) untuk referensi
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
    print(f"Threshold Jarak Validasi Warna HSV: {hsv_distance_threshold:.4f}")
    for name, rng in hsv_ranges.items():
        print(f"  {name}: mean={rng['mean']:.2f}, std={rng['std']:.2f}, range=[{rng['p1']:.2f}, {rng['p99']:.2f}]")
    
    # 6. Metrik Evaluasi Unsupervised
    print("Menghitung metrik evaluasi...")
    # a. Silhouette Score di ruang PCA
    sil_score = float(silhouette_score(X_pca, train_labels))
    print(f"Silhouette Score (Optimized): {sil_score:.4f}")
    
    # b. Distribusi Data Latih
    train_dist = {str(k): int(v) for k, v in zip(*np.unique(train_labels, return_counts=True))}
    
    # c. Sampel data latih untuk Scatter Plot
    # Ambil sampel max 200 data agar JSON tidak terlalu besar dan berat di Frontend
    sample_indices = np.random.choice(len(X_pca), min(200, len(X_pca)), replace=False)
    pca_data = []
    for idx in sample_indices:
        pca_data.append({
            'x': float(X_pca[idx, 0]),
            'y': float(X_pca[idx, 1]),
            'cluster': int(train_labels[idx])
        })

    # d. Distribusi Data Test (Evaluasi Test)
    X_test, test_paths = extract_dataset(TEST_DIR)
    test_dist = {}
    test_predictions = []
    if len(X_test) > 0:
        X_test_scaled = scaler.transform(X_test)
        X_test_pca = pca.transform(X_test_scaled)
        # Hitung jarak dan label di ruang PCA
        test_distances = kmeans.transform(X_test_pca)
        test_labels = np.argmin(test_distances, axis=1)
        test_dist = {str(k): int(v) for k, v in zip(*np.unique(test_labels, return_counts=True))}
        
        mapping = {0: "Matang", 1: "Mengkal", 2: "Mentah"}
        for i, path in enumerate(test_paths):
            filename = os.path.basename(path)
            cluster_id = int(test_labels[i])
            label_name = mapping.get(cluster_id, "Tidak Diketahui")
            test_predictions.append({
                "filename": filename,
                "cluster": cluster_id,
                "label": label_name
            })
    
    # e. Statistik centroid untuk evaluasi
    centroid_stats = {}
    for c_id in range(3):
        c_orig = centroids_original[sort_order[c_id]]
        centroid_stats[str(c_id)] = {
            'glcm': {
                'contrast': float(c_orig[0]),
                'dissimilarity': float(c_orig[1]),
                'homogeneity': float(c_orig[2]),
                'energy': float(c_orig[3]),
                'correlation': float(c_orig[4]),
                'ASM': float(c_orig[5]),
            },
            'hsv': {
                'mean_h': float(c_orig[6]),
                'std_h': float(c_orig[7]),
                'skew_h': float(c_orig[8]),
                'mean_s': float(c_orig[9]),
                'std_s': float(c_orig[10]),
                'skew_s': float(c_orig[11]),
                'mean_v': float(c_orig[12]),
                'std_v': float(c_orig[13]),
                'skew_v': float(c_orig[14]),
            }
        }
        
    evaluation_metrics = {
        'silhouette_score': sil_score,
        'train_distribution': train_dist,
        'test_distribution': test_dist,
        'test_predictions': test_predictions,
        'pca_scatter': pca_data,
        'centroid_stats': centroid_stats,
        'feature_info': {
            'total_features': 15,
            'glcm_features': ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'],
            'hsv_features': ['mean_h', 'std_h', 'skew_h', 'mean_s', 'std_s', 'skew_s', 'mean_v', 'std_v', 'skew_v'],
        },
        'cluster_mapping': {
            '0': 'Matang (Hue rendah: oranye kemerahan/merah)',
            '1': 'Mengkal (Hue menengah: warna transisi)',
            '2': 'Mentah (Hue tinggi: hijau gelap, kuning kehijauan)'
        }
    }

    # 7. Menyimpan Model
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, 'kmeans_model.pkl'), 'wb') as f:
        pickle.dump(kmeans, f)
    with open(os.path.join(MODELS_DIR, 'pca_model.pkl'), 'wb') as f:
        pickle.dump(pca, f)
    with open(os.path.join(MODELS_DIR, 'threshold.json'), 'w') as f:
        json.dump({'distance_threshold': threshold}, f)
    with open(os.path.join(MODELS_DIR, 'hsv_validation.json'), 'w') as f:
        json.dump(hsv_validation, f)
    with open(os.path.join(MODELS_DIR, 'evaluation_metrics.json'), 'w') as f:
        json.dump(evaluation_metrics, f)
        
    print(f"Model & Metrik Evaluasi berhasil disimpan.")
    print(f"  - Fitur: 6 GLCM + 9 HSV = 15 total")
    print(f"  - PCA: 15D → 2D")
    print(f"  - Silhouette Score: {sil_score:.4f}")

if __name__ == '__main__':
    train_kmeans()