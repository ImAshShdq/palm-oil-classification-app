import os
import pickle
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from ml_pipeline.extract_features import extract_all_features

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max upload

# Paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
KNN_PATH = os.path.join(MODELS_DIR, 'knn_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
PCA_PATH = os.path.join(MODELS_DIR, 'pca_model.pkl')
HSV_VALIDATION_PATH = os.path.join(MODELS_DIR, 'hsv_validation.json')

# Load models if they exist
knn = None
scaler = None
pca = None
hsv_validation = None

try:
    with open(KNN_PATH, 'rb') as f:
        knn = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(PCA_PATH, 'rb') as f:
        pca = pickle.load(f)
    with open(HSV_VALIDATION_PATH, 'r') as f:
        hsv_validation = json.load(f)
except Exception as e:
    print(f"Warning: Models not found or error loading models. {e}")

# Label mapping
LABEL_NAMES = {
    0: "Matang",
    1: "Mengkal",
    2: "Mentah"
}

def validate_hsv_color(hsv_dict):
    """
    Validates if the input image's HSV color distribution matches
    the training data's HSV distribution using z-score distance.
    
    Returns:
        is_valid (bool): True if colors match training distribution
        hsv_distance (float): Average z-score distance (lower = more similar)
        detail (dict): Per-feature z-score breakdown
    """
    if hsv_validation is None:
        return True, 0.0, {}
    
    hsv_mean = np.array(hsv_validation['hsv_mean'])
    hsv_std = np.array(hsv_validation['hsv_std'])
    hsv_threshold = hsv_validation['hsv_distance_threshold']
    
    # Build input HSV vector in same order as training
    input_hsv = np.array([
        hsv_dict['mean_h'], hsv_dict['std_h'], hsv_dict['skew_h'],
        hsv_dict['mean_s'], hsv_dict['std_s'], hsv_dict['skew_s'],
        hsv_dict['mean_v'], hsv_dict['std_v'], hsv_dict['skew_v'],
    ])
    
    # Calculate z-scores per feature
    z_scores = np.abs(input_hsv - hsv_mean) / hsv_std
    mean_z = float(np.mean(z_scores))
    
    feature_names = ['mean_h', 'std_h', 'skew_h', 'mean_s', 'std_s', 'skew_s', 'mean_v', 'std_v', 'skew_v']
    detail = {}
    for i, name in enumerate(feature_names):
        rng = hsv_validation['hsv_ranges'].get(name, {})
        detail[name] = {
            'value': float(input_hsv[i]),
            'z_score': float(z_scores[i]),
            'train_mean': float(hsv_mean[i]),
            'train_range': [rng.get('p1', 0), rng.get('p99', 0)]
        }
    
    # Adding 50% margin similar to original implementation
    is_valid = mean_z <= hsv_threshold * 1.5
    
    return is_valid, mean_z, detail

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dashboard_data', methods=['GET'])
def get_dashboard_data():
    eval_path = os.path.join(MODELS_DIR, 'evaluation_metrics.json')
    if not os.path.exists(eval_path):
        return jsonify({"error": "Dashboard data not found. Please run train_knn.py first."}), 404
    with open(eval_path, 'r') as f:
        data = json.load(f)
    
    # Include HSV validation ranges for dashboard display
    if hsv_validation:
        data['hsv_validation'] = {
            'threshold': hsv_validation['hsv_distance_threshold'],
            'ranges': hsv_validation['hsv_ranges'],
        }
    
    return jsonify(data)

@app.route('/api/predict', methods=['POST'])
def predict():
    if knn is None or scaler is None:
        return jsonify({"error": "Model KNN belum dilatih. Harap jalankan ml_pipeline/train_knn.py"}), 500
        
    if 'image' not in request.files:
        return jsonify({"error": "Tidak ada file gambar yang diunggah"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Tidak ada file yang dipilih"}), 400
        
    # Read image bytes
    try:
        img_bytes = file.read()
        # Extract combined GLCM (6) + HSV (9) = 15 features
        combined_vec, glcm_dict, hsv_dict = extract_all_features(image_bytes=img_bytes)
    except Exception as e:
        return jsonify({"error": f"Gagal memproses gambar: {str(e)}"}), 500
    
    # === VALIDASI WARNA HSV (TERPISAH) ===
    # Cek apakah distribusi warna input mirip dengan data latih kelapa sawit
    hsv_valid, hsv_distance, hsv_detail = validate_hsv_color(hsv_dict)
    
    if not hsv_valid:
        return jsonify({
            "valid": False,
            "error": "Distribusi warna gambar tidak cocok dengan kelapa sawit. Warna gambar terlalu berbeda dari data latih.",
            "hsv_valid": False,
            "hsv_distance": hsv_distance,
            "hsv_detail": hsv_detail,
            "features": glcm_dict,
            "hsv_features": hsv_dict,
        }), 400
        
    # Scale features (15-dimensional vector)
    X_scaled = scaler.transform([combined_vec])
    
    # KNN Prediction
    predicted_label = int(knn.predict(X_scaled)[0])
    probabilities = knn.predict_proba(X_scaled)[0]
    confidence = float(np.max(probabilities))
    
    # Get label name
    label = LABEL_NAMES.get(predicted_label, "Tidak Diketahui")
    
    # PCA coordinates for visualization (if PCA model exists)
    pca_x = None
    pca_y = None
    if pca is not None:
        X_pca = pca.transform(X_scaled)[0]
        pca_x = float(X_pca[0])
        pca_y = float(X_pca[1])
    
    # Build probability breakdown per class
    proba_detail = {}
    for i, prob in enumerate(probabilities):
        class_name = LABEL_NAMES.get(i, f"Class {i}")
        proba_detail[class_name] = float(prob)
    
    return jsonify({
        "valid": True,
        "label": label,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": proba_detail,
        "features": glcm_dict,
        "hsv_features": hsv_dict,
        "hsv_valid": True,
        "hsv_distance": hsv_distance,
        "hsv_detail": hsv_detail,
        "pca_x": pca_x,
        "pca_y": pca_y,
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)