import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

try:
    from ml_pipeline.extract_hsv_features import extract_hsv_features
except ImportError:
    from extract_hsv_features import extract_hsv_features

def extract_all_features(image_path=None, image_bytes=None):
    """
    Extracts combined GLCM (6) + HSV (9) = 15 features from an image.
    Returns the combined feature vector and a merged feature dictionary.
    """
    glcm_vec, glcm_dict = extract_glcm_features(image_path=image_path, image_bytes=image_bytes)
    hsv_vec, hsv_dict = extract_hsv_features(image_path=image_path, image_bytes=image_bytes)

    combined_vec = np.concatenate([glcm_vec, hsv_vec])
    combined_dict = {**glcm_dict, **hsv_dict}

    return combined_vec, glcm_dict, hsv_dict

def extract_glcm_features(image_path=None, image_bytes=None):
    """
    Reads an image, resizes to 256x256, converts to grayscale, and extracts 6 GLCM features.
    """
    if image_path is not None:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
    elif image_bytes is not None:
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes")
    else:
        raise ValueError("Provide either image_path or image_bytes")

    # Convert to grayscale directly from the original image size
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate GLCM
    # distances = [1], angles = [0, 45, 90, 135 degrees]
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(img_gray, distances=distances, angles=angles, 
                        levels=256, symmetric=True, normed=True)
    
    # Extract features
    features = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    feature_dict = {}
    
    for feature in features:
        # Calculate the mean value across all 4 angles
        vals = graycoprops(glcm, feature)[0]
        feature_dict[feature] = float(np.mean(vals))
        
    # Return as an array in a fixed order
    feature_vector = [
        feature_dict['contrast'],
        feature_dict['dissimilarity'],
        feature_dict['homogeneity'],
        feature_dict['energy'],
        feature_dict['correlation'],
        feature_dict['ASM']
    ]
    return np.array(feature_vector), feature_dict