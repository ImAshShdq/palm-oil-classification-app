import numpy as np
import cv2
from scipy.stats import skew

def extract_hsv_features(image_path=None, image_bytes=None):
    """
    Reads an image and extracts 9 statistical HSV color features:
    - Mean, Std, Skewness for each of H, S, V channels.
    
    HSV is chosen because:
    - Hue represents the 'type' of color independent of brightness
    - Saturation represents color intensity
    - Value represents brightness
    This makes it more robust to lighting variations than RGB.
    
    Returns:
        feature_vector (np.array): 9-element array [mean_h, std_h, skew_h, mean_s, std_s, skew_s, mean_v, std_v, skew_v]
        feature_dict (dict): Dictionary with named features
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

    # Convert BGR to HSV
    # OpenCV HSV ranges: H=[0,179], S=[0,255], V=[0,255]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Split channels
    h_channel = img_hsv[:, :, 0].flatten().astype(np.float64)
    s_channel = img_hsv[:, :, 1].flatten().astype(np.float64)
    v_channel = img_hsv[:, :, 2].flatten().astype(np.float64)

    # Calculate statistics for each channel
    feature_dict = {
        'mean_h': float(np.mean(h_channel)),
        'std_h': float(np.std(h_channel)),
        'skew_h': float(skew(h_channel)),
        'mean_s': float(np.mean(s_channel)),
        'std_s': float(np.std(s_channel)),
        'skew_s': float(skew(s_channel)),
        'mean_v': float(np.mean(v_channel)),
        'std_v': float(np.std(v_channel)),
        'skew_v': float(skew(v_channel)),
    }

    # Return as array in fixed order
    feature_vector = [
        feature_dict['mean_h'],
        feature_dict['std_h'],
        feature_dict['skew_h'],
        feature_dict['mean_s'],
        feature_dict['std_s'],
        feature_dict['skew_s'],
        feature_dict['mean_v'],
        feature_dict['std_v'],
        feature_dict['skew_v'],
    ]
    return np.array(feature_vector), feature_dict