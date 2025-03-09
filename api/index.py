import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from PIL import Image
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 
@app.route('/')
def serve_frontend():
    return jsonify({"message": "Welcome to the auth API!", "status": "running"}), 200

# Load MobileNetV2 from TensorFlow Hub
model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
model = hub.KerasLayer(model_url, input_shape=(224, 224, 3), trainable=False)

# Preprocess image for MobileNetV2
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = img.astype(np.float32)
    return np.expand_dims(img, axis=0)

# Extract features
def extract_features(image_path):
    img = preprocess_image(image_path)
    features = model(img)
    return features.numpy().flatten()

# Convert WebP/JFIF to JPG if needed
def convert_to_jpg(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.lower().endswith((".webp", ".jfif")):
        img = Image.open(image_path).convert("RGB")
        new_path = image_path.rsplit(".", 1)[0] + ".jpg"
        img.save(new_path)
        print(f"Converted {image_path} to {new_path}")
        return new_path
    return image_path

# Check for EXIF metadata
def has_exif_data(image_path):
    try:
        img = Image.open(image_path)
        exif = img.getexif()
        has_exif = bool(exif)
        print(f"EXIF data present: {has_exif}")
        return has_exif
    except Exception:
        print("No EXIF data (exception occurred)")
        return False

# Calculate image sharpness (Laplacian variance)
def calculate_sharpness(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image for sharpness: {image_path}")
    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
    print(f"Image sharpness (Laplacian variance): {sharpness}")
    return sharpness

# Save result to a text file
def save_result_to_file(image_path, result, file_path):
    with open(file_path, "a") as f:
        f.write(f"\nTesting: {image_path}\n")
        f.write(f"Feature vector norm: {result['norm']}\n")
        f.write(f"Feature variance: {result['variance']}\n")
        f.write(f"Image sharpness: {result['sharpness']}\n")
        f.write(f"EXIF data present: {'True' if has_exif_data(image_path) else 'False'}\n")
        f.write(f"Result: {result['result']}, Confidence: {result['confidence']}\n")
        f.write("-" * 50 + "\n")

# Check authenticity (heuristic-based)
def check_authenticity(image_path):
    try:
        image_path = convert_to_jpg(image_path)
        
        # Extract features
        features = extract_features(image_path)
        feature_norm = np.linalg.norm(features)
        feature_variance = np.var(features)
        print(f"Feature vector norm: {feature_norm}")
        print(f"Feature variance: {feature_variance}")
        
        # Check metadata
        has_metadata = has_exif_data(image_path)
        
        # Check sharpness
        sharpness = calculate_sharpness(image_path)
        
        # Decision logic (adjust thresholds based on your tests)
        threshold_norm = 25.0
        threshold_variance = 0.1
        threshold_sharpness = 100.0
        
        if (feature_norm < threshold_norm and feature_variance < threshold_variance) or has_metadata:
            result = "Likely authentic"
            confidence = 0.9 if has_metadata else 0.7
        elif feature_norm > threshold_norm or feature_variance > threshold_variance or sharpness > threshold_sharpness:
            result = "Likely from internet"
            confidence = 0.8
        else:
            result = "Uncertain"
            confidence = 0.6
        
        # Prepare result
        result_dict = {
            "result": result,
            "confidence": confidence,
            "norm": float(feature_norm),
            "variance": float(feature_variance),
            "sharpness": float(sharpness)
        }
        
        # Save to text file
        result_file = r"C:\Users\Muhammad Haroon\Desktop\py_img\authenticity_results.txt"
        save_result_to_file(image_path, result_dict, result_file)
        
        # Clean up temporary file
        if image_path != image_path.rsplit(".", 1)[0] + ".jpg":
            os.remove(image_path)
        
        return result_dict
    except Exception as e:
        print(f"Error: {e}")
        return {"result": "Error in processing", "confidence": 0.0}

# API endpoint to check image authenticity
@app.route('/check_authenticity', methods=['POST'])
def api_check_authenticity():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    # Save the uploaded image temporarily
    upload_folder = r"C:\Users\Muhammad Haroon\Desktop\py_img\uploads"
    os.makedirs(upload_folder, exist_ok=True)
    image_path = os.path.join(upload_folder, image_file.filename)
    image_file.save(image_path)
    
    # Check authenticity
    result = check_authenticity(image_path)
    
    # Clean up the uploaded file
    if os.path.exists(image_path):
        os.remove(image_path)
    
    return jsonify(result)

# Run the API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
