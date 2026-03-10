import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import keras
from keras.models import load_model
from keras.applications.efficientnet_v2 import preprocess_input
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import components

app = Flask(__name__)
CORS(app)

base_options = mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE, 
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

@keras.saving.register_keras_serializable()
def custom_preprocess(x):
    return preprocess_input(x)

IMG_SIZE = 224
CLASS_NAMES =  ["a", "b", "c", "d", "e", "f", "g", "h",
                "i", "j", "k", "l", "m", "_", "n", "\u00f1", 
                "o", "p", "q", "r", "s", "t", "u", "v", 
                "w", "x", "y", "z", "!"]
 
MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH, custom_objects={'preprocess_input': custom_preprocess}, compile=False)

def prepare_image(file_stream):
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        x_coords = [int(lm.x * w) for lm in hand_landmarks]
        y_coords = [int(lm.y * h) for lm in hand_landmarks]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        coordenates = { 
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max
        }

        hand_w = x_max - x_min
        hand_h = y_max - y_min
        side = int(max(hand_w, hand_h) * 1.3)
        center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
        
        nx_min = max(0, center_x - side // 2)
        ny_min = max(0, center_y - side // 2)
        nx_max = min(w, nx_min + side)
        ny_max = min(h, ny_min + side)

        final_img = img_rgb[ny_min:ny_max, nx_min:nx_max]
        final_img = cv2.resize(final_img, (IMG_SIZE, IMG_SIZE))
    else:
        final_img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        coordenates = False

    x = np.array(final_img, dtype='float32')
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x), coordenates


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image sent'}), 400
    
    file = request.files['image']
    try:
        processed_img, coordenates = prepare_image(file)
        predictions = model.predict(processed_img, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)

        return jsonify({
            'class': CLASS_NAMES[class_idx],
            'confidence': f"{confidence:.2f}%",
            'coordenates': coordenates
        })
    
    except Exception as e:
        return jsonify({'error': f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)