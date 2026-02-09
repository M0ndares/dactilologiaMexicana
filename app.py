
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import keras
from keras.models import load_model
from keras.applications.resnet_v2 import preprocess_input
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands

hands_detector = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)

@keras.saving.register_keras_serializable()
def custom_preprocess(x):
    return preprocess_input(x)

app = Flask(__name__)
CORS(app)

IMG_SIZE = 224
CLASS_NAMES = [
    "b", "c", "d", "e", "f", "g", "h", "i",
    "j", "k", "l", "m", "_", "n", "\u00f1",
    "o", "p", "r", "s", "t", "u", "v", "w",
    "x", "y", "z", "a", "q"
]

model = None
load_error = None

MODEL_PATH = 'model.h5'
custom_dict = {'preprocess_input': custom_preprocess}
model = load_model(MODEL_PATH, custom_objects=custom_dict, compile=False)

def prepare_image(file_stream):
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
    
        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
        
        margin = 30
        x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
        x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)
        hand_crop = img[y_min:y_max, x_min:x_max]
     
        if hand_crop.size != 0:
            final_img = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
        else:
            final_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    else:
        final_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    x = np.expand_dims(final_img, axis=0).astype('float32')
    return preprocess_input(x)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model could not be loaded.', 'Details': load_error}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image sent'}), 400
    
    file = request.files['image']
    try:
        processed_img = prepare_image(file)
        predictions = model.predict(processed_img, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        
        return jsonify({
            'class': CLASS_NAMES[class_idx],
            'confidence': f"{confidence:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10001)