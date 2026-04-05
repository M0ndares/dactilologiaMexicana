import numpy as np
import cv2
import os 
from flask import Flask, request, jsonify
from flask_cors import CORS
import mediapipe as mp
from mediapipe.tasks.python import vision
import gc 
import tflite_runtime.interpreter as tflite 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
app = Flask(__name__)
CORS(app)

base_options = mp.tasks.BaseOptions(model_asset_path='modelo/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE, 
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)
IMG_SIZE = 224
CLASS_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h",
                "i", "j", "k", "l", "m", "_", "n", "ñ", 
                "o", "p", "q", "r", "s", "t", "u", "v", 
                "w", "x", "y", "z", "!"]
MODEL_PATH = 'modelo/model3.tflite'
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def prepare_image(file_stream):
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None: return None
    
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
        side = int(max(x_max - x_min, y_max - y_min) * 1.3)
        center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
        
        nx_min = max(0, center_x - side // 2)
        ny_min = max(0, center_y - side // 2)
        nx_max = min(w, nx_min + side)
        ny_max = min(h, ny_min + side)

        final_img = img_rgb[ny_min:ny_max, nx_min:nx_max]
        
        if final_img.size == 0: return None

        final_img = cv2.resize(final_img, (IMG_SIZE, IMG_SIZE))
        
        x = np.array(final_img, dtype='float32')
        x = np.expand_dims(x, axis=0)
        
        del img, img_rgb, file_bytes
        return (x / 127.5) - 1.0
    
    return None

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    processed_img = None 
    try:
        processed_img = prepare_image(file)
        if processed_img is None:
            return jsonify({'class': 'None', 'confidence': 'Ninguna seña detectada'})
        
        interpreter.set_tensor(input_details[0]['index'], processed_img)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        class_idx = np.argmax(predictions)
        confidence = float(predictions[0][class_idx] * 100)


        #if confidence < 70: 
        #   return jsonify({'class': 'None', 'confidence': "Ninguna seña detectada"})

        return jsonify({
            'class': CLASS_NAMES[class_idx],
            'confidence': f"{confidence:.2f}%",
        })
    
    except Exception as e:
        return jsonify({'error': f"Error: {str(e)}"}), 500
    finally:
        if processed_img is not None:
            del processed_img
        gc.collect()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)