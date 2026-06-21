# Dactimex: LSM Sign Language Recognizer

Full-Stack Computer Vision web application designed to detect and translate the 27 letters of the Mexican Sign Language (LSM) alphabet in real-time.

![Dactimex Project Demo](dactimex.gif)

**Dactimex** bridges the communication gap for the deaf and hard-of-hearing community by leveraging Deep Learning to translate hand gestures into text.

---

### Demo
- **Frontend:** [https://m0ndares.github.io/dactilologiaMexicana/html.html](https://m0ndares.github.io/dactilologiaMexicana/html.html)
- **Backend API:** [https://dactilologiamexicana.onrender.com](https://dactilologiamexicana.onrender.com)

---

### The Problem
Real-time image classification in the cloud can be slow and resource-heavy. Traditional models process full frames, which introduces background noise and lowers prediction accuracy for complex gesture-based alphabets like LSM (Lengua de Señas Mexicana).

### The Solution
This project implements an optimized Full-Stack pipeline:
1. **Frontend:** Captures video stream frames and sends them via API.
2. **Backend (Hybrid AI):** Uses MediaPipe to isolate the exact coordinates of the hand, crops the Region of Interest (ROI) dynamically with OpenCV, and feeds only the cropped hand into a custom-trained **TensorFlow Lite** model for alphabet classification.

---

### Key Features
* **Full LSM Alphabet Detection:** Accurately recognizes 27 letters and 2 special characters.
* **Smart ROI Cropping:** Automatically centers, pads, and resizes the hand bounding box to maximize model accuracy.
* **Thread-Safe Architecture:** Implements a Python `threading.Lock` mechanism to handle concurrent API inference requests safely.
* **Memory-Optimized Pipeline:** Integrates active garbage collection (`gc`) and manual tensor cleanup to prevent memory leaks in cloud deployments.

---

### Architecture
- **Frontend:** HTML5, CSS3, JavaScript
- **Backend API:** Python 3.11.9, Flask
- **AI/ML Engine:** TensorFlow Lite, MediaPipe, EfficientNet2
- **Image Processing:** OpenCV, NumPy

---

### Core Dependencies
* `tflite_runtime`
* `mediapipe` 
* `opencv-python`
* `flask` & `flask-cors` 