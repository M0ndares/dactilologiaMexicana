import tensorflow as tf 
model = tf.keras.models.load_model('modelo/model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
model3 = converter.convert()
with open('modelo/model3.tflite', 'wb') as f:
    f.write(model3)