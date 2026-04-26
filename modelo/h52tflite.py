import tensorflow as tf 
model = tf.keras.models.load_model('model2.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
modelLite = converter.convert()
with open('model2.tflite', 'wb') as f:
    f.write(modelLite)