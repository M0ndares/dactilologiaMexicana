import tensorflow as tf 
model = tf.keras.models.load_model('modelo/daModel.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
model3 = converter.convert()
with open('modelo/model.tflite', 'wb') as f:
    f.write(model3)