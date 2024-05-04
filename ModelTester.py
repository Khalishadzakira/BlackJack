from pydoc import classname
import tensorflow as tf
import cv2
import numpy as np
from keras.models import load_model

modelkartu = load_model("ModelKartu")
classes = ['10', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'J', 'K', 'Q']

image = cv2.imread ("/Users/khalishadzakira/Documents/PCV/FP v2.0/Dataset_fix/2/0.0.jpg")
image = tf.keras.utils.img_to_array(image)
image = np.expand_dims(image, axis = 0)

prediction = modelkartu.predict(image)
print (classes[np.argmax(prediction[0])])