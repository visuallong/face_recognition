from keras.models import load_model
import cv2
import numpy as np


mask_detector_path = r'storage/model/extended_model/mask_detector.h5'
model = load_model(mask_detector_path)

def mask_detect(pixels):
    pixels_resized = cv2.resize(pixels, (224,224))
    pixels_normalized = (pixels_resized.astype('float32') / 127.0) - 1
    pixels_input = np.expand_dims(pixels_normalized, axis=0)
    prediction = model.predict(pixels_input)
    max_prob = np.max(prediction[0])
    max_index = list(prediction[0]).index(max_prob)
    if max_index == 0:
        is_mask = True
    else:
        is_mask = False
    return is_mask