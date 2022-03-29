import tensorflow as tf
from keras import Model
import numpy as np
from keras.models import load_model
import os
import gdown
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D, AveragePooling2D, LocallyConnected2D
from keras.layers import Convolution2D, MaxPool2D, Input, Lambda, concatenate
from keras import Sequential
from keras import backend as K
import zipfile
from keras.layers import add, Concatenate, Add, PReLU
from tensorflow.python.keras.engine import training
import cv2
import time


feature_extraction_model_path = r'storage\model\feature_extraction_model\mobilefacenet.tflite'
interpreter = tf.lite.Interpreter(model_path=feature_extraction_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]['index']
input_shape = input_details[0]['shape']
embedding_index = output_details[0]['index']
height, width = input_shape[1:3]


def img_normalize(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean = face_pixels.mean()
    std  = face_pixels.std()
    face_pixels = (face_pixels - mean)/std
    return face_pixels


def feature_extraction(pixels):
	image = pixels
	base_width, base_height = pixels.shape[1], pixels.shape[0]
	image_resized = cv2.resize(image,(input_shape[1],input_shape[2]),interpolation=cv2.INTER_CUBIC)
	image_array = img_normalize(image_resized)
	input_data = np.expand_dims(image_array, axis=0)
	interpreter.set_tensor(input_index, input_data)
	interpreter.invoke()
	embedding = interpreter.get_tensor(embedding_index)
	return embedding[0]
  

# import cv2
# import time

# x= cv2.imread(r'C:\Trong\Projects\faceReg\faceReg\media\detectedFaces\2\2022-03-01_082218603567.png')
# # y= cv2.imread(r'C:\Trong\Projects\faceReg\faceReg\media\detectedFaces\2\2022-03-01_082213267423.png')
# y= cv2.imread(r'C:\Trong\Projects\faceReg\faceReg\media\detectedFaces\1\2022-03-01_082159122334.png')
# t1= time.process_time()
# feature=feature_extraction(x)
# audit_feature=feature_extraction(y)
# probability = euclideanDistance(feature,audit_feature)
# print(probability)
# t2= time.process_time()
# print(str(t2-t1))


