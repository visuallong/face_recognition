import model_
from keras import Model
import numpy as np
from keras.models import load_model
import os


def img_normalize(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean = face_pixels.mean()
    std  = face_pixels.std()
    face_pixels = (face_pixels - mean)/std
    return face_pixels

# vgg_face_model_path = r'storage\model\vgg_face_weights.h5'
# if os.path.exists(vgg_face_model_path) is False:
# vgg_face_model = model_.vgg_face_model()
# vgg_face_model.load_weights(vgg_face_model_path)


# from keras.applications.mobilenet_v2 import MobileNetV2

# base_model = MobileNetV2(
#     # input_shape=(224,224,3),
#     include_top=True,
#     weights="imagenet",
#     # input_tensor=None,
#     # pooling=None,
#     # classes=1000,
#     # classifier_activation="softmax",
# )
# base_model = vgg_face_model
base_model = load_model('storage\model\landmark_model\model.h5')
x = base_model.layers[-2].output
model = Model(inputs=base_model.input, outputs=x)
# model.summary()

def feature_extraction(face_pixels):
    face_pixels = img_normalize(face_pixels)
    samples = np.expand_dims(face_pixels,axis=0)
    yhat = model.predict(samples)
    embedding = yhat[0]
    return embedding