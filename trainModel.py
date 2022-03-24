from keras import Model
from dataPreparation import data_train_enriched
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D
from keras.layers import Convolution2D, MaxPool2D, Input, Lambda, concatenate
from featureExtraction import img_normalize
import json
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping


def read_user_json():
    names = []
    fld_list = []
    with open(r'storage\something\users.json', encoding='utf-8') as json_file:
        data = json.load(json_file)
        users = data['users']
        for user in users:
            name = user['name']
            fld = user['fld_name']
            fld_list.append(fld)
            names.append(name)
    return names, fld_list


imageTrain_dir = r'storage\imageTrain'
names_path = r'storage\model\landmark_model\names.npy'

def preprocessing():
    names_, fld_list = read_user_json()
    imgs_norm = []
    labels = []
    names = []
    for fld in os.listdir(imageTrain_dir):
        sub_dir = os.path.join(imageTrain_dir,fld)
        for image in os.listdir(sub_dir):
            img_path = os.path.join(sub_dir,image)
            img = cv2.imread(img_path)
            img_norm = img_normalize(img)
            imgs_norm.append(img_norm)                   
            names.append(names_[fld_list.index(fld)])
    # imgs_norm=np.expand_dims(imgs_norm,axis=0)
    le = LabelEncoder()
    labels = le.fit_transform(names)
    np.save(names_path, le.classes_)
    return imgs_norm, labels, names


def train_model_landmark(batch_size=24,lr=0.001,epochs=20):

    data_train_enriched()

    X, y, names = preprocessing()
    num_class = len(list(dict.fromkeys(names)))
    X, y = shuffle(X, y)
    X = tf.stack(X)

    base_model = MobileNetV2(
        # input_shape=None,
        # alpha=1.0,
        # include_top=True,
        weights="imagenet",
        # input_tensor=None,
        # pooling=None,
        # classes=1000,
        # classifier_activation="softmax",
        )
    base_model.trainable = False
    x = base_model.layers[-2].output
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_class, activation='softmax')(x)
    final_model = Model(inputs=base_model.input, outputs=x)
    model_path = r'storage\model\landmark_model\model.h5'
    if os.path.exists(model_path):
        final_model.load_weights(model_path)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    if num_class > 2:
        loss = 'sparse_categorical_crossentropy'
    elif num_class == 2:
        loss = 'binary_crossentropy'
    else:
        return
    final_model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
    checkpoint = ModelCheckpoint(model_path, monitor='accuracy',verbose=1,save_weights_only=False,save_best_only=True,mode='max')
    callbacks_list = [checkpoint]
    final_model.fit(
        x=X,
        y=y,
        batch_size = batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks_list,
        validation_split=0.2,
        )

# train_model_landmark()
