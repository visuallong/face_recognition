from keras import Sequential, layers
import numpy as np
import os
import tensorflow as tf
import PIL
import shutil

imageBase_dir = r'storage\imageBase'
imageTrain_dir = r'storage\imageTrain'


def data_train_enriched():   
    if os.path.exists(imageTrain_dir):
        shutil.rmtree(imageTrain_dir)
    data_augmentation = Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),])
    for label in os.listdir(imageBase_dir):
        sub_dir = os.path.join(imageBase_dir,label)
        for j, image in enumerate(os.listdir(sub_dir)):
            image_dir = os.path.join(sub_dir,image)
            base_image = PIL.Image.open(image_dir, 'r')
            base_image = base_image.resize((224, 224))
            save_dir = os.path.join(imageTrain_dir,'{}'.format(label))
            if not os.path.exists(imageTrain_dir):
                os.mkdir(imageTrain_dir)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            base_image.save(os.path.join(save_dir,'{}_0.jpg'.format(j)))
            for i in range(1, 10):
                augmented_image = data_augmentation(tf.expand_dims(base_image, 0), training=True)
                augmented_image = PIL.Image.fromarray(augmented_image[0].numpy().astype(np.uint8))
                augmented_image.save(os.path.join(save_dir,'{}_{}.jpg'.format(j,i)))


import cv2
import json


def create_mask_dataset():
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
    dataset =[]
    dataset_names = []
    data_augmentation = Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),])
    for sub_fld in os.listdir(imageBase_dir):
        sub_fld_path = os.path.join(imageBase_dir,sub_fld)
        name = names[fld_list.index(sub_fld)]
        for j, file in enumerate(os.listdir(sub_fld_path)):
            file_path = os.path.join(sub_fld_path,file)
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                base_img = cv2.imread(file_path)
                base_img = cv2.resize(base_img, (224, 224))
                dataset.append(tf.expand_dims(base_img, 0))
                dataset_names.append(name)
                for i in range(1, 10):
                    augmented_img = data_augmentation(tf.expand_dims(base_img, 0), training=True)
                    dataset.append(augmented_img)
                    dataset_names.append(name)

# create_mask_dataset()