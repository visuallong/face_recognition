import numpy as np
from featureExtraction import feature_extraction
import os
import time


feature_ds_path = r'storage\model\feature_ds\feature_ds.npz'

def cosine_similarity_classify(face_pixels):
    t1_start = time.process_time()
    if os.path.exists(feature_ds_path):
        feature_ds = np.load(feature_ds_path)
    else:
        return None, None
    probability_list = []
    audit_feature = feature_extraction(face_pixels)
    for feature in feature_ds['feature']:
        probability = np.dot(audit_feature, feature)/(np.linalg.norm(audit_feature)*np.linalg.norm(feature))
        probability_list.append(probability)
    max_prob = np.max(probability_list)
    max_index = probability_list.index(max_prob)
    label = feature_ds['label'][max_index]
    print('Label: %s (%.3f)' % (label, max_prob))
    t1_stop = time.process_time()
    print("Recognize face time: " + str(t1_stop-t1_start))
    return label, max_prob*100


import math 

def euclideanDistance(e1, e2):
  sum = 0.0
  for i,e1_ in enumerate(e1):
    sum += math.pow((e1[i] - e2[i]), 2)
  return math.sqrt(sum)

def compare(face_pixels):
    t1_start = time.process_time()
    if os.path.exists(feature_ds_path):
        feature_ds = np.load(feature_ds_path)
    else:
        return None, None
    dist_list = []
    audit_feature = feature_extraction(face_pixels)
    for feature in feature_ds['feature']:
        dist = euclideanDistance(audit_feature, feature)
        dist_list.append(dist)
    min_dist = np.min(dist_list)
    min_index = dist_list.index(min_dist)
    label = feature_ds['label'][min_index]
    print('Label: %s (%.3f)' % (label, min_dist))
    t1_stop = time.process_time()
    print("Recognize face time: " + str(t1_stop-t1_start))
    return label, min_dist