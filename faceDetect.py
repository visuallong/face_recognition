import os
import cv2
import numpy as np
import time
import tensorflow as tf

PATH = os.getcwd()

# with open(r'storage\model\ssd_mobilenetv2\lbl.txt', 'r') as f:
#     labels = [line.strip() for line in f.readlines()]

# interpreter = tf.lite.Interpreter(model_path=r'storage\model\ssd_mobilenetv2\model4.tflite')  #experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# height = input_details[0]['shape'][1]
# width = input_details[0]['shape'][2]

# def face_detector(pixels):
#     t1_start = time.process_time()
#     image = pixels
#     base_width, base_height = pixels.shape[1], pixels.shape[0]
#     scale_y = base_height/height
#     scale_x = base_width/width
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     imH, imW = image.shape[:2]
#     image_resized = cv2.resize(image_rgb, (width, height),interpolation=cv2.INTER_CUBIC)
#     image_array = np.float32(image_resized)/255.
#     input_data = np.expand_dims(image_array, axis=0)

#     interpreter.set_tensor(input_details[0]['index'],input_data)
#     interpreter.invoke()

#     boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
#     classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
#     scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
#     faces = []
#     faces_location = []
#     is_mask_list = []
#     # print(boxes)
#     for i in range(len(scores)):
#         if ((scores[i] > 0.5) and (scores[i] <= 1.0)):
#             ymin = int(max(1,(boxes[i][0] * imH))* scale_y)
#             xmin = int(max(1,(boxes[i][1] * imW))* scale_x)
#             ymax = int(min(imH,(boxes[i][2] * imH))* scale_y)
#             xmax = int(min(imW,(boxes[i][3] * imW))* scale_x)
#             bb_width = xmax-xmin
#             bb_height = ymax-ymin
#             offset_x = 0
#             offset_y = 0
#             if classes[i] == 0.0:
#                 is_mask = True
#             elif classes[i] == 1.0:
#                 is_mask = False
#             if bb_width > bb_height:
#                 offset_y = round((bb_width - bb_height)/2)
#                 bb_height = bb_width
#             elif bb_width < bb_height:
#                 offset_x = round((bb_height - bb_width)/2)
#                 bb_width = bb_height
#             try:
#                 face = pixels[ymin-offset_y:ymin-offset_y+bb_height,xmin-offset_x:xmin-offset_x+bb_width]
#                 face = cv2.resize(face, (224, 224), interpolation = cv2.INTER_CUBIC)
#                 faces.append(face)
#                 faces_location.append((xmin-offset_x,ymin-offset_y,bb_width,bb_height))
#                 is_mask_list.append(is_mask)
#             except Exception as e:
#                 print(e)
#     if faces_location == []:
#         print("No face detected")
#     t1_stop = time.process_time()
#     print("Detect face(s) time: " + str(t1_stop-t1_start))
#     return faces, faces_location, is_mask_list


from nms import nms
import torch
from keras.models import load_model

interpreter = tf.lite.Interpreter(model_path=r'storage\model\second_plan\face_detection_short_range.tflite')
interpreter.allocate_tensors()
mask_detector = load_model(r'storage\model\second_plan\mask_detector.h5')

# (reference: modules/face_detection/face_detection_short_range_common.pbtxt)
SSD_OPTIONS_SHORT = {
    'num_layers': 4,
    'input_size_height': 128,
    'input_size_width': 128,
    'anchor_offset_x': 0.5,
    'anchor_offset_y': 0.5,
    'strides': [8, 16, 16, 16],
    'interpolated_scale_aspect_ratio': 1.0
}

# (reference: modules/face_detection/face_detection_full_range_common.pbtxt)
SSD_OPTIONS_FULL = {
    'num_layers': 1,
    'input_size_height': 192,
    'input_size_width': 192,
    'anchor_offset_x': 0.5,
    'anchor_offset_y': 0.5,
    'strides': [4],
    'interpolated_scale_aspect_ratio': 0.0
}

def ssd_generate_anchors(opts: dict) -> np.ndarray:
    """This is a trimmed down version of the C++ code; all irrelevant parts
    have been removed.
    (reference: mediapipe/calculators/tflite/ssd_anchors_calculator.cc)
    """
    layer_id = 0
    num_layers = opts['num_layers']
    strides = opts['strides']
    assert len(strides) == num_layers
    input_height = opts['input_size_height']
    input_width = opts['input_size_width']
    anchor_offset_x = opts['anchor_offset_x']
    anchor_offset_y = opts['anchor_offset_y']
    interpolated_scale_aspect_ratio = opts['interpolated_scale_aspect_ratio']
    anchors = []
    while layer_id < num_layers:
        last_same_stride_layer = layer_id
        repeats = 0
        while (last_same_stride_layer < num_layers and
               strides[last_same_stride_layer] == strides[layer_id]):
            last_same_stride_layer += 1
            # aspect_ratios are added twice per iteration
            repeats += 2 if interpolated_scale_aspect_ratio == 1.0 else 1
        stride = strides[layer_id]
        feature_map_height = input_height // stride
        feature_map_width = input_width // stride
        for y in range(feature_map_height):
            y_center = (y + anchor_offset_y) / feature_map_height
            for x in range(feature_map_width):
                x_center = (x + anchor_offset_x) / feature_map_width
                for _ in range(repeats):
                    anchors.append((x_center, y_center))
        layer_id = last_same_stride_layer
    return np.array(anchors, dtype=np.float32)

input_index = interpreter.get_input_details()[0]['index']
input_shape = interpreter.get_input_details()[0]['shape']
bbox_index = interpreter.get_output_details()[0]['index']
score_index = interpreter.get_output_details()[1]['index']
anchors = ssd_generate_anchors(SSD_OPTIONS_SHORT)
RAW_SCORE_LIMIT = 80
height, width = input_shape[1:3]
MIN_SCORE = 0.5

def face_detector(pixels):
    t1_start = time.process_time()
    # pixels = cv2.resize(pixels,(1000,1000),interpolation=cv2.INTER_CUBIC)
    image = pixels
    base_width, base_height = pixels.shape[1], pixels.shape[0]
    # scale_x = base_width/width
    # scale_y = base_height/height
    image = cv2.resize(image,(300,300),interpolation=cv2.INTER_CUBIC)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # imH, imW = image.shape[:2]
    image_resized = cv2.resize(image_rgb, (width, height))
    image_array = (image_resized/255.).astype('float32')
    input_data = np.expand_dims(image_array, axis=0)

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    raw_boxes = interpreter.get_tensor(bbox_index)
    raw_scores = interpreter.get_tensor(score_index)

    boxes = decode_boxes(raw_boxes)
    scores = get_sigmoid_scores(raw_scores)

    def is_valid(box: np.ndarray) -> bool:
        return np.all(box[1] > box[0])

    score_above_threshold = scores > MIN_SCORE
    filtered_boxes = boxes[np.argwhere(score_above_threshold)[:, 1], :]
    filtered_scores = scores[score_above_threshold]
    bboxes = []
    for box, score in zip(filtered_boxes, filtered_scores):
        if is_valid(box):
            data = box.reshape(-1, 2)
            xmin, ymin = data[0]
            xmax, ymax = data[1]
            bboxes.append([xmin, ymin, xmax, ymax, score])
    P = torch.tensor(bboxes)
    bb = nms(P, 0.5)
    # x1 = int(bb[0][0]*imW)
    # y1 = int(bb[0][1]*imH)
    # x2 = int(bb[0][2]*imW)
    # y2 = int(bb[0][3]*imH)
    # # image = cv2.rectangle(image, (x1,y1), (x2,y2), (255, 0, 0), 2)
    # image = image[y1:y2,x1:x2]
    # cv2.imshow('x',image)
    # cv2.waitKey()
    faces = []
    faces_location = []
    is_mask_list = []
    # print(boxes)
    for i,b in enumerate(bb):
        xmin = int(max(1,(bb[i][0] * base_width)))
        ymin = int(max(1,(bb[i][1] * base_height)))
        xmax = int(min(base_width,(bb[i][2] * base_width)))
        ymax = int(min(base_height,(bb[i][3] * base_height)))
        bb_width = xmax-xmin
        bb_height = ymax-ymin
        offset_x = 0
        offset_y = 0
        if bb_width > bb_height:
            offset_y = round((bb_width - bb_height)/2)
            bb_height = bb_width
        elif bb_width < bb_height:
            offset_x = round((bb_height - bb_width)/2)
            bb_width = bb_height
        try:
            face = pixels[ymin-offset_y:ymin-offset_y+bb_height,xmin-offset_x:xmin-offset_x+bb_width]
            # face = pixels[ymin:ymax,xmin:xmax]
            face = cv2.resize(face, (224, 224), interpolation = cv2.INTER_CUBIC)
            face_sample = np.expand_dims(face, axis=0)
            mask_prob = mask_detector.predict(face_sample)
            max_prob_index= np.argmax(mask_prob)
            if max_prob_index == 0:
                is_mask = True
            else:
                is_mask = False
            faces.append(face)
            faces_location.append((xmin-offset_x,ymin-offset_y,bb_width,bb_height))
            is_mask_list.append(is_mask)
        except Exception as e:
            print(e)
    if faces_location == []:
        print("No face detected")
    t1_stop = time.process_time()
    print("Detect face(s) time: " + str(t1_stop-t1_start))
    return faces, faces_location, is_mask_list


def decode_boxes(raw_boxes: np.ndarray) -> np.ndarray:
    """Simplified version of
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    """
    # width == height so scale is the same across the board
    scale = height
    num_points = raw_boxes.shape[-1] // 2
    # scale all values (applies to positions, width, and height alike)
    boxes = raw_boxes.reshape(-1, num_points, 2) / scale
    # adjust center coordinates and key points to anchor positions
    boxes[:, 0] += anchors
    for i in range(2, num_points):
        boxes[:, i] += anchors
    # convert x_center, y_center, w, h to xmin, ymin, xmax, ymax
    center = np.array(boxes[:, 0])
    half_size = boxes[:, 1] / 2
    boxes[:, 0] = center - half_size
    boxes[:, 1] = center + half_size
    return boxes

def get_sigmoid_scores(raw_scores: np.ndarray) -> np.ndarray:
    """Extracted loop from ProcessCPU (line 327) in
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    """
    # just a single class ("face"), which simplifies this a lot
    # 1) thresholding; adjusted from 100 to 80, since sigmoid of [-]100
    #    causes overflow with IEEE single precision floats (max ~10e38)
    raw_scores[raw_scores < -RAW_SCORE_LIMIT] = -RAW_SCORE_LIMIT
    raw_scores[raw_scores > RAW_SCORE_LIMIT] = RAW_SCORE_LIMIT
    # 2) apply sigmoid function on clipped confidence scores
    return sigmoid(raw_scores)

def sigmoid(data: np.ndarray) -> np.ndarray:
    """Return sigmoid activation of the given data
    Args:
        data (ndarray): Numpy array containing data
    Returns:
        (ndarray) Sigmoid activation of the data with element range (0,1]
    """
    return 1 / (1 + np.exp(-data))


# x = cv2.imread(r'C:\Trong\python\st\data\A.Long\long1.jpg')
# faces, faces_location, is_mask_list = face_detector(x)
# for face in faces:
#     cv2.imshow('x',face)
#     cv2.waitKey()
# print(is_mask_list)

