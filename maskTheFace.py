import tensorflow as tf
import numpy as np
import cv2


interpreter = tf.lite.Interpreter(model_path=r'storage\model\face_landmark.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
floating_model = input_details[0]['dtype'] == np.float32
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

def get_face_landmark(pixels):
    image = pixels
    base_width, base_height = pixels.shape[1], pixels.shape[0]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # imH, imW = image.shape[:2]
    image_resized = cv2.resize(image_rgb, (width, height),interpolation=cv2.INTER_CUBIC)
    image_array = (np.float32(image_resized) - 0.0)/255.
    input_data = np.expand_dims(image_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    output_face_landmarks = interpreter.get_tensor(output_details[0]['index'])[0]
    output_face_flag = interpreter.get_tensor(output_details[1]['index'])[0]

    output_face_landmarks = tf.reshape(tensor=output_face_landmarks, shape=(468,3))
    face_landmark_x = output_face_landmarks[:, 0:1]
    face_landmark_y = output_face_landmarks[:, 1:2]
    face_landmark_z = output_face_landmarks[:, 2:3]

    return face_landmark_x.numpy(), face_landmark_y.numpy(), face_landmark_z.numpy(), image_resized

def mask_the_face(pixels):
    face_landmark_x, face_landmark_y, face_landmark_z = get_face_landmark(pixels)


a = cv2.imread(r'storage\imageBase\21-03-22-15-44-26\img-21-03-22-15-44-30.jpg')
# a = cv2.imread(r'C:\Users\TrongTN\Downloads\geoffrey-hinton.jpg')
face_landmark_x, face_landmark_y, face_landmark_z, output_face_landmarks = get_face_landmark(a)
for i, x in enumerate(face_landmark_x):
    x = x[0]
    x = int(x)
    y = face_landmark_y[i][0]
    y = int(y)
    a = cv2.putText(output_face_landmarks,'*',(x,y), cv2.FONT_HERSHEY_TRIPLEX, 1,(255,255,255),1,cv2.LINE_AA)
cv2.imshow('x',output_face_landmarks)
cv2.waitKey()
