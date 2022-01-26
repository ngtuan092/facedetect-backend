from mtcnn import MTCNN
import math
from PIL import Image
import numpy as np
detector = MTCNN()


def load_image(path):
    img = Image.open(path)
    img = img.convert('RGB')
    pixels = np.asarray(img)
    return pixels


def alignment(face, left_eye, right_eye):
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock
    a = math.dist(np.array(left_eye), np.array(point_3rd))
    b = math.dist(np.array(right_eye), np.array(point_3rd))
    c = math.dist(np.array(right_eye), np.array(left_eye))

    # -----------------------

    # apply cosine rule

    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation

        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        face = Image.fromarray(face)
        face = face.rotate(direction * angle)
    return face


def extract_face(path, threshold=0.9, required_size=(160, 160), align=True):
    pixels = load_image(path)
    detections = detector.detect_faces(pixels)
    for detection in detections:
        score = detection['confidence']
        if score > threshold:
            x, y, w, h = detection["box"]
            keypoints = detection["keypoints"]
            left_eye = keypoints["left_eye"]
            right_eye = keypoints["right_eye"]
            face = pixels[int(y):int(y+h), int(x):int(x+w)]
            if align:
                image = alignment(face, left_eye, right_eye)
            else:
                image = Image.fromarray(face)
            image = image.resize(required_size)
            return image




