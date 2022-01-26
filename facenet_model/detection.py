from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image
import math

detector = MTCNN(thresholds=[0.7, 0.7, 0.8])


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
    a = math.dist(np.asarray(left_eye), np.asarray(point_3rd))
    b = math.dist(np.asarray(right_eye), np.asarray(point_3rd))
    c = math.dist(np.asarray(right_eye), np.asarray(left_eye))

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
    boxes, prob, keypoint = detector.detect(pixels, landmarks=True)
    if prob[0] >= threshold:
        x1, y1, x2, y2 = boxes[0]
        face = pixels[int(y1):int(y2), int(x1):int(x2)]
        left_eye = keypoint[0][0]
        right_eye = keypoint[0][1]
        if align:
            img = alignment(face, left_eye, right_eye)
        else:
            img = Image.fromarray(face)
        img = img.resize(required_size)
        return img

if __name__ == '__main__':
    extract_face('1.jpg')
