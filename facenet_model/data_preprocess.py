from py import process
from faceDetection import extract_face
import os 

DB_PATH = os.path.join("..", "images", "Data") 
PROCESSED_DB_PATH = os.path.join("..", "images", "processed_data") 


# try to create new db dir if not exists

try:
    os.mkdir(PROCESSED_DB_PATH)
except OSError:
    pass

people = os.listdir(DB_PATH)

for person in people:
    try:
        os.mkdir(os.path.join(PROCESSED_DB_PATH, person))
        for picture in os.listdir(os.path.join(DB_PATH, person)):
            try:
                img = extract_face(os.path.join(DB_PATH, person, picture))
                name = len(os.listdir(os.path.join(os.path.join(PROCESSED_DB_PATH), person)))
                img.save(os.path.join(PROCESSED_DB_PATH, person, '{}.jpg'.format(name)))
            except AttributeError:
                with open('log.txt', 'a', encoding='utf-8') as f:
                    f.writelines("can't find face in {} / {}\n".format(person, picture))
    except OSError:
        pass
