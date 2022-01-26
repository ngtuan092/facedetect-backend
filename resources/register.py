import base64
from flask_restful import Resource, reqparse
import numpy as np
import os
from io import BytesIO
from PIL import Image
import pickle
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from facenet_model.detection import extract_face
parser = reqparse.RequestParser()
parser.add_argument('name')
parser.add_argument('image_base64')
model = InceptionResnetV1(
    classify=False,
    pretrained="vggface2"
)
model.eval()
transform = transforms.Compose([
    transforms.ToTensor(),
])


class FaceRegister(Resource):
    def post(self):
        args = parser.parse_args()
        name = args['name']
        image_base64 = args['image_base64']
        with open('facenet_model/class.pkl', 'rb') as f:
            classes = pickle.load(f)
        embeddings = torch.load('facenet_model/embeddings.pth')
        new_label_path = os.path.join('images', 'processed_data', name)
        if name not in classes:
            classes.append(name)
            with open('facenet_model/class.pkl', 'wb') as f:
                pickle.dump(classes, f)
            try:
                os.mkdir(new_label_path)
            except OSError:
                pass
        try:
            img = BytesIO(base64.b64decode(image_base64[22:]))
            face = extract_face(img)
            if face is not None:
                lendir = len(os.listdir(new_label_path))
                face.save(os.path.join(new_label_path, '{}.png'.format(lendir)))
                with torch.no_grad():
                    embed = model(transform(face).unsqueeze(0))
                class_id = classes.index(name)
                try:
                    embeddings[class_id] = (embeddings[class_id] * lendir + embed) / (lendir + 1)
                except Exception as e:
                    print(e)
                    embeddings = torch.cat((embeddings, embed), dim=0)
                torch.save(embeddings, 'facenet_model/embeddings.pth')
                return 'ok'
            else:
                return 'not ok'
        except Exception as e:
            print(e)
            return 404

