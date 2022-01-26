from flask_restful import Resource, reqparse
from io import BytesIO
from facenet_model.faceRegconition import infer_facenet
from facenet_model.detection import extract_face
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import torch
import base64
import pickle
model = InceptionResnetV1(
    classify=False,
    pretrained="vggface2"
)
model.eval()
transform = transforms.Compose([
    transforms.ToTensor(),
])
parser = reqparse.RequestParser()
parser.add_argument('image_base64')


class FaceRecognition(Resource):
    def post(self):
        args = parser.parse_args()
        image_base64 = args['image_base64']
        try:
            img = BytesIO(base64.b64decode(image_base64[22:]))
            face = extract_face(img)
            embeds = torch.load('facenet_model/embeddings.pth')
            with open('facenet_model/class.pkl', 'rb') as f:
                classes = pickle.load(f)
            if face is not None:
                idx = infer_facenet(model, face, embeds,
                                    transform=transforms.ToTensor())
                print(idx)
                if idx == -1:
                    return {"result": {"name": "unknown"}}, 200
                return {"result": {"name": classes[idx]}}, 200
            else:
                return {"result": {"msg": "Khong thay mat"}}, 201
        except Exception as e:
            print(e)
            return 404
