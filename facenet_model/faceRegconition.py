from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
import torch
import pickle


def infer_facenet(model, face, embeds, threshold=1, transform=None):
    detect_embeds = model(transform(face).unsqueeze(0))  # [1,512]
    norm_diff = detect_embeds - embeds
    norm_score = (norm_diff ** 2).sum(axis=1)
    idx = torch.argmin(norm_score)
    min_dist = norm_score[idx]
    print(min_dist)
    print(norm_score)
    if min_dist > threshold:
        return -1
    else:
        return idx


if __name__ == '__main__':
    model = InceptionResnetV1(
        classify=False,
        pretrained="vggface2"
    )
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    with open('class.pkl', 'rb') as f:
        classes = pickle.load(f)
    embeds = torch.load('embeddings.pth')
    print(embeds.shape)
    face = Image.open('Duc.jpg')
    face1 = Image.open('7.jpg')

    print(infer_facenet(model, face, embeds,transform=transform))
