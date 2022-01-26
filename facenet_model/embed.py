from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import os
import pickle
transform = transforms.Compose([
    transforms.ToTensor(),
])

PROCESSED_DB_PATH = os.path.join('..', 'images', 'processed_data')
classes = os.listdir(PROCESSED_DB_PATH)
model = InceptionResnetV1(pretrained="vggface2").eval()
with open('class.pkl', 'wb') as f:
    pickle.dump(classes, f)
if __name__ == '__main__':
    embeddings = []
    for label, person in enumerate(classes):
        embeds = []
        for picture in os.listdir(os.path.join(PROCESSED_DB_PATH, person)):
            face = Image.open(os.path.join(PROCESSED_DB_PATH, person, picture))
            with torch.no_grad():
                embed = model(transform(face).unsqueeze(0))
                embeds.append(embed)
        if len(embeds):
            # print(torch.cat(embeds).shape)
            embedding = torch.cat(embeds).mean(0).unsqueeze(0)
            embeddings.append(embedding)
    embeddings = torch.cat(embeddings)
    torch.save(embeddings, "embeddings.pth")
    with open('class.pkl', 'rb') as f:
        print(pickle.load(f)) 
