import cv2
import torch
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reid_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
reid_model.fc = torch.nn.Identity()
reid_model.eval()
reid_model = reid_model.to(device)

def extract_features(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    cropped = frame[y1:y2, x1:x2]
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        return None
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    cropped = cv2.resize(cropped, (224, 224))
    tensor = torch.tensor(np.transpose(cropped, (2, 0, 1)), dtype=torch.float32) / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        features = reid_model(tensor).cpu().numpy().flatten()
    return features / np.linalg.norm(features)

def save_features_db(features_db, filename="features_db.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(features_db, f)

def load_features_db(filename="features_db.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)