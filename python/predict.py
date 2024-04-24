import torch
import torch.nn as nn
from torchvision import transforms
import os
import torchvision
from cnn import CNN
import io
from PIL import Image

def predict(image_bytes):
    device = "cpu"

    custom_image = Image.open(io.BytesIO(image_bytes))

    custom_image = custom_image / 255.0

    IMAGE_SIZE = (224, 224)

    custom_image_transformed = transforms.Compose([transforms.Resize(IMAGE_SIZE),])

    model = CNN()
    model.load_state_dict(torch.load("cat-dog-model.pth"))
    model.eval()
    with torch.no_grad():
        custom_image_pred = model(custom_image_transformed.unsqueeze(dim=0).to(device))
        custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
        custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
        class_names = ["cats", "dogs"]
        custom_image_pred_class = class_names[custom_image_pred_label]
        return custom_image_pred_class