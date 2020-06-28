import os
import matplotlib.image as mpimg
import numpy as np
import torch
from torchvision import transforms
from predictions.model import DRUnet
from PIL import Image

H = 272
W = 400

MODEL_PATH = "weights/drunet_best.pth"


class Segmentation:
    def __init__(self, image_path):
        self._transformer = transforms.Compose(
            [transforms.Resize((H, W)),
             transforms.ToTensor(),
             transforms.Normalize(0.5, 0.5)]
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DRUnet()
        self.model.to(device)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        self.model.load_state_dict(checkpoint['network'])
        self.model.eval()
        self.image_path = image_path

    def segment(self):
        image = mpimg.imread(self.image_path)
        image = Image.fromarray(image)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_tensor = self._transformer(image)
        c, h, w = image_tensor.shape
        image_tensor = image_tensor.view(1, c, h, w)

        with torch.no_grad():
            self.model.to(device)
            mask_tensor = self.model(image_tensor.to(device))
            mask = mask_tensor.cpu().numpy()

        mask = np.squeeze(mask)
        mask = np.argmax(mask, axis=0)
        return mask
