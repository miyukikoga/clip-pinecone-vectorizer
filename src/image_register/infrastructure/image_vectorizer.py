import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np


class ImageVectorizer:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def vectorize(self, image_path: str) -> np.ndarray:
        """画像をベクトル化する"""
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt", padding=True).to(
            self.device
        )

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        image_vector = image_features.numpy()
        normalized_vector = image_vector / np.linalg.norm(image_vector)
        return normalized_vector[0]
