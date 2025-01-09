from typing import cast
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from numpy.typing import NDArray


class ImageVectorizer:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def vectorize(self, image_path: str) -> NDArray[np.float32]:
        """画像をベクトル化する"""
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt", padding=True).to(
            self.device
        )

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # CPU上のNumPy配列に変換し、正規化
        numpy_array: NDArray[np.float32] = image_features.cpu().numpy()
        norm = np.linalg.norm(numpy_array, axis=1, keepdims=True)
        normalized_vector = (numpy_array / norm).astype(np.float32)
        return normalized_vector[0] # type: ignore
