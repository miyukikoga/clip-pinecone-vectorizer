from typing import cast
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from numpy.typing import NDArray
from rembg import remove
from io import BytesIO


class ImageVectorizer:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def remove_background(self, image: Image.Image) -> Image.Image:
        """画像の背景を除去する"""
        # PILイメージをバイト列に変換
        img_byte = BytesIO()
        image.save(img_byte, format="PNG")
        img_byte = img_byte.getvalue()

        # 背景除去
        output = remove(img_byte)

        # バイト列からPILイメージに戻す
        removed_bg = Image.open(BytesIO(output))

        # RGBモードに変換して返す
        return removed_bg.convert("RGB")

    def vectorize(self, image_path: str) -> NDArray[np.float32]:
        """画像をベクトル化する"""
        # 画像を読み込み
        image = Image.open(image_path)

        # 背景除去
        image = self.remove_background(image)

        # CLIP用の前処理
        inputs = self.processor(images=image, return_tensors="pt", padding=True).to(
            self.device
        )

        # 特徴量抽出
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # ベクトルの正規化
        image_vector = image_features.cpu().numpy()
        normalized_vector = image_vector / np.linalg.norm(image_vector)
        return normalized_vector[0] # type: ignore
