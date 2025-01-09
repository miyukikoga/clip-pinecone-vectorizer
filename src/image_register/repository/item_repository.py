import json
from typing import List, Any, Dict
from ..domain.models import ImageItem


class ItemRepository:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load_all(self) -> List[ImageItem]:
        """JSONファイルから画像アイテムデータを読み込む"""
        with open(self.file_path, "r", encoding="utf-8") as f:
            data: Dict[str, List[Dict[str, Any]]] = json.load(f)
        return [
            ImageItem(
                image_path=item["image_path"],
                image_name=item["image_name"],
                metadata=item.get("metadata", {}),
            )
            for item in data["items"]
        ]
