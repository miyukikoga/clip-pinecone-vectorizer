from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ImageItem:
    """画像アイテムを表すドメインモデル"""

    image_path: str
    image_name: str
    metadata: Dict[str, Any]
