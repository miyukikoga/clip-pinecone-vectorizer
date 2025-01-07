import os
import sys
import pytest
from unittest.mock import Mock
import numpy as np
from image_register.domain.models import ImageItem
from image_register.infrastructure.image_vectorizer import ImageVectorizer
from image_register.infrastructure.pinecone_client import PineconeClient

# src ディレクトリへのパスを追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# 共通フィクスチャ
@pytest.fixture
def sample_item() -> ImageItem:
    """テスト用の画像アイテムデータ"""
    return ImageItem(
        image_path="./images/test_image.jpg",
        image_name="test_image_001",
        metadata={
            "title": "テスト画像",
            "category": "テストカテゴリ",
            "tags": ["テスト", "サンプル"],
            "description": "テスト用の画像データ"
        }
    )

@pytest.fixture
def mock_vectorizer():
    """画像ベクトル化モジュールのモック"""
    vectorizer = Mock(spec=ImageVectorizer)
    vectorizer.vectorize.return_value = np.array([0.1, 0.2, 0.3])
    return vectorizer

@pytest.fixture
def mock_pinecone_client():
    """Pinecone クライアントのモック"""
    return Mock(spec=PineconeClient) 