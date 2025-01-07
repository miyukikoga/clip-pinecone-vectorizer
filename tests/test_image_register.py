import pytest
from unittest.mock import Mock
import numpy as np
from pathlib import Path
from image_register.domain.models import ImageItem
from image_register.infrastructure.image_vectorizer import ImageVectorizer
from image_register.infrastructure.pinecone_client import PineconeClient
from image_register.service.image_register import ImageRegisterService
from image_register.repository.item_repository import ItemRepository

class TestImageRegistration:
    """画像登録システムの仕様"""

    @pytest.fixture
    def sample_item(self) -> ImageItem:
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
    def mock_vectorizer(self):
        """画像ベクトル化モジュールのモック"""
        vectorizer = Mock(spec=ImageVectorizer)
        vectorizer.vectorize.return_value = np.array([0.1, 0.2, 0.3])
        return vectorizer

    @pytest.fixture
    def mock_pinecone_client(self):
        """Pinecone クライアントのモック"""
        return Mock(spec=PineconeClient)

    def test_image_registration_flow(self, sample_item, mock_vectorizer, mock_pinecone_client):
        """
        シナリオ: 画像とメタデータを登録する
        前提: 
            - 画像ファイルとメタデータが用意されている
            - ベクトル化サービスが利用可能
            - Pinecone が利用可能

        もし: 画像を登録しようとすると
        ならば: 
            - 画像がベクトル化される
            - メタデータと共に Pinecone に保存される
            - 画像名がベクトル ID として使用される
        """
        # テスト対象のサービスを初期化
        service = ImageRegisterService(mock_vectorizer, mock_pinecone_client)

        # 画像を登録
        service.register_item(sample_item)

        # 画像のベクトル化が行われたことを確認
        mock_vectorizer.vectorize.assert_called_once_with(sample_item.image_path)

        # Pinecone への保存が正しく行われたことを確認
        mock_pinecone_client.upsert.assert_called_once_with(
            vector_id=sample_item.image_name,
            vector=mock_vectorizer.vectorize.return_value,
            metadata=sample_item.metadata
        )

class TestItemRepository:
    """画像アイテムリポジトリの仕様"""

    @pytest.fixture
    def sample_json_path(self, tmp_path):
        """テスト用の JSON ファイル"""
        json_file = tmp_path / "test_items.json"
        json_file.write_text('''
        {
            "items": [
                {
                    "image_path": "./images/test_image.jpg",
                    "image_name": "test_image_001",
                    "metadata": {
                        "title": "テスト画像",
                        "category": "テストカテゴリ",
                        "tags": ["テスト", "サンプル"],
                        "description": "テスト用の画像データ"
                    }
                }
            ]
        }
        ''')
        return str(json_file)

    def test_load_items_from_json(self, sample_json_path):
        """
        シナリオ: JSON ファイルから画像アイテムデータを読み込む
        前提: 
            - 正しい形式の JSON ファイルが存在する
        
        もし: データを読み込もうとすると
        ならば: 
            - JSON データが ImageItem オブジェクトのリストとして読み込まれる
            - 各フィールドが正しく設定される
        """
        repository = ItemRepository(sample_json_path)
        items = repository.load_all()

        assert len(items) == 1
        item = items[0]
        assert isinstance(item, ImageItem)
        assert item.image_path == "./images/test_image.jpg"
        assert item.image_name == "test_image_001"
        assert item.metadata["title"] == "テスト画像"
        assert "tags" in item.metadata
        assert len(item.metadata["tags"]) == 2

class TestImageVectorizer:
    """画像ベクトル化モジュールの仕様"""

    def test_vectorize_image(self, tmp_path):
        """
        シナリオ: 画像をベクトル化する
        前提: 
            - 有効な画像ファイルが存在する
            - CLIP モデルが利用可能
        
        もし: 画像をベクトル化しようとすると
        ならば: 
            - 正規化された特徴ベクトルが生成される
            - ベクトルの次元数が正しい
        """
        # Note: 実際のモデルを使用するため、統合テストとして実装する必要があります
        pass 