from ..domain.models import ImageItem
from ..infrastructure.image_vectorizer import ImageVectorizer
from ..infrastructure.pinecone_client import PineconeClient


class ImageRegisterService:
    def __init__(self, vectorizer: ImageVectorizer, pinecone_client: PineconeClient) -> None:
        self.vectorizer = vectorizer
        self.pinecone_client = pinecone_client

    def register_item(self, item: ImageItem) -> None:
        """画像アイテムを登録"""
        vector = self.vectorizer.vectorize(item.image_path)
        self.pinecone_client.upsert(
            vector_id=item.image_name, vector=vector, metadata=item.metadata
        )
