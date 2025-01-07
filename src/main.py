import os
import time
from pathlib import Path
from dotenv import load_dotenv
from image_register.infrastructure.image_vectorizer import ImageVectorizer
from image_register.infrastructure.pinecone_client import PineconeClient
from image_register.repository.item_repository import ItemRepository
from image_register.service.image_register import ImageRegisterService


def main():
    load_dotenv()

    # プロジェクトルートパスの取得
    root_dir = Path(__file__).parent.parent
    data_file = root_dir / "data" / "items.json"

    # 依存オブジェクトの初期化
    vectorizer = ImageVectorizer()
    pinecone_client = PineconeClient(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name=os.getenv("PINECONE_INDEX_NAME"),
    )
    repository = ItemRepository(str(data_file))
    service = ImageRegisterService(vectorizer, pinecone_client)

    # 画像データの処理
    items = repository.load_all()
    for item in items:
        try:
            service.register_item(item)
            print(f"登録完了: {item.image_name}")
            time.sleep(0.5)
        except Exception as e:
            print(f"エラーが発生しました（{item.image_name}）: {str(e)}")
            continue


if __name__ == "__main__":
    main()
