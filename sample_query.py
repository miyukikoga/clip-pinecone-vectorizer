import os
from pinecone import Pinecone
from dotenv import load_dotenv
from src.image_register.infrastructure.image_vectorizer import ImageVectorizer


def search_similar_images(vector, top_k=5):
    """
    ベクトルを使用してPinecone内の類似画像を検索する関数。
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    # Pineconeで直接類似度検索を実行
    response = index.query(
        namespace="ns1",
        vector=vector.tolist(),
        top_k=top_k,
        include_values=True,
        include_metadata=True,
    )

    return response.matches


def main():
    load_dotenv()

    # ImageVectorizerの初期化
    vectorizer = ImageVectorizer()

    # 画像ファイルのパスを指定
    image_path = "./images/***.png"

    try:
        # 画像をベクトル化
        vector = vectorizer.vectorize(image_path)

        # 類似画像を検索
        similar_images = search_similar_images(vector, top_k=5)

        # 検索結果の表示
        print("\n類似商品の検索結果:")
        for idx, match in enumerate(similar_images, 1):
            print(f"\n{idx}位:")
            print(f"ID: {match['id']}")
            print(f"スコア: {match['score']:.4f}")
            if "metadata" in match:
                print(f"メタデータ: {match['metadata']}")

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
