# clip-pinecone-vectorizer

CLIP を使用して画像をベクトル化し、Pinecone に登録するシステム

## 概要

このシステムは、画像とメタデータを受け取り、以下の処理を行います：

1. 画像を CLIP モデルを使用してベクトル化
2. メタデータと共に Pinecone ベクトルデータベースに保存
3. 後続の類似画像検索に利用可能な形式で保存

## セットアップ

### 1. 環境構築

```bash
uv sync
```

### 2. 環境変数の設定

`.env`ファイルを作成し、以下の内容を設定:

```env
PINECONE_API_KEY=your_api_key
PINECONE_INDEX_NAME=your_index_name
```

## 使用方法

### 1. 画像データの準備

`data/items.json`に画像の情報を記載:

```json
{
  "items": [
    {
      "image_path": "./images/example.jpg",
      "image_name": "unique_image_id",
      "metadata": {
        "title": "画像タイトル",
        "category": "カテゴリ",
        "tags": ["タグ1", "タグ2"],
        "description": "画像の説明"
      }
    }
  ]
}
```

### 2. 実行

```bash
python src/main.py
```

## プロジェクト構造

```
.
├── src/
│   ├── image_register/
│   │   ├── domain/
│   │   │   └── models.py          # データモデル
│   │   ├── infrastructure/
│   │   │   ├── image_vectorizer.py  # 画像のベクトル化
│   │   │   └── pinecone_client.py   # Pineconeとの通信
│   │   ├── repository/
│   │   │   └── item_repository.py   # JSONファイルの読み込み
│   │   └── service/
│   │       └── image_register.py     # メインのビジネスロジック
│   └── main.py                    # エントリーポイント
├── data/
│   └── items.json                 # 画像メタデータ
├── tests/
│   └── test_image_register.py     # テストコード
├── .env                           # 環境変数
└── README.md
```

## テスト

テストの実行:

```bash
uv run pytest tests -v
```

## アーキテクチャ

- ドメイン駆動設計（DDD）の考え方を採用
- クリーンアーキテクチャに基づく層分け
  - Domain 層: ビジネスロジックの中心
  - Infrastructure 層: 外部サービスとの連携
  - Repository 層: データアクセス
  - Service 層: ユースケースの実装
- 依存性注入を活用した疎結合な設計

## 開発ガイドライン

1. テストファーストな開発を推奨
2. BDD スタイルのテスト記述
3. 型ヒントの活用
4. docstring によるドキュメント化
5. コードフォーマット

   ```bash
   # コードの自動フォーマット
   uv run ruff format .

   # フォーマットチェック（CI用）
   uv run ruff format . --check
   ```

## 必要要件

- Python 3.12 以上
- 必要なパッケージ:
  - torch
  - transformers
  - pillow
  - pinecone-client
  - python-dotenv
  - numpy
  - ruff
