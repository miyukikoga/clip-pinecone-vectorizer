from typing import Dict, Optional
import numpy as np
from pinecone import Pinecone


class PineconeClient:
    def __init__(self, api_key: str, index_name: str):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

    def upsert(
        self, vector_id: str, vector: np.ndarray, metadata: Optional[Dict] = None
    ) -> None:
        """ベクトルをPineconeに登録"""
        self.index.upsert(
            vectors=[
                {"id": vector_id, "values": vector.tolist(), "metadata": metadata or {}}
            ],
            namespace="ns1",
        )
