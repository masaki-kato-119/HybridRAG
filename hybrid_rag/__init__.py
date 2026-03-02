"""
Hybrid RAG System

Dense（ベクトル）と Sparse（キーワード）を組み合わせた
ハイブリッド検索による RAG（Retrieval-Augmented Generation）実装です。

主なコンポーネント:
    - **ingestion**: PDF / Markdown / HTML / テキストの取り込み
    - **chunking**: セマンティックチャンキング（見出し・コード・表を考慮）
    - **indexing**: Dense (FAISS / Qdrant / ChromaDB / pgvector) + Sparse (TF-IDF)
    - **retrieval**: RRF（Reciprocal Rank Fusion）によるハイブリッド検索
    - **reranking**: Cross-encoder による再ランキング
    - **context**: トークン上限を考慮したコンテキスト構築
    - **storage**: SQLite によるメタデータ・チャンク保存
    - **evaluation**: 検索評価メトリクス（Precision@k, Recall@k, MRR, NDCG）

バックエンド:
    - **FAISS版**: 標準版（FAISS + SQLite）
    - **Qdrant版**: 高速フィルタリング版（50-70%高速）
    - **ChromaDB版**: 永続化ベクトル DB（実験・比較用）
    - **PostgreSQL版**: pgvector による Dense 検索（再ランク込み）

使用例:
    >>> # 推奨: ファクトリー関数でバックエンドを指定
    >>> from hybrid_rag import create_rag_system
    >>> rag = create_rag_system(backend="faiss")   # または "qdrant", "chroma", "postgres"
    >>> rag.ingest_documents(["doc.pdf"])
    >>> result = rag.query("質問文")

    >>> # 直接インポート（FAISS版）
    >>> from hybrid_rag import HybridRAGSystem
    >>> rag = HybridRAGSystem()
"""

__version__ = "1.0.0"

# 標準版（FAISS + SQLite）
from .rag_system import HybridRAGSystem

# ファクトリー関数（推奨）
from .rag_system_factory import (
    create_rag_system,
    get_available_backends,
    print_backend_comparison,
)

# Qdrant版は明示的にインポートが必要（qdrant-clientが必須のため）
# from .rag_system_qdrant import QdrantHybridRAGSystem

__all__ = [
    "HybridRAGSystem",
    "create_rag_system",
    "get_available_backends",
    "print_backend_comparison",
]
