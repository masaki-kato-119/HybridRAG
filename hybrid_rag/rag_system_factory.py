"""
RAGシステムのファクトリー。

FAISS / Qdrant / ChromaDB / PostgreSQL を backend で切り替える統一インターフェースを提供。

使用例:
    # バックエンドを指定
    rag = create_rag_system(backend="faiss")    # または "qdrant", "chroma", "postgres"
    
    # 環境変数で切り替え
    import os
    os.environ["RAG_BACKEND"] = "qdrant"
    rag = create_rag_system()
"""

import os
from typing import Optional

from .query_expansion import QueryExpander


def create_rag_system(
    backend: Optional[str] = None,
    db_path: str = "hybrid_rag.db",
    index_path: str = "indices",
    qdrant_path: str = "./qdrant_data",
    chroma_path: str = "./chroma_data",
    log_dir: str = "logs",
    dense_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    rerank_model: str = "BAAI/bge-reranker-v2-m3",
    max_chunk_size: int = 512,
    max_context_tokens: int = 4000,
    rrf_k: int = 60,
    retrieval_candidates_multiplier: int = 2,
    query_expander: Optional[QueryExpander] = None,
):
    """
    RAGシステムを作成する。
    
    バックエンドを指定するだけで、FAISS版とQdrant版を切り替え可能。
    
    Args:
        backend: "faiss", "qdrant", "chroma", "postgres" のいずれか。省略時は環境変数 RAG_BACKEND を参照、
                 それも無ければ "faiss" をデフォルトとする。
        db_path: SQLite データベースのパス。
        index_path: FAISS版のインデックス保存先（backend="faiss" 時）。Postgres 版では Sparse 等の保存先にも使用。
        qdrant_path: Qdrant版のデータ保存先（backend="qdrant" 時のみ使用）。
        chroma_path: ChromaDB版のデータ保存先（backend="chroma" 時のみ使用）。
        log_dir: ログ出力ディレクトリ。
        dense_model: Dense埋め込み用モデル名。
        rerank_model: 再ランキング用モデル名。
        max_chunk_size: チャンクの最大文字数。
        max_context_tokens: コンテキストの最大トークン数。
        rrf_k: RRF定数。
        retrieval_candidates_multiplier: 候補数の倍率。
        query_expander: クエリ拡張用のQueryExpanderインスタンス。
    
    Returns:
        HybridRAGSystem / QdrantHybridRAGSystem / ChromaHybridRAGSystem / PostgresHybridRAGSystem のいずれか。
    
    Raises:
        ValueError: 不正なbackend指定の場合。
        ImportError: Qdrant版を使用するがqdrant-clientが未インストールの場合。
    
    Examples:
        >>> # FAISS版を使用
        >>> rag = create_rag_system(backend="faiss")
        
        >>> # Qdrant版を使用（高速フィルタリング）
        >>> rag = create_rag_system(backend="qdrant")
        
        >>> # 環境変数で切り替え
        >>> import os
        >>> os.environ["RAG_BACKEND"] = "qdrant"
        >>> rag = create_rag_system()  # Qdrant版が使われる
    """
    # バックエンドの決定
    if backend is None:
        backend = os.environ.get("RAG_BACKEND", "faiss").lower()
    else:
        backend = backend.lower()
    
    # バックエンドの検証
    if backend not in ["faiss", "qdrant", "chroma", "postgres"]:
        raise ValueError(
            f"Invalid backend: {backend}. Must be 'faiss', 'qdrant', 'chroma', or 'postgres'."
        )
    
    print(f"Creating RAG system with backend: {backend}")
    
    if backend == "faiss":
        # FAISS版を使用
        from .rag_system import HybridRAGSystem
        
        return HybridRAGSystem(
            db_path=db_path,
            index_path=index_path,
            log_dir=log_dir,
            dense_model=dense_model,
            rerank_model=rerank_model,
            max_chunk_size=max_chunk_size,
            max_context_tokens=max_context_tokens,
            rrf_k=rrf_k,
            retrieval_candidates_multiplier=retrieval_candidates_multiplier,
            query_expander=query_expander,
        )
    
    elif backend == "qdrant":
        # Qdrant版を使用
        try:
            from .rag_system_qdrant import QdrantHybridRAGSystem
        except ImportError as e:
            raise ImportError(
                "Qdrant backend requires qdrant-client. "
                "Install it with: pip install qdrant-client"
            ) from e
        
        return QdrantHybridRAGSystem(
            db_path=db_path,
            qdrant_path=qdrant_path,
            log_dir=log_dir,
            dense_model=dense_model,
            rerank_model=rerank_model,
            max_chunk_size=max_chunk_size,
            max_context_tokens=max_context_tokens,
            rrf_k=rrf_k,
            retrieval_candidates_multiplier=retrieval_candidates_multiplier,
            query_expander=query_expander,
        )
    
    elif backend == "chroma":
        # ChromaDB版を使用
        try:
            from .rag_system_chroma import ChromaHybridRAGSystem
        except ImportError as e:
            raise ImportError(
                "Chroma backend requires chromadb. "
                "Install it with: pip install chromadb"
            ) from e
        
        return ChromaHybridRAGSystem(
            db_path=db_path,
            chroma_path=chroma_path,
            log_dir=log_dir,
            dense_model=dense_model,
            rerank_model=rerank_model,
            max_chunk_size=max_chunk_size,
            max_context_tokens=max_context_tokens,
            rrf_k=rrf_k,
            retrieval_candidates_multiplier=retrieval_candidates_multiplier,
            query_expander=query_expander,
        )
    
    else:  # backend == "postgres"
        try:
            from .rag_system_postgres import PostgresHybridRAGSystem
        except ImportError as e:
            raise ImportError(
                "Postgres backend requires psycopg[binary] and pgvector.\n"
                "Install with: pip install psycopg[binary] pgvector"
            ) from e
        
        return PostgresHybridRAGSystem(
            db_path=db_path,
            index_path=index_path,
            log_dir=log_dir,
            dense_model=dense_model,
            rerank_model=rerank_model,
            max_chunk_size=max_chunk_size,
            max_context_tokens=max_context_tokens,
            rrf_k=rrf_k,
            retrieval_candidates_multiplier=retrieval_candidates_multiplier,
            query_expander=query_expander,
        )


def get_available_backends():
    """
    利用可能なバックエンドのリストを返す。
    
    Returns:
        利用可能なバックエンド名のリスト。
    
    Examples:
        >>> backends = get_available_backends()
        >>> print(backends)
        ['faiss', 'qdrant', 'chroma', 'postgres']  # 依存が入っていれば
    """
    backends = ["faiss"]  # FAISSは常に利用可能
    
    try:
        import qdrant_client  # noqa: F401
        backends.append("qdrant")
    except ImportError:
        pass
    
    try:
        import chromadb  # noqa: F401
        backends.append("chroma")
    except ImportError:
        pass
    
    try:
        import psycopg  # noqa: F401
        from pgvector.psycopg import register_vector  # noqa: F401
        backends.append("postgres")
    except ImportError:
        pass
    
    return backends


def print_backend_comparison():
    """
    FAISS / Qdrant / ChromaDB / PostgreSQL の比較表を表示する。

    Examples:
        >>> print_backend_comparison()
    """
    print("=" * 95)
    print("RAGシステム バックエンド比較")
    print("=" * 95)
    print()
    print("| 項目         | FAISS+SQLite   | Qdrant           | ChromaDB      | PostgreSQL(pgvector) |")
    print("|--------------|----------------|------------------|---------------|------------------------|")
    print("| フィルタ検索 | 遅い（後処理） | 高速（単一ステージ）| 後処理        | 後処理                 |")
    print("| メモリ使用量 | 中             | 小（量子化可能） | 中            | 中（DB に依存）       |")
    print("| セットアップ | pip のみ       | pip のみ         | pip のみ      | Docker+pip            |")
    print("| サーバー     | 不要           | 不要             | 不要          | Postgres 必要         |")
    print("| 永続化       | ファイル       | ファイル         | ファイル      | DB 永続化             |")
    print("| スケール     | 1000-2000万    | 5000万           | 中規模向け    | 大規模向け            |")
    print()
    print("推奨:")
    print("  • 小規模（<500万chunk）: FAISS版で十分")
    print("  • フィルタ検索を多用: Qdrant版（50-70%高速）")
    print("  • 実験・比較用: ChromaDB版（pip install chromadb で追加）")
    print("  • 既存 Postgres 活用・大規模: PostgreSQL 版（pip install psycopg[binary] pgvector、Docker 等）")
    print()
    print("利用可能なバックエンド:", ", ".join(get_available_backends()))
    print("=" * 95)
