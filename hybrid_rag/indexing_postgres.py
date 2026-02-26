"""
PostgreSQL + pgvector を使った Dense インデックスと TF-IDF スパースインデックス。

FAISS / Qdrant / Chroma と同じインターフェースで扱えるように、
PostgresDenseIndex / PostgresSparseIndex / PostgresHybridIndex を提供する。
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk


class PostgresDenseIndex:
    """
    PostgreSQL(pgvector) に埋め込みを保存する Dense インデックス。

    - ベクトルは pgvector の `vector` 型で保存
    - cosine 距離用の HNSW インデックスを作成
    - 検索時は `embedding <=> query_vector` で近傍検索
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        table_name: str = "rag_embeddings_pg",
    ):
        self.model = SentenceTransformer(model_name)
        self.table_name = table_name
        self.dimension: Optional[int] = None

    def _get_conn(self):
        host = os.environ.get("PGHOST", "localhost")
        port = int(os.environ.get("PGPORT", "5432"))
        dbname = os.environ.get("PGDATABASE", "hybridrag")
        user = os.environ.get("PGUSER", "hybridrag")
        password = os.environ.get("PGPASSWORD", "hybridrag")
        connect_timeout = int(os.environ.get("PGCONNECT_TIMEOUT", "10"))

        conn = psycopg.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            connect_timeout=connect_timeout,
        )
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        register_vector(conn)
        return conn

    def build_index(self, chunks: List[Chunk]) -> None:
        """チャンク一覧から Postgres 上にベクトルインデックスを構築する。"""
        if not chunks:
            raise ValueError("No chunks provided for indexing")

        texts = [c.content for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True).astype("float32")
        self.dimension = int(embeddings.shape[1])

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS %s;" % self.table_name)
                cur.execute(
                    f"""
                    CREATE TABLE {self.table_name} (
                        id          bigserial PRIMARY KEY,
                        doc_id      text,
                        section     text,
                        chunk_index integer,
                        chunk_type  text,
                        source_path text,
                        content     text,
                        embedding   vector({self.dimension})
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX {self.table_name}_embedding_idx
                    ON {self.table_name}
                    USING hnsw (embedding vector_cosine_ops);
                    """
                )

                for chunk, emb in zip(chunks, embeddings):
                    cur.execute(
                        f"""
                        INSERT INTO {self.table_name}
                            (doc_id, section, chunk_index, chunk_type, source_path, content, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s);
                        """,
                        (
                            chunk.doc_id,
                            chunk.section,
                            chunk.chunk_index,
                            chunk.chunk_type.value,
                            chunk.source_path,
                            chunk.content,
                            emb.tolist(),
                        ),
                    )
            conn.commit()
        finally:
            conn.close()

    def build_index_batch(
        self, chunk_batches: List[List[Chunk]], show_progress: bool = True
    ) -> None:
        """バッチ単位のインデックス構築。内部的には一括で build_index する。"""
        all_chunks: List[Chunk] = []
        for batch in chunk_batches:
            all_chunks.extend(batch)
        self.build_index(all_chunks)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        クエリに類似するチャンクを pgvector で検索する。

        Returns:
            (メタデータ辞書, 類似度スコア) のリスト。
        """
        conn = self._get_conn()
        try:
            q_emb = self.model.encode([query]).astype("float32")[0].tolist()
            results: List[Tuple[Dict, float]] = []
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        doc_id,
                        section,
                        chunk_index,
                        chunk_type,
                        source_path,
                        content,
                        (1.0 - (embedding <=> %s::vector)) AS score
                    FROM {self.table_name}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (q_emb, q_emb, top_k),
                )
                for row in cur.fetchall():
                    doc_id, section, chunk_index, chunk_type, source_path, content, score = row
                    metadata = {
                        "doc_id": doc_id,
                        "section": section,
                        "chunk_index": chunk_index,
                        "chunk_type": chunk_type,
                        "source_path": source_path,
                        "content": content,
                    }
                    results.append((metadata, float(score)))
            return results
        finally:
            conn.close()

    def save(self, index_path: str) -> None:
        """Postgres は自前で永続化されるため、ここでは何もしない。"""
        print("Postgres dense index is stored inside the database.")

    def load(self, index_path: str) -> None:
        """テーブル存在チェックのみ行う（簡易実装）。"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT to_regclass(%s);
                    """,
                    (self.table_name,),
                )
                exists = cur.fetchone()[0]
                if not exists:
                    raise FileNotFoundError(
                        f"Postgres table '{self.table_name}' not found. Build index first."
                    )
        finally:
            conn.close()


class PostgresSparseIndex:
    """
    TF-IDF によるスパースインデックス。
    Qdrant / Chroma 版とほぼ同じ実装。
    """

    def __init__(self, max_features: int = 10000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            lowercase=True,
            token_pattern=r"\b\w+\b",
        )
        self.tfidf_matrix: Any = None
        self.chunk_metadata: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("No chunks provided for indexing")
        texts = [c.content for c in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.chunk_metadata = [
            {
                "doc_id": c.doc_id,
                "section": c.section,
                "chunk_index": c.chunk_index,
                "chunk_type": c.chunk_type.value,
                "source_path": c.source_path,
                "content": c.content,
            }
            for c in chunks
        ]

    def build_index_batch(
        self, chunk_batches: List[List[Chunk]], show_progress: bool = True
    ) -> None:
        if not chunk_batches or not any(chunk_batches):
            raise ValueError("No chunks provided for indexing")
        all_texts = []
        all_metadata = []
        for chunks in chunk_batches:
            if not chunks:
                continue
            all_texts.extend([c.content for c in chunks])
            all_metadata.extend(
                [
                    {
                        "doc_id": c.doc_id,
                        "section": c.section,
                        "chunk_index": c.chunk_index,
                        "chunk_type": c.chunk_type.value,
                        "source_path": c.source_path,
                        "content": c.content,
                    }
                    for c in chunks
                ]
            )
        self.tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        self.chunk_metadata = all_metadata

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        if self.tfidf_matrix is None:
            raise ValueError("Index not built. Call build_index first.")
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_indices = sims.argsort()[::-1][:top_k]
        results: List[Tuple[Dict, float]] = []
        for idx in top_indices:
            if sims[idx] > 0:
                results.append((self.chunk_metadata[idx], float(sims[idx])))
        return results

    def save(self, index_path: str) -> None:
        path = Path(index_path)
        path.mkdir(parents=True, exist_ok=True)
        import pickle

        with open(path / "postgres_sparse_index.pkl", "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "tfidf_matrix": self.tfidf_matrix,
                    "chunk_metadata": self.chunk_metadata,
                },
                f,
            )

    def load(self, index_path: str) -> None:
        path = Path(index_path)
        pkl_path = path / "postgres_sparse_index.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Sparse index not found at {pkl_path}")
        import pickle

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["tfidf_matrix"]
        self.chunk_metadata = data["chunk_metadata"]


class PostgresHybridIndex:
    """
    Postgres Dense + TF-IDF Sparse のハイブリッドインデックス。

    RRFRetriever が期待する `dense_index` / `sparse_index` を公開する。
    """

    def __init__(
        self,
        dense_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        sparse_max_features: int = 10000,
    ):
        self.dense_index = PostgresDenseIndex(model_name=dense_model_name)
        self.sparse_index = PostgresSparseIndex(max_features=sparse_max_features)

    def build_index(self, chunks: List[Chunk]) -> None:
        self.dense_index.build_index(chunks)
        self.sparse_index.build_index(chunks)

    def build_index_batch(
        self, chunk_batches: List[List[Chunk]], show_progress: bool = True
    ) -> None:
        self.dense_index.build_index_batch(chunk_batches, show_progress=show_progress)
        self.sparse_index.build_index_batch(chunk_batches, show_progress=show_progress)

    def save(self, index_path: str) -> None:
        self.dense_index.save(index_path)
        self.sparse_index.save(index_path)

    def load(self, index_path: str) -> None:
        self.dense_index.load(index_path)
        self.sparse_index.load(index_path)

