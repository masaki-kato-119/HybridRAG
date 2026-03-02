"""
ChromaDB-based Hybrid Indexing Module

ChromaDB を使用したハイブリッドインデックス実装。
Dense は ChromaDB（永続化・コサイン類似度）、Sparse は TF-IDF で FAISS/Qdrant と同一の検索パイプラインを共有。
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk

# ChromaDB はオプション依存
try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class ChromaDenseIndex:
    """
    ChromaDB を使用した密ベクトルインデックス。

    コサイン類似度で検索。Sentence Transformers で埋め込みを生成し、
    ChromaDB に永続化する。
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        collection_name: str = "dense_vectors",
        chroma_path: str = "./chroma_data",
    ):
        """
        Args:
            model_name: Sentence Transformers のモデル名。
            collection_name: ChromaDB のコレクション名。
            chroma_path: ChromaDB の永続化ディレクトリ。
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is required for ChromaDenseIndex. "
                "Install it with: pip install chromadb"
            )
        self.model = SentenceTransformer(model_name)
        self.collection_name = collection_name
        self.chroma_path = chroma_path
        self._client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.dimension: Optional[int] = None

    def _get_collection(self):
        """コレクションを取得または作成（cosine distance）。"""
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def build_index(self, chunks: List[Chunk]) -> None:
        """
        チャンク一覧から密ベクトルインデックスを構築する。

        Args:
            chunks: インデックス対象の Chunk のリスト。
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing")

        texts = [chunk.content for chunk in chunks]
        print("Generating embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = embeddings.astype("float32")
        self.dimension = embeddings.shape[1]

        collection = self._get_collection()
        # 既存データを削除してから投入
        try:
            self._client.delete_collection(name=self.collection_name)
            collection = self._get_collection()
        except Exception:
            pass

        ids = [f"{chunk.doc_id}_{chunk.chunk_index}" for chunk in chunks]
        metadatas = [
            {
                "doc_id": chunk.doc_id,
                "section": chunk.section or "",
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type.value,
                "source_path": chunk.source_path or "",
            }
            for chunk in chunks
        ]
        # Chroma はメタデータ値を str/int/float に制限する場合があるため、content は documents で渡す
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=texts,
        )
        print(f"Dense index built with {len(chunks)} vectors!")

    def build_index_batch(
        self, chunk_batches: List[List[Chunk]], show_progress: bool = True
    ) -> None:
        """
        チャンクをバッチ単位で処理し、密ベクトルインデックスを構築する。

        Args:
            chunk_batches: チャンクのバッチのリスト。
            show_progress: 進捗表示を行うかどうか。
        """
        if not chunk_batches or not any(chunk_batches):
            raise ValueError("No chunks provided for indexing")

        all_chunks = []
        for batch in chunk_batches:
            all_chunks.extend(batch)
        # ChromaDB はバッチ add をサポートするので、一括で構築
        self.build_index(all_chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filters: Optional[Dict] = None,
    ) -> List[Tuple[Dict, float]]:
        """
        クエリに類似するチャンクを検索する。

        Args:
            query: 検索クエリ文字列。
            top_k: 返す件数。
            metadata_filters: メタデータフィルタ（Chroma の where に渡す）。省略可。

        Returns:
            (メタデータ辞書, スコア) のリスト。スコアは類似度（大きいほど類似）。
        """
        collection = self._get_collection()
        query_embedding = self.model.encode([query]).astype("float32").tolist()

        where = None
        if metadata_filters:
            conds = []
            for key, value in metadata_filters.items():
                if isinstance(value, list):
                    conds.append({key: {"$in": value}})
                else:
                    conds.append({key: value})
            where = {"$and": conds} if len(conds) > 1 else conds[0]

        result = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where,
            include=["metadatas", "documents", "distances"],
        )

        # Chroma の cosine distance: 0 = 同一, 大きいほど非類似 → 類似度 = 1 - distance
        results = []
        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for i, (meta, dist) in enumerate(zip(metadatas or [], distances or [])):
            metadata = dict(meta) if meta else {}
            if documents and i < len(documents) and documents[i]:
                metadata["content"] = documents[i]
            # 類似度: 距離が小さいほど高い。cosine distance は [0, 2] の範囲
            score = float(1.0 - dist) if dist is not None else 0.0
            results.append((metadata, score))
        return results

    def save(self, index_path: str) -> None:
        """
        ChromaDB は永続化クライアントのため自動保存。
        互換性のためのダミーメソッド。
        """
        print(f"ChromaDB data is persisted to {self.chroma_path}")

    def load(self, index_path: str) -> None:
        """
        コレクションの存在確認。ChromaDB はパスで自動読み込み済み。
        互換性のためのダミーメソッド。
        """
        collections = [c.name for c in self._client.list_collections()]
        if self.collection_name not in collections:
            raise FileNotFoundError(
                f"Collection '{self.collection_name}' not found. Build index first."
            )
        print(f"ChromaDB collection '{self.collection_name}' loaded from {self.chroma_path}")


class ChromaSparseIndex:
    """
    TF-IDF（BM25風）によるスパースインデックス。
    FAISS/Qdrant と同一インターフェース。
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
        texts = [chunk.content for chunk in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.chunk_metadata = [
            {
                "doc_id": chunk.doc_id,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type.value,
                "source_path": chunk.source_path,
                "content": chunk.content,
            }
            for chunk in chunks
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
            all_texts.extend([chunk.content for chunk in chunks])
            all_metadata.extend(
                [
                    {
                        "doc_id": chunk.doc_id,
                        "section": chunk.section,
                        "chunk_index": chunk.chunk_index,
                        "chunk_type": chunk.chunk_type.value,
                        "source_path": chunk.source_path,
                        "content": chunk.content,
                    }
                    for chunk in chunks
                ]
            )
        self.tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        self.chunk_metadata = all_metadata

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        if self.tfidf_matrix is None:
            raise ValueError("Index not built. Call build_index first.")
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((self.chunk_metadata[idx], float(similarities[idx])))
        return results

    def save(self, index_path: str) -> None:
        path = Path(index_path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "sparse_index.pkl", "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "tfidf_matrix": self.tfidf_matrix,
                    "chunk_metadata": self.chunk_metadata,
                },
                f,
            )
        print(f"Sparse index saved to {path / 'sparse_index.pkl'}")

    def load(self, index_path: str) -> None:
        path = Path(index_path)
        pkl_path = path / "sparse_index.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Sparse index not found at {pkl_path}")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.tfidf_matrix = data["tfidf_matrix"]
            self.chunk_metadata = data["chunk_metadata"]
        print(f"Sparse index loaded from {pkl_path}")


class ChromaHybridIndex:
    """
    ChromaDB Dense と Sparse の両インデックスを保持し、ハイブリッド検索に供する。
    RRFRetriever が期待する .dense_index / .sparse_index の search() インターフェースを提供。
    """

    def __init__(
        self,
        dense_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        sparse_max_features: int = 10000,
        chroma_path: str = "./chroma_data",
    ):
        self.dense_index = ChromaDenseIndex(
            model_name=dense_model_name,
            collection_name="dense_vectors",
            chroma_path=chroma_path,
        )
        self.sparse_index = ChromaSparseIndex(sparse_max_features)

    def build_index(self, chunks: List[Chunk]) -> None:
        print("Building Chroma dense index...")
        self.dense_index.build_index(chunks)
        print("Building sparse index...")
        self.sparse_index.build_index(chunks)
        print("Chroma hybrid index built successfully!")

    def build_index_batch(
        self, chunk_batches: List[List[Chunk]], show_progress: bool = True
    ) -> None:
        print("Building Chroma dense index in batches...")
        self.dense_index.build_index_batch(chunk_batches, show_progress=show_progress)
        print("Building sparse index in batches...")
        self.sparse_index.build_index_batch(chunk_batches, show_progress=show_progress)
        print("Chroma hybrid index built successfully!")

    def save(self, index_path: str) -> None:
        self.dense_index.save(index_path)
        self.sparse_index.save(index_path)

    def load(self, index_path: str) -> None:
        self.dense_index.load(index_path)
        self.sparse_index.load(index_path)
