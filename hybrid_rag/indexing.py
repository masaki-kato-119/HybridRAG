"""
Hybrid Indexing Module
Implements both dense (FAISS) and sparse (BM25) indexing.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk


class DenseIndex:
    """
    FAISS と Sentence Transformers を用いた密ベクトルインデックス。

    コサイン類似度で検索するため、ベクトルは L2 正規化して内積で比較する。
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Args:
            model_name: Sentence Transformers のモデル名（HuggingFace）。
        """
        self.model = SentenceTransformer(model_name)
        self.index: Any = None
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.dimension: Optional[int] = None

    def build_index(self, chunks: List[Chunk]) -> None:
        """
        チャンク一覧から密ベクトルインデックスを構築する。

        Args:
            chunks: インデックス対象の Chunk のリスト。

        Raises:
            ValueError: chunks が空の場合。
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing")

        # Extract text content
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = embeddings.astype("float32")

        # Initialize FAISS index
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to index
        self.index.add(embeddings)

        # Store metadata (without content to save memory)
        # Content can be retrieved from database when needed
        self.chunk_metadata = [
            {
                "doc_id": chunk.doc_id,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type.value,
                "source_path": chunk.source_path,
                # "content": chunk.content,  # Removed to save memory
            }
            for chunk in chunks
        ]

    def build_index_batch(
        self, chunk_batches: List[List[Chunk]], show_progress: bool = True
    ) -> None:
        """
        チャンクをバッチ単位で処理し、密ベクトルインデックスを構築する（メモリ節約用）。

        Args:
            chunk_batches: チャンクのバッチのリスト。各要素は Chunk のリスト。
            show_progress: 進捗表示を行うかどうか。
        """
        if not chunk_batches or not any(chunk_batches):
            raise ValueError("No chunks provided for indexing")

        all_embeddings = []
        all_metadata = []

        # Process each batch
        for batch_idx, chunks in enumerate(chunk_batches):
            if not chunks:
                continue

            # Extract text content from batch
            texts = [chunk.content for chunk in chunks]

            # Generate embeddings for batch
            batch_embeddings = self.model.encode(
                texts, show_progress_bar=show_progress and batch_idx == 0
            )
            batch_embeddings = batch_embeddings.astype("float32")

            # Store batch embeddings
            all_embeddings.append(batch_embeddings)

            # Store metadata (without content to save memory)
            batch_metadata = [
                {
                    "doc_id": chunk.doc_id,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type.value,
                    "source_path": chunk.source_path,
                    # "content": chunk.content,  # Removed to save memory
                }
                for chunk in chunks
            ]
            all_metadata.extend(batch_metadata)

            # Clear batch from memory
            del texts, batch_embeddings, batch_metadata

        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)

        # Initialize FAISS index
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to index
        self.index.add(embeddings)

        # Store metadata
        self.chunk_metadata = all_metadata

        # Clear embeddings from memory
        del embeddings, all_embeddings

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        クエリに類似するチャンクを検索する。

        Args:
            query: 検索クエリ文字列。
            top_k: 返す件数。

        Returns:
            (メタデータ辞書, スコア) のリスト。メタデータに content は含まれない場合あり。

        Raises:
            ValueError: インデックス未構築の場合。
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        index = self.index

        # Encode query
        query_embedding = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = index.search(query_embedding, top_k)

        # Return results with metadata
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunk_metadata):
                results.append((self.chunk_metadata[idx], float(score)))

        return results

    def save(self, index_path: str) -> None:
        """
        インデックスとメタデータをディスクに保存する。

        Args:
            index_path: 保存先ディレクトリ。dense_index.faiss, dense_metadata.json 等が作成される。
        """
        path = Path(index_path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / "dense_index.faiss"))

        # Save metadata
        with open(path / "dense_metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.chunk_metadata, f, ensure_ascii=False, indent=2)

        # Save model info
        model_info = {
            "model_name": self.model._modules["0"].auto_model.name_or_path,
            "dimension": self.dimension,
        }
        with open(path / "dense_model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

    def load(self, index_path: str) -> None:
        """
        ディスクからインデックスとメタデータを読み込む。

        Args:
            index_path: 保存済みインデックスがあるディレクトリのパス。
        """
        path = Path(index_path)

        # Load FAISS index
        self.index = faiss.read_index(str(path / "dense_index.faiss"))

        # Load metadata
        with open(path / "dense_metadata.json", "r", encoding="utf-8") as f:
            self.chunk_metadata = json.load(f)

        # Load model info
        with open(path / "dense_model_info.json", "r") as f:
            model_info = json.load(f)
            self.dimension = model_info["dimension"]


class SparseIndex:
    """
    TF-IDF（BM25 風）によるスパースインデックス。

    キーワードマッチング向け。バイグラムを含む。
    """

    def __init__(self, max_features: int = 10000):
        """
        Args:
            max_features: TF-IDF の最大特徴量数。
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),  # Include bigrams
            lowercase=True,
            token_pattern=r"\b\w+\b",
        )
        self.tfidf_matrix: Any = None
        self.chunk_metadata: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Chunk]) -> None:
        """
        チャンク一覧からスパースインデックスを構築する。

        Args:
            chunks: インデックス対象の Chunk のリスト。

        Raises:
            ValueError: chunks が空の場合。
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing")

        # Extract text content
        texts = [chunk.content for chunk in chunks]

        # Build TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        # Store metadata (without content to save memory)
        self.chunk_metadata = [
            {
                "doc_id": chunk.doc_id,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type.value,
                "source_path": chunk.source_path,
                # "content": chunk.content,  # Removed to save memory
            }
            for chunk in chunks
        ]

    def build_index_batch(
        self, chunk_batches: List[List[Chunk]], show_progress: bool = True
    ) -> None:
        """
        Build sparse index from chunks in batches to avoid OOM.

        Args:
            chunk_batches: List of chunk batches, each batch is a list of Chunk objects
            show_progress: Whether to show progress
        """
        if not chunk_batches or not any(chunk_batches):
            raise ValueError("No chunks provided for indexing")

        all_texts = []
        all_metadata = []

        # Collect all texts and metadata from batches
        for chunks in chunk_batches:
            if not chunks:
                continue

            # Extract text content from batch
            batch_texts = [chunk.content for chunk in chunks]
            all_texts.extend(batch_texts)

            # Store metadata (without content to save memory)
            batch_metadata = [
                {
                    "doc_id": chunk.doc_id,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type.value,
                    "source_path": chunk.source_path,
                    # "content": chunk.content,  # Removed to save memory
                }
                for chunk in chunks
            ]
            all_metadata.extend(batch_metadata)

            # Clear batch from memory
            del batch_texts, batch_metadata

        # Build TF-IDF matrix from all texts
        self.tfidf_matrix = self.vectorizer.fit_transform(all_texts)

        # Store metadata
        self.chunk_metadata = all_metadata

        # Clear texts from memory
        del all_texts

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        TF-IDF 類似度で関連チャンクを検索する。

        Args:
            query: 検索クエリ。
            top_k: 返す件数。

        Returns:
            (メタデータ辞書, 類似度スコア) のリスト。

        Raises:
            ValueError: インデックス未構築の場合。
        """
        if self.tfidf_matrix is None:
            raise ValueError("Index not built. Call build_index first.")

        # Transform query
        query_vector = self.vectorizer.transform([query])

        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return results with metadata
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only return non-zero similarities
                results.append((self.chunk_metadata[idx], float(similarities[idx])))

        return results

    def save(self, index_path: str) -> None:
        """
        ベクトライザー・行列・メタデータをディスクに保存する。

        Args:
            index_path: 保存先ディレクトリ。
        """
        path = Path(index_path)
        path.mkdir(parents=True, exist_ok=True)

        # Save TF-IDF vectorizer and matrix
        with open(path / "sparse_vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)

        with open(path / "sparse_matrix.pkl", "wb") as f:
            pickle.dump(self.tfidf_matrix, f)

        # Save metadata
        with open(path / "sparse_metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.chunk_metadata, f, ensure_ascii=False, indent=2)

    def load(self, index_path: str) -> None:
        """
        ディスクからベクトライザー・行列・メタデータを読み込む。

        Args:
            index_path: 保存済みインデックスがあるディレクトリ。
        """
        path = Path(index_path)

        # Load TF-IDF vectorizer and matrix
        with open(path / "sparse_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

        with open(path / "sparse_matrix.pkl", "rb") as f:
            self.tfidf_matrix = pickle.load(f)

        # Load metadata
        with open(path / "sparse_metadata.json", "r", encoding="utf-8") as f:
            self.chunk_metadata = json.load(f)


class HybridIndex:
    """
    Dense と Sparse の両インデックスを保持し、ハイブリッド検索に供する。
    """

    def __init__(
        self,
        dense_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        sparse_max_features: int = 10000,
    ):
        """
        Args:
            dense_model_name: Dense インデックス用 Sentence Transformers モデル名。
            sparse_max_features: Sparse インデックス用の最大特徴量数。
        """
        self.dense_index = DenseIndex(dense_model_name)
        self.sparse_index = SparseIndex(sparse_max_features)

    def build_index(self, chunks: List[Chunk]) -> None:
        """
        Dense と Sparse の両インデックスをチャンクから構築する。

        Args:
            chunks: インデックス対象の Chunk のリスト。
        """
        print("Building dense index...")
        self.dense_index.build_index(chunks)

        print("Building sparse index...")
        self.sparse_index.build_index(chunks)

        print("Hybrid index built successfully!")

    def build_index_batch(
        self, chunk_batches: List[List[Chunk]], show_progress: bool = True
    ) -> None:
        """
        チャンクをバッチ単位で処理し、Dense と Sparse の両インデックスを構築する（メモリ節約用）。

        Args:
            chunk_batches: チャンクのバッチのリスト。各要素は Chunk のリスト。
            show_progress: 進捗表示を行うかどうか。
        """
        print("Building dense index in batches...")
        self.dense_index.build_index_batch(chunk_batches, show_progress=show_progress)

        print("Building sparse index in batches...")
        self.sparse_index.build_index_batch(chunk_batches, show_progress=show_progress)

        print("Hybrid index built successfully!")

    def save(self, index_path: str) -> None:
        """
        Dense と Sparse の両インデックスを保存する。

        Args:
            index_path: 保存先ディレクトリのパス。
        """
        self.dense_index.save(index_path)
        self.sparse_index.save(index_path)

    def load(self, index_path: str) -> None:
        """
        Dense と Sparse の両インデックスを読み込む。

        Args:
            index_path: 保存済みインデックスがあるディレクトリのパス。
        """
        self.dense_index.load(index_path)
        self.sparse_index.load(index_path)
