"""
BM25インデックスモジュール。

TF-IDFよりも高速かつ高精度なBM25アルゴリズムを実装する。
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .chunking import Chunk


class BM25Index:
    """
    BM25アルゴリズムによるスパース検索インデックス。

    TF-IDFよりも検索精度・速度ともに優れている。
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """
        BM25インデックスを初期化する。

        Args:
            k1: 用語頻度の飽和パラメータ。デフォルト 1.5。
            b: 文書長の正規化パラメータ。デフォルト 0.75。
            epsilon: IDF計算時の平滑化パラメータ。デフォルト 0.25。
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.vocabulary: set = set()

    def _tokenize(self, text: str) -> List[str]:
        """
        テキストをトークン化する。

        Args:
            text: 入力テキスト。

        Returns:
            トークンのリスト。
        """
        # シンプルな空白分割（必要に応じて高度なトークナイザーに変更可能）
        return text.lower().split()

    def _calc_idf(self, nd: Dict[str, int]) -> Dict[str, float]:
        """
        IDF（逆文書頻度）を計算する。

        Args:
            nd: 各単語の文書頻度。

        Returns:
            各単語のIDFスコア。
        """
        idf = {}
        for word, freq in nd.items():
            # BM25のIDF計算式
            idf[word] = np.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
        return idf

    def build_index(self, chunks: List[Chunk]) -> None:
        """
        チャンク一覧からBM25インデックスを構築する。

        Args:
            chunks: インデックス対象の Chunk のリスト。

        Raises:
            ValueError: chunks が空の場合。
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing")

        self.corpus_size = len(chunks)
        self.doc_freqs = []
        self.doc_len = []
        self.chunk_metadata = []
        nd = {}  # 各単語の文書頻度

        # 各チャンクを処理
        for chunk in chunks:
            # トークン化
            tokens = self._tokenize(chunk.content)
            self.doc_len.append(len(tokens))

            # 単語頻度を計算
            frequencies: Dict[str, int] = {}
            for token in tokens:
                self.vocabulary.add(token)
                frequencies[token] = frequencies.get(token, 0) + 1

            self.doc_freqs.append(frequencies)

            # 文書頻度を更新
            for token in frequencies.keys():
                nd[token] = nd.get(token, 0) + 1

            # メタデータを保存
            self.chunk_metadata.append(
                {
                    "doc_id": chunk.doc_id,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type.value,
                    "source_path": chunk.source_path,
                }
            )

        # 平均文書長を計算
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0

        # IDFを計算
        self.idf = self._calc_idf(nd)

    def build_index_batch(
        self, chunk_batches: List[List[Chunk]], show_progress: bool = True
    ) -> None:
        """
        チャンクをバッチ単位で処理し、BM25インデックスを構築する。

        Args:
            chunk_batches: チャンクのバッチのリスト。
            show_progress: 進捗表示を行うかどうか。
        """
        if not chunk_batches or not any(chunk_batches):
            raise ValueError("No chunks provided for indexing")

        # すべてのチャンクを結合
        all_chunks = []
        for chunks in chunk_batches:
            if chunks:
                all_chunks.extend(chunks)

        # 通常のbuild_indexを呼び出す
        self.build_index(all_chunks)

    def _get_scores(self, query_tokens: List[str]) -> np.ndarray:
        """
        クエリトークンに対する各文書のBM25スコアを計算する。

        Args:
            query_tokens: クエリのトークンリスト。

        Returns:
            各文書のスコア配列。
        """
        scores = np.zeros(self.corpus_size)

        for token in query_tokens:
            if token not in self.idf:
                continue

            token_idf = self.idf[token]

            for idx, doc_freq in enumerate(self.doc_freqs):
                if token not in doc_freq:
                    continue

                freq = doc_freq[token]
                doc_len = self.doc_len[idx]

                # BM25スコア計算
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[idx] += token_idf * (numerator / denominator)

        return scores

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        BM25スコアで関連チャンクを検索する。

        Args:
            query: 検索クエリ。
            top_k: 返す件数。

        Returns:
            (メタデータ辞書, BM25スコア) のリスト。

        Raises:
            ValueError: インデックス未構築の場合。
        """
        if not self.doc_freqs:
            raise ValueError("Index not built. Call build_index first.")

        # クエリをトークン化
        query_tokens = self._tokenize(query)

        # BM25スコアを計算
        scores = self._get_scores(query_tokens)

        # Top-k結果を取得
        top_indices = np.argsort(scores)[::-1][:top_k]

        # 結果を返す（スコアが0より大きいもののみ）
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.chunk_metadata[idx], float(scores[idx])))

        return results

    def save(self, index_path: str) -> None:
        """
        BM25インデックスをディスクに保存する。

        Args:
            index_path: 保存先ディレクトリ。
        """
        path = Path(index_path)
        path.mkdir(parents=True, exist_ok=True)

        # BM25データを保存
        bm25_data = {
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
            "corpus_size": self.corpus_size,
            "avgdl": self.avgdl,
            "doc_freqs": self.doc_freqs,
            "idf": self.idf,
            "doc_len": self.doc_len,
            "vocabulary": list(self.vocabulary),
        }

        with open(path / "bm25_index.pkl", "wb") as f:
            pickle.dump(bm25_data, f)

        # メタデータを保存
        with open(path / "bm25_metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.chunk_metadata, f, ensure_ascii=False, indent=2)

    def load(self, index_path: str) -> None:
        """
        ディスクからBM25インデックスを読み込む。

        Args:
            index_path: 保存済みインデックスがあるディレクトリ。
        """
        path = Path(index_path)

        # BM25データを読み込み
        with open(path / "bm25_index.pkl", "rb") as f:
            bm25_data = pickle.load(f)

        self.k1 = bm25_data["k1"]
        self.b = bm25_data["b"]
        self.epsilon = bm25_data["epsilon"]
        self.corpus_size = bm25_data["corpus_size"]
        self.avgdl = bm25_data["avgdl"]
        self.doc_freqs = bm25_data["doc_freqs"]
        self.idf = bm25_data["idf"]
        self.doc_len = bm25_data["doc_len"]
        self.vocabulary = set(bm25_data["vocabulary"])

        # メタデータを読み込み
        with open(path / "bm25_metadata.json", "r", encoding="utf-8") as f:
            self.chunk_metadata = json.load(f)
