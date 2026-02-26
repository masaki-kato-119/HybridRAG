"""
検索モジュール。

クエリ前処理と Reciprocal Rank Fusion（RRF）によるハイブリッド検索を実装する。
"""

import re
from collections import defaultdict, OrderedDict
from typing import Any, DefaultDict, Dict, Hashable, List, Optional, Tuple

from .indexing import HybridIndex


def get_adaptive_candidate_multiplier(query: str) -> int:
    """
    クエリの長さに応じた retrieval_candidates_multiplier を返す。
    短いクエリは少なめ、長いクエリは多めの候補を取得して検索精度と速度のバランスを取る。

    Args:
        query: 検索クエリ文字列。

    Returns:
        2（短い）, 3（中）, 4（長い）のいずれか。
    """
    n = len(query.split())
    if n < 5:
        return 2
    if n < 10:
        return 3
    return 4


class QueryProcessor:
    """
    クエリの正規化と前処理を行う。

    空白の統一、不要記号の除去などを行う。
    """

    def __init__(self) -> None:
        # Common normalization patterns
        self.normalization_patterns = [
            (r"\s+", " "),  # Multiple spaces to single space
            (r"[^\w\s\-\.]", ""),  # Remove special characters except hyphens and dots
            (r"\.{2,}", "."),  # Multiple dots to single dot
        ]

    def normalize_query(self, query: str) -> str:
        """
        クエリを正規化する（空白・記号・小文字化）。

        Args:
            query: 元のクエリ文字列。

        Returns:
            正規化されたクエリ文字列。
        """
        normalized = query.strip().lower()

        # Apply normalization patterns
        for pattern, replacement in self.normalization_patterns:
            normalized = re.sub(pattern, replacement, normalized)

        return str(normalized).strip()

    def extract_keywords(self, query: str) -> List[str]:
        """
        クエリからキーワードを抽出する（ストップワード除去）。

        Args:
            query: クエリ文字列。

        Returns:
            キーワードのリスト。
        """
        # Simple keyword extraction - can be enhanced with NLP
        words = query.split()

        # Filter out common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
        }

        keywords = [word for word in words if word.lower() not in stop_words and len(word) > 2]
        return keywords


class MetadataFilterRetriever:
    """
    メタデータ条件で検索結果をフィルタする。

    辞書で key と value（または value のリスト）を指定し、
    メタデータが一致する結果のみ残す。
    """

    def __init__(self) -> None:
        pass

    def filter_by_metadata(
        self, results: List[Tuple[Dict, float]], filters: Optional[Dict] = None
    ) -> List[Tuple[Dict, float]]:
        """
        メタデータ条件で結果をフィルタする。

        Args:
            results: (metadata, score) のリスト。
            filters: メタデータの key -> value または key -> [values]。

        Returns:
            条件を満たす (metadata, score) のリスト。
        """
        if not filters:
            return results

        filtered_results = []

        for metadata, score in results:
            include = True

            for key, value in filters.items():
                if key in metadata:
                    if isinstance(value, list):
                        if metadata[key] not in value:
                            include = False
                            break
                    else:
                        if metadata[key] != value:
                            include = False
                            break

            if include:
                filtered_results.append((metadata, score))

        return filtered_results


class RRFRetriever:
    """
    Reciprocal Rank Fusion（RRF）で Dense と Sparse の結果を統合する検索器。

    各ランキングの順位に基づくスコア 1/(k+rank) を合算し、
    スコアの降順で統合結果を返す。
    """

    def __init__(
        self,
        hybrid_index: HybridIndex,
        k: int = 60,
        retrieval_candidates_multiplier: int = 2,
        enable_cache: bool = False,
        cache_size: int = 1000,
    ):
        """
        RRF 検索器を初期化する。

        Args:
            hybrid_index: Dense / Sparse 両方を用いるハイブリッドインデックス。
            k: RRF 定数（小さいほど上位ランクを重視）。
            retrieval_candidates_multiplier: Dense/Sparse それぞれが返す候補数
                = top_k × この値。デフォルト 2。
            enable_cache: True のとき同一クエリ結果を LRU キャッシュする。デフォルト False。
            cache_size: キャッシュするエントリ数の上限。enable_cache 時のみ有効。デフォルト 1000。
        """
        self.hybrid_index = hybrid_index
        self.k = k
        self.retrieval_candidates_multiplier = retrieval_candidates_multiplier
        self.query_processor = QueryProcessor()
        self.metadata_filter = MetadataFilterRetriever()
        # Simple in-memory LRU キャッシュ（頻出クエリ高速化用）
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self._cache: "OrderedDict[Hashable, Any]" = OrderedDict()

    def _make_cache_key(
        self,
        *,
        mode: str,
        normalized_queries: Tuple[str, ...],
        top_k: int,
        metadata_filters: Optional[Dict],
        retrieval_candidates_multiplier: int,
    ) -> Hashable:
        """
        キャッシュキー生成。

        - 正規化済みクエリ列
        - top_k
        - retrieval_candidates_multiplier
        - metadata_filters（ソートして順序依存を排除）
        """
        if metadata_filters:
            filters_tuple = tuple(sorted(metadata_filters.items()))
        else:
            filters_tuple = None

        return (
            mode,
            normalized_queries,
            top_k,
            retrieval_candidates_multiplier,
            filters_tuple,
        )

    def _get_from_cache(self, key: Hashable):
        if not self.enable_cache:
            return None
        if key not in self._cache:
            return None
        # LRU: 参照されたキーを末尾に移動
        self._cache.move_to_end(key)
        return self._cache[key]

    def _store_in_cache(self, key: Hashable, value: Any) -> None:
        if not self.enable_cache:
            return
        # 既存キーは更新して末尾へ
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
            return
        # 容量オーバーなら最古のエントリを削除
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        self._cache[key] = value

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        metadata_filters: Optional[Dict] = None,
        retrieval_candidates_multiplier: Optional[int] = None,
    ) -> List[Tuple[Dict, float]]:
        """
        Dense と Sparse の結果を RRF で統合して検索する。

        Args:
            query: 検索クエリ文字列。
            top_k: 返す結果数。
            dense_weight: Dense の重み（RRF では未使用）。
            sparse_weight: Sparse の重み（RRF では未使用）。
            metadata_filters: メタデータフィルタ。省略可。
            retrieval_candidates_multiplier: 各検索の候補数の上書き（top_k × この値）。大きいほど RRF の候補が増える。

        Returns:
            (メタデータ, RRF スコア) タプルのリスト。
        """
        mult = (
            retrieval_candidates_multiplier
            if retrieval_candidates_multiplier is not None
            else self.retrieval_candidates_multiplier
        )
        fetch_k = top_k * mult

        # Normalize query（キャッシュキーにも使用）
        normalized_query = self.query_processor.normalize_query(query)

        # キャッシュヒットなら即返す
        cache_key = self._make_cache_key(
            mode="single",
            normalized_queries=(normalized_query,),
            top_k=top_k,
            metadata_filters=metadata_filters,
            retrieval_candidates_multiplier=mult,
        )
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        # Get results from both retrievers
        dense_results = self.hybrid_index.dense_index.search(normalized_query, fetch_k)
        sparse_results = self.hybrid_index.sparse_index.search(normalized_query, fetch_k)

        # Apply RRF
        rrf_results = self._apply_rrf(dense_results, sparse_results)

        # Apply metadata filtering if specified
        if metadata_filters:
            rrf_results = self.metadata_filter.filter_by_metadata(rrf_results, metadata_filters)

        # Return top-k results
        final_results = rrf_results[:top_k]

        # キャッシュに保存
        self._store_in_cache(cache_key, final_results)
        return final_results

    def _get_unique_key(self, metadata: Dict) -> str:
        """メタデータから重複判定用の一意キーを返す。"""
        if "content" in metadata and metadata["content"]:
            return str(metadata["content"])
        return f"{metadata.get('doc_id', '')}_{metadata.get('chunk_index', '')}"

    def retrieve_multi(
        self,
        queries: List[str],
        top_k: int = 10,
        metadata_filters: Optional[Dict] = None,
        retrieval_candidates_multiplier: Optional[int] = None,
    ) -> List[Tuple[Dict, float]]:
        """
        複数クエリで Dense/Sparse 検索を行い、全結果を RRF で統合して返す。
        クエリ拡張（Query Expansion）と組み合わせて使用する。

        Args:
            queries: 検索に使うクエリのリスト（先頭は元クエリ、以降は拡張キーワード推奨）。
            top_k: 返す結果数。
            metadata_filters: メタデータフィルタ。省略可。
            retrieval_candidates_multiplier: 各クエリの候補数の倍率（top_k × この値）。

        Returns:
            (メタデータ, RRF スコア) タプルのリスト。
        """
        mult = (
            retrieval_candidates_multiplier
            if retrieval_candidates_multiplier is not None
            else self.retrieval_candidates_multiplier
        )
        fetch_k = top_k * mult

        key_to_metadata: Dict[str, Dict] = {}
        rrf_scores: DefaultDict[str, float] = defaultdict(float)

        # 全クエリを正規化しておく（キャッシュキー兼用）
        normalized_queries: List[str] = []

        for query in queries:
            normalized = self.query_processor.normalize_query(query)
            normalized_queries.append(normalized)
            dense_results = self.hybrid_index.dense_index.search(normalized, fetch_k)
            sparse_results = self.hybrid_index.sparse_index.search(normalized, fetch_k)

            for rank, (metadata, _) in enumerate(dense_results, 1):
                key = self._get_unique_key(metadata)
                key_to_metadata[key] = metadata
                rrf_scores[key] += 1.0 / (self.k + rank)

            for rank, (metadata, _) in enumerate(sparse_results, 1):
                key = self._get_unique_key(metadata)
                key_to_metadata[key] = metadata
                rrf_scores[key] += 1.0 / (self.k + rank)

        # キャッシュキー作成（クエリ拡張も含めて完全一致した場合のみヒット）
        cache_key = self._make_cache_key(
            mode="multi",
            normalized_queries=tuple(normalized_queries),
            top_k=top_k,
            metadata_filters=metadata_filters,
            retrieval_candidates_multiplier=mult,
        )
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        final_results = [
            (key_to_metadata[key], rrf_score) for key, rrf_score in sorted_results
        ]

        if metadata_filters:
            final_results = self.metadata_filter.filter_by_metadata(
                final_results, metadata_filters
            )

        final_results = final_results[:top_k]
        self._store_in_cache(cache_key, final_results)
        return final_results

    def _apply_rrf(
        self, dense_results: List[Tuple[Dict, float]], sparse_results: List[Tuple[Dict, float]]
    ) -> List[Tuple[Dict, float]]:
        """
        Reciprocal Rank Fusion で Dense と Sparse の結果を統合する。

        Args:
            dense_results: Dense 検索結果の (メタデータ, スコア) リスト。
            sparse_results: Sparse 検索結果の (メタデータ, スコア) リスト。

        Returns:
            RRF スコアでソートした (メタデータ, RRF スコア) のリスト。
        """
        key_to_metadata: Dict[str, Dict] = {}
        rrf_scores: DefaultDict[str, float] = defaultdict(float)

        for rank, (metadata, score) in enumerate(dense_results, 1):
            key = self._get_unique_key(metadata)
            key_to_metadata[key] = metadata
            rrf_scores[key] += 1.0 / (self.k + rank)

        for rank, (metadata, score) in enumerate(sparse_results, 1):
            key = self._get_unique_key(metadata)
            key_to_metadata[key] = metadata
            rrf_scores[key] += 1.0 / (self.k + rank)

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        final_results = [
            (key_to_metadata[key], rrf_score) for key, rrf_score in sorted_results
        ]
        return final_results

    def get_retrieval_stats(self, query: str, top_k: int = 10) -> Dict:
        """
        検索の詳細統計を取得する（Dense/Sparse/重複数など）。

        Args:
            query: 検索クエリ。
            top_k: 取得件数。

        Returns:
            クエリ・件数・オーバーラップ数等の辞書。
        """
        normalized_query = self.query_processor.normalize_query(query)

        # Get results from both retrievers
        dense_results = self.hybrid_index.dense_index.search(normalized_query, top_k * 2)
        sparse_results = self.hybrid_index.sparse_index.search(normalized_query, top_k * 2)

        # Apply RRF
        rrf_results = self._apply_rrf(dense_results, sparse_results)[:top_k]

        # Calculate statistics
        def get_content_key(metadata: Dict) -> str:
            """コンテンツ比較用の一意キーを返す。"""
            if "content" in metadata and metadata["content"]:
                return str(metadata["content"])
            return f"{metadata.get('doc_id', '')}_{metadata.get('chunk_index', '')}"

        dense_contents = {get_content_key(metadata) for metadata, _ in dense_results}
        sparse_contents = {get_content_key(metadata) for metadata, _ in sparse_results}
        rrf_contents = {get_content_key(metadata) for metadata, _ in rrf_results}

        stats = {
            "query": query,
            "normalized_query": normalized_query,
            "dense_results_count": len(dense_results),
            "sparse_results_count": len(sparse_results),
            "rrf_results_count": len(rrf_results),
            "dense_only_count": len(dense_contents - sparse_contents),
            "sparse_only_count": len(sparse_contents - dense_contents),
            "overlap_count": len(dense_contents & sparse_contents),
            "rrf_from_dense_only": len(rrf_contents & (dense_contents - sparse_contents)),
            "rrf_from_sparse_only": len(rrf_contents & (sparse_contents - dense_contents)),
            "rrf_from_overlap": len(rrf_contents & (dense_contents & sparse_contents)),
        }

        return stats


class EnsembleRetriever:
    """
    Dense と Sparse のスコアを重み付きで線形結合する検索器。

    RRF の代わりに、正規化したスコアに重みを掛けて合算する。
    """

    def __init__(self, hybrid_index: HybridIndex) -> None:
        """
        EnsembleRetriever を初期化する。

        Args:
            hybrid_index: Dense / Sparse 両方を持つ HybridIndex インスタンス。
        """
        self.hybrid_index = hybrid_index
        self.query_processor = QueryProcessor()
        self.metadata_filter = MetadataFilterRetriever()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        metadata_filters: Optional[Dict] = None,
    ) -> List[Tuple[Dict, float]]:
        """
        Dense と Sparse の結果を重み付きで線形結合して検索する。

        Args:
            query: 検索クエリ文字列。
            top_k: 返す結果数。
            dense_weight: Dense スコアの重み。
            sparse_weight: Sparse スコアの重み。
            metadata_filters: メタデータフィルタ（省略可）。

        Returns:
            (メタデータ, スコア) タプルのリスト。
        """
        # Normalize query
        normalized_query = self.query_processor.normalize_query(query)

        # Get results from both retrievers
        dense_results = self.hybrid_index.dense_index.search(normalized_query, top_k * 2)
        sparse_results = self.hybrid_index.sparse_index.search(normalized_query, top_k * 2)

        # Combine results with weights
        def get_unique_key(metadata: Dict) -> str:
            """Get unique key for deduplication."""
            if "content" in metadata and metadata["content"]:
                return str(metadata["content"])
            return f"{metadata.get('doc_id', '')}_{metadata.get('chunk_index', '')}"

        combined_scores: DefaultDict[str, float] = defaultdict(float)
        key_to_metadata: Dict[str, Dict] = {}

        # Normalize and weight dense scores
        if dense_results:
            max_dense_score = max(score for _, score in dense_results)
            for metadata, score in dense_results:
                key = get_unique_key(metadata)
                key_to_metadata[key] = metadata
                normalized_score = score / max_dense_score if max_dense_score > 0 else 0
                combined_scores[key] += dense_weight * normalized_score

        # Normalize and weight sparse scores
        if sparse_results:
            max_sparse_score = max(score for _, score in sparse_results)
            for metadata, score in sparse_results:
                key = get_unique_key(metadata)
                key_to_metadata[key] = metadata
                normalized_score = score / max_sparse_score if max_sparse_score > 0 else 0
                combined_scores[key] += sparse_weight * normalized_score

        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Convert to expected format
        final_results = []
        for key, combined_score in sorted_results:
            metadata = key_to_metadata[key]
            final_results.append((metadata, combined_score))

        # Apply metadata filtering if specified
        if metadata_filters:
            final_results = self.metadata_filter.filter_by_metadata(final_results, metadata_filters)

        return final_results[:top_k]
