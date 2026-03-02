"""
埋め込みキャッシングモジュール。

クエリ埋め込みをキャッシュして、同じクエリの埋め込み生成を高速化する。
"""

from collections import OrderedDict
from typing import Optional

import numpy as np


class EmbeddingCache:
    """
    クエリ埋め込みをキャッシュするLRUキャッシュ。

    同じクエリの埋め込みを再利用することで、モデル推論を回避し、
    90-99%の高速化を実現する。
    """

    def __init__(self, cache_size: int = 10000):
        """
        埋め込みキャッシュを初期化する。

        Args:
            cache_size: キャッシュに保持する最大エントリ数。デフォルト 10000。
        """
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cache_size = cache_size
        self.hits = 0
        self.misses = 0

    def _normalize_query(self, query: str) -> str:
        """
        クエリを正規化する（大文字小文字、前後の空白を統一）。

        Args:
            query: 元のクエリ文字列。

        Returns:
            正規化されたクエリ文字列。
        """
        return query.lower().strip()

    def _get_cache_key(self, query: str) -> str:
        """
        クエリからキャッシュキーを生成する。

        Args:
            query: クエリ文字列。

        Returns:
            キャッシュキー（正規化されたクエリ）。
        """
        return self._normalize_query(query)

    def get(self, query: str) -> Optional[np.ndarray]:
        """
        キャッシュから埋め込みを取得する。

        Args:
            query: クエリ文字列。

        Returns:
            キャッシュされた埋め込み。存在しない場合は None。
        """
        cache_key = self._get_cache_key(query)

        if cache_key in self.cache:
            # キャッシュヒット
            self.hits += 1
            # LRU: 最近使用したものを末尾に移動
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key].copy()  # コピーを返す

        # キャッシュミス
        self.misses += 1
        return None

    def put(self, query: str, embedding: np.ndarray) -> None:
        """
        埋め込みをキャッシュに保存する。

        Args:
            query: クエリ文字列。
            embedding: 埋め込みベクトル。
        """
        cache_key = self._get_cache_key(query)

        # キャッシュに保存（コピーを保存）
        self.cache[cache_key] = embedding.copy()

        # LRU: 最近使用したものを末尾に移動
        self.cache.move_to_end(cache_key)

        # キャッシュサイズ制限
        if len(self.cache) > self.cache_size:
            # 最も古いエントリを削除
            self.cache.popitem(last=False)

    def get_or_compute(self, query: str, compute_fn, *args, **kwargs) -> np.ndarray:
        """
        キャッシュから取得、なければ計算して保存。

        Args:
            query: クエリ文字列。
            compute_fn: 埋め込みを計算する関数。
            *args: compute_fn に渡す位置引数。
            **kwargs: compute_fn に渡すキーワード引数。

        Returns:
            埋め込みベクトル。
        """
        # キャッシュから取得を試みる
        embedding = self.get(query)

        if embedding is not None:
            # キャッシュヒット
            return embedding

        # キャッシュミス：計算
        embedding = compute_fn(*args, **kwargs)

        # キャッシュに保存
        self.put(query, embedding)

        return embedding

    def clear(self) -> None:
        """キャッシュをクリアする。"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> dict:
        """
        キャッシュ統計を取得する。

        Returns:
            統計情報の辞書（size, max_size, hits, misses, hit_rate）。
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "size": len(self.cache),
            "max_size": self.cache_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }

    def __len__(self) -> int:
        """キャッシュ内のエントリ数を返す。"""
        return len(self.cache)
