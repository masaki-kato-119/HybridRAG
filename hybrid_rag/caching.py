"""
クエリ結果キャッシング機能

RAGシステムのクエリ結果をキャッシュして、同一または類似クエリの高速化を実現します。
"""

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


class QueryCache:
    """
    LRU（Least Recently Used）戦略を使用したクエリ結果キャッシュ。

    キャッシュキーは以下の要素から生成されます：
    - クエリ文字列
    - top_k
    - rerank_top_k
    - metadata_filters
    - content_keywords

    Attributes:
        cache_size: キャッシュに保持する最大エントリ数
        ttl_seconds: キャッシュエントリの有効期限（秒）。Noneの場合は無期限
        cache: キャッシュデータを保持するOrderedDict
        stats: キャッシュ統計情報
    """

    def __init__(self, cache_size: int = 1000, ttl_seconds: Optional[int] = 3600):
        """
        QueryCacheを初期化する。

        Args:
            cache_size: キャッシュに保持する最大エントリ数（デフォルト: 1000）
            ttl_seconds: キャッシュエントリの有効期限（秒）。Noneの場合は無期限（デフォルト: 3600秒 = 1時間）
        """
        self.cache_size = cache_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0,
        }

    def _generate_cache_key(
        self,
        query: str,
        top_k: int,
        rerank_top_k: int,
        metadata_filters: Optional[Dict] = None,
        content_keywords: Optional[list] = None,
        retrieval_candidates_multiplier: Optional[int] = None,
    ) -> str:
        """
        クエリパラメータからキャッシュキーを生成する。

        Args:
            query: クエリ文字列
            top_k: 最終結果数
            rerank_top_k: 再ランキング前の候補数
            metadata_filters: メタデータフィルタ
            content_keywords: コンテンツキーワード
            retrieval_candidates_multiplier: 候補倍率

        Returns:
            MD5ハッシュによるキャッシュキー
        """
        # パラメータを正規化してJSON文字列に変換
        key_data = {
            "query": query.strip().lower(),  # 正規化
            "top_k": top_k,
            "rerank_top_k": rerank_top_k,
            "metadata_filters": metadata_filters or {},
            "content_keywords": sorted(content_keywords) if content_keywords else [],
            "retrieval_candidates_multiplier": retrieval_candidates_multiplier,
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(
        self,
        query: str,
        top_k: int,
        rerank_top_k: int,
        metadata_filters: Optional[Dict] = None,
        content_keywords: Optional[list] = None,
        retrieval_candidates_multiplier: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        キャッシュから結果を取得する。

        Args:
            query: クエリ文字列
            top_k: 最終結果数
            rerank_top_k: 再ランキング前の候補数
            metadata_filters: メタデータフィルタ
            content_keywords: コンテンツキーワード
            retrieval_candidates_multiplier: 候補倍率

        Returns:
            キャッシュされた結果、またはNone（キャッシュミス時）
        """
        cache_key = self._generate_cache_key(
            query,
            top_k,
            rerank_top_k,
            metadata_filters,
            content_keywords,
            retrieval_candidates_multiplier,
        )

        if cache_key not in self.cache:
            self.stats["misses"] += 1
            return None

        result, timestamp = self.cache[cache_key]

        # TTLチェック
        if self.ttl_seconds is not None:
            age = time.time() - timestamp
            if age > self.ttl_seconds:
                # 期限切れ
                del self.cache[cache_key]
                self.stats["expired"] += 1
                self.stats["misses"] += 1
                return None

        # LRU: アクセスされたエントリを最後に移動
        self.cache.move_to_end(cache_key)
        self.stats["hits"] += 1

        return result

    def put(
        self,
        query: str,
        top_k: int,
        rerank_top_k: int,
        result: Dict[str, Any],
        metadata_filters: Optional[Dict] = None,
        content_keywords: Optional[list] = None,
        retrieval_candidates_multiplier: Optional[int] = None,
    ) -> None:
        """
        結果をキャッシュに保存する。

        Args:
            query: クエリ文字列
            top_k: 最終結果数
            rerank_top_k: 再ランキング前の候補数
            result: キャッシュする結果
            metadata_filters: メタデータフィルタ
            content_keywords: コンテンツキーワード
            retrieval_candidates_multiplier: 候補倍率
        """
        cache_key = self._generate_cache_key(
            query,
            top_k,
            rerank_top_k,
            metadata_filters,
            content_keywords,
            retrieval_candidates_multiplier,
        )

        # キャッシュサイズ制限チェック
        if len(self.cache) >= self.cache_size:
            # LRU: 最も古いエントリを削除
            self.cache.popitem(last=False)
            self.stats["evictions"] += 1

        # 新しいエントリを追加（タイムスタンプ付き）
        self.cache[cache_key] = (result, time.time())

    def clear(self) -> None:
        """キャッシュをクリアする。"""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        キャッシュ統計情報を取得する。

        Returns:
            統計情報の辞書（hits, misses, hit_rate, size, evictions, expired）
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0

        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.cache_size,
            "evictions": self.stats["evictions"],
            "expired": self.stats["expired"],
        }

    def reset_stats(self) -> None:
        """統計情報をリセットする。"""
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0,
        }
