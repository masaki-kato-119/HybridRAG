"""
再ランキングモジュール。

Cross-encoder を用いた再ランキングで関連度を改善する。
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """クエリとチャンクの関連度を Cross-encoder で再ランキングする。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        batch_size: int = 32,
    ):
        """
        Cross-encoder 再ランカーを初期化する。

        Args:
            model_name: HuggingFace の Cross-encoder モデル名。
            batch_size: 推論時のバッチサイズ。大きいとメモリ増、小さいとオーバーヘッド増。
        """
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        results: List[Tuple[Dict, float]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[Tuple[Dict, float]]:
        """
        Cross-encoder で検索結果を再ランキングする。
        推論は batch_size ごとに分割して実行し、メモリと速度のバランスを取る。

        Args:
            query: 元の検索クエリ。
            results: 検索結果の (メタデータ, スコア) タプルのリスト。
            top_k: 返す上位件数。None の場合は全て返す。
            batch_size: 推論バッチサイズ。None の場合はインスタンスの batch_size を使用。

        Returns:
            再ランキング後の (メタデータ, 再ランクスコア) タプルのリスト。
        """
        if not results:
            return []

        bs = batch_size if batch_size is not None else self.batch_size

        # Prepare query-chunk pairs for cross-encoder
        pairs = []
        metadatas = []

        for metadata, _ in results:
            chunk_text = metadata.get("content") or ""
            pairs.append([query, chunk_text])
            metadatas.append(metadata)

        # Get cross-encoder scores in batches (30-50% faster, lower memory spikes)
        all_scores: List[float] = []
        for i in range(0, len(pairs), bs):
            batch = pairs[i : i + bs]
            batch_scores = self.model.predict(batch)
            batch_scores = np.asarray(batch_scores).flatten()
            all_scores.extend(batch_scores.tolist())

        scores = np.asarray(all_scores)

        # Combine metadata with new scores
        reranked_results = list(zip(metadatas, scores))

        # Sort by cross-encoder score (descending)
        reranked_results.sort(key=lambda x: x[1], reverse=True)

        # Return top-k if specified
        if top_k is not None:
            reranked_results = reranked_results[:top_k]

        return reranked_results

    def get_rerank_stats(self, query: str, results: List[Tuple[Dict, float]]) -> Dict:
        """
        再ランキングの統計を取得する（分析用）。

        Args:
            query: 検索クエリ。
            results: 検索結果の (メタデータ, スコア) タプルのリスト。

        Returns:
            統計の辞書（original_count, reranked_count, score_changes など）。
        """
        if not results:
            return {"original_count": 0, "reranked_count": 0, "score_changes": []}

        # Get original and reranked results
        reranked = self.rerank(query, results)

        # Calculate score changes
        original_order = {metadata["content"]: i for i, (metadata, _) in enumerate(results)}
        reranked_order = {metadata["content"]: i for i, (metadata, _) in enumerate(reranked)}

        score_changes = []
        for content in original_order:
            if content in reranked_order:
                position_change = original_order[content] - reranked_order[content]
                score_changes.append(position_change)

        stats = {
            "original_count": len(results),
            "reranked_count": len(reranked),
            "avg_position_change": np.mean(score_changes) if score_changes else 0,
            "max_position_change": max(score_changes) if score_changes else 0,
            "min_position_change": min(score_changes) if score_changes else 0,
            "rerank_scores": [float(score) for _, score in reranked],
        }

        return stats


class LLMReranker:
    """LLM を用いた再ランカー（将来実装用のプレースホルダ）。"""

    def __init__(self, llm_client: Optional[Any] = None) -> None:
        """
        LLM 再ランカーを初期化する。

        Args:
            llm_client: LLM クライアント（OpenAI, Anthropic など）。省略可。
        """
        self.llm_client = llm_client

    def rerank(
        self, query: str, results: List[Tuple[Dict, float]], top_k: Optional[int] = None
    ) -> List[Tuple[Dict, float]]:
        """
        LLM で検索結果を再ランキングする（プレースホルダ実装）。

        実装時はクエリとチャンクを LLM に送り、関連度スコアを得て並べ替える想定。

        Args:
            query: 検索クエリ。
            results: 検索結果の (メタデータ, スコア) タプルのリスト。
            top_k: 返す上位件数。None の場合は全て返す。

        Returns:
            再ランキング後の (メタデータ, スコア) タプルのリスト。
        """
        # Placeholder: return original results
        # In real implementation, you would:
        # 1. Format query and chunks for LLM
        # 2. Send to LLM with scoring prompt
        # 3. Parse LLM response for scores
        # 4. Re-rank based on LLM scores

        if top_k is not None:
            return results[:top_k]
        return results


class HybridReranker:
    """複数の再ランキング手法を組み合わせる。"""

    def __init__(
        self,
        use_cross_encoder: bool = True,
        use_llm: bool = False,
        cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
        rerank_batch_size: int = 32,
    ):
        """
        ハイブリッド再ランカーを初期化する。

        Args:
            use_cross_encoder: Cross-encoder による再ランキングを行うかどうか。
            use_llm: LLM による再ランキングを行うかどうか。
            cross_encoder_model: Cross-encoder のモデル名。
            rerank_batch_size: Cross-encoder 推論のバッチサイズ。
        """
        self.use_cross_encoder = use_cross_encoder
        self.use_llm = use_llm

        if use_cross_encoder:
            self.cross_encoder = CrossEncoderReranker(
                cross_encoder_model, batch_size=rerank_batch_size
            )

        if use_llm:
            self.llm_reranker = LLMReranker()

    def rerank(
        self, query: str, results: List[Tuple[Dict, float]], top_k: Optional[int] = None
    ) -> List[Tuple[Dict, float]]:
        """
        ハイブリッド再ランキングを適用する。

        Args:
            query: 元の検索クエリ。
            results: 検索結果の (メタデータ, スコア) タプルのリスト。
            top_k: 返す上位件数。None の場合は全て返す。

        Returns:
            再ランキング後の (メタデータ, スコア) タプルのリスト。
        """
        current_results = results

        # Apply cross-encoder re-ranking (with batching)
        if self.use_cross_encoder and hasattr(self, "cross_encoder"):
            current_results = self.cross_encoder.rerank(query, current_results, top_k=top_k)

        # Apply LLM re-ranking
        if self.use_llm and hasattr(self, "llm_reranker"):
            current_results = self.llm_reranker.rerank(query, current_results)

        # Return top-k results (cross_encoder may already have truncated)
        if top_k is not None and len(current_results) > top_k:
            current_results = current_results[:top_k]

        return current_results

    def get_rerank_stats(self, query: str, results: List[Tuple[Dict, float]]) -> Dict:
        """
        再ランキングの総合統計を取得する。

        Args:
            query: 検索クエリ。
            results: 検索結果の (メタデータ, スコア) タプルのリスト。

        Returns:
            統計の辞書（original_count, methods_used, final_count, final_scores など）。
        """
        methods_used: List[str] = []
        stats: Dict[str, Any] = {"original_count": len(results), "methods_used": methods_used}

        if self.use_cross_encoder and hasattr(self, "cross_encoder"):
            ce_stats = self.cross_encoder.get_rerank_stats(query, results)
            stats["cross_encoder"] = ce_stats
            methods_used.append("cross_encoder")

        if self.use_llm and hasattr(self, "llm_reranker"):
            methods_used.append("llm")

        # Get final results
        final_results = self.rerank(query, results)
        stats["final_count"] = len(final_results)
        stats["final_scores"] = [float(score) for _, score in final_results]

        return stats
