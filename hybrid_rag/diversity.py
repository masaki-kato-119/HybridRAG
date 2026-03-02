"""
多様性選択モジュール。

MMR（Maximal Marginal Relevance）を使用して、関連度と多様性のバランスを取る。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


class MMRSelector:
    """
    MMR（Maximal Marginal Relevance）による多様性選択。

    関連度（relevance）と多様性（diversity）のバランスを取り、
    重複・冗長なチャンクを減らしてコンテキストの情報量を増やす。
    """

    def __init__(
        self,
        embedding_model: Optional[SentenceTransformer] = None,
        lambda_param: float = 0.6,
    ):
        """
        Args:
            embedding_model: 埋め込みモデル。Noneの場合は外部から埋め込みを渡す必要がある。
            lambda_param: 関連度と多様性のバランスパラメータ（0.0〜1.0）。
                1.0に近いほど関連度重視、0.0に近いほど多様性重視。
                推奨値: 0.5〜0.7
        """
        self.embedding_model = embedding_model
        self.lambda_param = lambda_param

    def select(
        self,
        results: List[Tuple[Dict, float]],
        top_k: int,
        query_embedding: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> List[Tuple[Dict, float]]:
        """
        MMRを使用して多様性を考慮した上位k件を選択する。

        Args:
            results: (メタデータ, 関連度スコア)のタプルのリスト。
            top_k: 選択する件数。
            query_embedding: クエリの埋め込みベクトル（オプション）。
            embeddings: 各結果の埋め込みベクトル（オプション）。
                Noneの場合は、メタデータの'content'から生成する。

        Returns:
            MMRで選択された(メタデータ, MMRスコア)のタプルのリスト。
        """
        if len(results) <= top_k:
            return results

        # 埋め込みの準備
        if embeddings is None:
            if self.embedding_model is None:
                raise ValueError("embedding_model is None and embeddings not provided")
            contents = [r[0].get("content", "") for r in results]
            embeddings = self.embedding_model.encode(
                contents, convert_to_numpy=True, show_progress_bar=False
            )
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # クエリ埋め込みの準備（関連度スコアを使う場合は不要）
        if query_embedding is not None:
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # MMR選択
        selected_indices = []
        remaining_indices = list(range(len(results)))

        for _ in range(min(top_k, len(results))):
            if not remaining_indices:
                break

            # 各候補のMMRスコアを計算
            mmr_scores = []
            for idx in remaining_indices:
                # 関連度スコア（既存の再ランクスコアを使用）
                relevance_score = results[idx][1]

                # 多様性スコア（既に選択されたものとの最大類似度）
                if selected_indices:
                    selected_embeddings = embeddings[selected_indices]
                    candidate_embedding = embeddings[idx : idx + 1]
                    similarities = np.dot(selected_embeddings, candidate_embedding.T).flatten()
                    max_similarity = np.max(similarities)
                else:
                    max_similarity = 0.0

                # MMRスコア = λ * relevance - (1-λ) * max_similarity
                mmr_score = (
                    self.lambda_param * relevance_score - (1 - self.lambda_param) * max_similarity
                )
                mmr_scores.append((idx, mmr_score))

            # 最大MMRスコアの候補を選択
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # 選択された結果を返す（MMRスコアで並べ替え）
        selected_results = [(results[idx][0], results[idx][1]) for idx in selected_indices]

        return selected_results


def get_mmr_selector(
    embedding_model: Optional[SentenceTransformer] = None,
    lambda_param: float = 0.6,
) -> MMRSelector:
    """
    MMRSelectorのファクトリー関数。

    Args:
        embedding_model: 埋め込みモデル。
        lambda_param: 関連度と多様性のバランスパラメータ（0.0〜1.0）。

    Returns:
        MMRSelectorインスタンス。
    """
    return MMRSelector(embedding_model=embedding_model, lambda_param=lambda_param)
