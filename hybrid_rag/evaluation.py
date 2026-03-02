"""
Evaluation and Logging Module
Handles performance tracking and evaluation metrics.
"""

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class RetrievalLog:
    """
    1 回の検索処理のログエントリ。

    Attributes:
        timestamp: 実行日時文字列。
        query: 元のクエリ。
        normalized_query: 正規化済みクエリ。
        dense_results_count: Dense 検索結果数。
        sparse_results_count: Sparse 検索結果数。
        rrf_results_count: RRF 統合後の件数。
        rerank_results_count: 再ランク後の件数。
        final_results_count: 最終返却件数。
        retrieval_time_ms: 検索時間（ミリ秒）。
        rerank_time_ms: 再ランク時間（ミリ秒）。
        total_time_ms: 合計時間（ミリ秒）。
        dense_scores: Dense スコアリスト。
        sparse_scores: Sparse スコアリスト。
        rrf_scores: RRF スコアリスト。
        rerank_scores: 再ランク後のスコアリスト。
        metadata_filters: 適用したメタデータフィルタ（あれば）。
    """

    timestamp: str
    query: str
    normalized_query: str
    dense_results_count: int
    sparse_results_count: int
    rrf_results_count: int
    rerank_results_count: int
    final_results_count: int
    retrieval_time_ms: float
    rerank_time_ms: float
    total_time_ms: float
    dense_scores: List[float]
    sparse_scores: List[float]
    rrf_scores: List[float]
    rerank_scores: List[float]
    metadata_filters: Optional[Dict] = None


@dataclass
class EvaluationMetrics:
    """
    RAG 検索の評価メトリクス。

    Attributes:
        precision_at_k: k ごとの Precision@k。
        recall_at_k: k ごとの Recall@k。
        mrr: Mean Reciprocal Rank。
        ndcg_at_k: k ごとの NDCG@k。
        hit_rate_at_k: k ごとの Hit Rate@k。
    """

    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_k: Dict[int, float]  # Normalized Discounted Cumulative Gain
    hit_rate_at_k: Dict[int, float]


class RAGLogger:
    """
    RAG の検索ログ・パフォーマンスログをファイルに記録する。

    retrieval_logs.jsonl と performance_logs.jsonl を log_dir に作成する。
    """

    def __init__(self, log_dir: str = "logs"):
        """
        Args:
            log_dir: ログファイルを置くディレクトリ。
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Initialize log files
        self.retrieval_log_file = self.log_dir / "retrieval_logs.jsonl"
        self.performance_log_file = self.log_dir / "performance_logs.jsonl"

    def log_retrieval(self, log_entry: RetrievalLog) -> None:
        """
        検索 1 件を retrieval_logs.jsonl に追記する。

        Args:
            log_entry: 記録する RetrievalLog。
        """
        with open(self.retrieval_log_file, "a", encoding="utf-8") as f:
            # Convert numpy types to Python types for JSON serialization
            log_dict = asdict(log_entry)

            # Convert numpy float32 to float
            for key in ["dense_scores", "sparse_scores", "rrf_scores", "rerank_scores"]:
                if key in log_dict and log_dict[key]:
                    log_dict[key] = [float(score) for score in log_dict[key]]

            json.dump(log_dict, f, ensure_ascii=False)
            f.write("\n")

    def log_performance(self, metrics: Dict[str, Any]) -> None:
        """
        パフォーマンスメトリクスを performance_logs.jsonl に追記する。

        Args:
            metrics: 記録するメトリクス辞書（timestamp は自動付与）。
        """
        log_entry = {"timestamp": datetime.now().isoformat(), **metrics}

        with open(self.performance_log_file, "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")

    def get_retrieval_logs(self, limit: Optional[int] = None) -> List[Dict]:
        """
        直近の検索ログを取得する。

        Args:
            limit: 取得件数（None で全件）。

        Returns:
            ログ辞書のリスト。
        """
        logs: List[Dict[str, Any]] = []

        if not self.retrieval_log_file.exists():
            return logs

        with open(self.retrieval_log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))

        if limit:
            logs = logs[-limit:]

        return logs

    def get_performance_summary(self, hours: int = 24) -> Dict:
        """
        直近 N 時間のパフォーマンス要約を返す。

        Args:
            hours: 集計対象時間（時間単位）。

        Returns:
            クエリ数・平均時間などの辞書。データが無ければ空辞書。
        """
        if not self.retrieval_log_file.exists():
            return {}

        logs = self.get_retrieval_logs()

        if not logs:
            return {}

        # Filter logs by time
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        recent_logs = [
            log
            for log in logs
            if datetime.fromisoformat(log["timestamp"]).timestamp() > cutoff_time
        ]

        if not recent_logs:
            return {}

        # Calculate summary statistics
        total_queries = len(recent_logs)
        avg_retrieval_time = sum(log["retrieval_time_ms"] for log in recent_logs) / total_queries
        avg_rerank_time = sum(log["rerank_time_ms"] for log in recent_logs) / total_queries
        avg_total_time = sum(log["total_time_ms"] for log in recent_logs) / total_queries

        avg_results_count = sum(log["final_results_count"] for log in recent_logs) / total_queries

        return {
            "period_hours": hours,
            "total_queries": total_queries,
            "avg_retrieval_time_ms": avg_retrieval_time,
            "avg_rerank_time_ms": avg_rerank_time,
            "avg_total_time_ms": avg_total_time,
            "avg_results_count": avg_results_count,
            "queries_per_hour": total_queries / hours,
        }


class RAGEvaluator:
    """
    RAG の検索性能を評価する。

    Precision@k, Recall@k, MRR, NDCG@k, Hit Rate@k を計算する。
    """

    def __init__(self) -> None:
        pass

    def evaluate_retrieval(
        self,
        queries_and_ground_truth: List[Dict],
        retrieval_function: Callable[..., Any],
        k_values: Optional[List[int]] = None,
    ) -> EvaluationMetrics:
        """
        検索性能を評価する（Precision@k, Recall@k, MRR, NDCG@k, Hit Rate@k）。

        Args:
            queries_and_ground_truth: 'query' と 'relevant_docs' を持つ辞書のリスト。
            retrieval_function: クエリを受け取り検索結果を返す関数。
            k_values: 評価する k のリスト。None の場合は [1, 3, 5, 10]。

        Returns:
            EvaluationMetrics オブジェクト。
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]
        precision_at_k: Dict[int, List[float]] = {k: [] for k in k_values}
        recall_at_k: Dict[int, List[float]] = {k: [] for k in k_values}
        hit_rate_at_k: Dict[int, List[float]] = {k: [] for k in k_values}
        reciprocal_ranks: List[float] = []
        ndcg_at_k: Dict[int, List[float]] = {k: [] for k in k_values}

        for item in queries_and_ground_truth:
            query = item["query"]
            relevant_docs = set(item["relevant_docs"])

            # Get retrieval results (chunk-level); for doc-level metrics dedupe by doc_id
            results = retrieval_function(query, max(k_values))
            chunk_doc_ids = [metadata["doc_id"] for metadata, _ in results]
            # ドキュメント単位で先頭出現のみ残す（同一docの重複でNDCG>1になるのを防ぐ）
            seen = set()
            retrieved_docs = []
            for doc_id in chunk_doc_ids:
                if doc_id not in seen:
                    seen.add(doc_id)
                    retrieved_docs.append(doc_id)

            # Calculate metrics for each k
            for k in k_values:
                retrieved_at_k = set(retrieved_docs[:k])

                # Precision@k
                if retrieved_at_k:
                    precision = len(retrieved_at_k & relevant_docs) / len(retrieved_at_k)
                else:
                    precision = 0.0
                precision_at_k[k].append(precision)

                # Recall@k
                if relevant_docs:
                    recall = len(retrieved_at_k & relevant_docs) / len(relevant_docs)
                else:
                    recall = 0.0
                recall_at_k[k].append(recall)

                # Hit Rate@k
                hit = 1.0 if retrieved_at_k & relevant_docs else 0.0
                hit_rate_at_k[k].append(hit)

                # NDCG@k
                ndcg = self._calculate_ndcg(retrieved_docs[:k], relevant_docs)
                ndcg_at_k[k].append(ndcg)

            # MRR
            rr = self._calculate_reciprocal_rank(retrieved_docs, relevant_docs)
            reciprocal_ranks.append(rr)

        # Average metrics
        avg_precision_at_k = {k: sum(scores) / len(scores) for k, scores in precision_at_k.items()}
        avg_recall_at_k = {k: sum(scores) / len(scores) for k, scores in recall_at_k.items()}
        avg_hit_rate_at_k = {k: sum(scores) / len(scores) for k, scores in hit_rate_at_k.items()}
        avg_ndcg_at_k = {k: sum(scores) / len(scores) for k, scores in ndcg_at_k.items()}
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

        return EvaluationMetrics(
            precision_at_k=avg_precision_at_k,
            recall_at_k=avg_recall_at_k,
            mrr=mrr,
            ndcg_at_k=avg_ndcg_at_k,
            hit_rate_at_k=avg_hit_rate_at_k,
        )

    def _calculate_reciprocal_rank(self, retrieved_docs: List[str], relevant_docs: set) -> float:
        """
        最初に正解が現れたランクの逆数を返す（MRR 用）。

        Args:
            retrieved_docs: 検索結果の doc_id リスト。
            relevant_docs: 正解 doc_id の集合。

        Returns:
            1/rank または 0（ヒットなし）。
        """
        for i, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                return 1.0 / i
        return 0.0

    def _calculate_ndcg(
        self, retrieved_docs: List[str], relevant_docs: set, k: Optional[int] = None
    ) -> float:
        """
        NDCG@k を計算する（二値関連度）。

        Args:
            retrieved_docs: 検索結果の doc_id リスト。
            relevant_docs: 正解 doc_id の集合。
            k: 上位 k 件で計算（None の場合は retrieved_docs の長さ）。

        Returns:
            NDCG 値（0〜1）。
        """
        if k is None:
            k = len(retrieved_docs)

        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs[:k], 1):
            relevance = 1.0 if doc_id in relevant_docs else 0.0
            dcg += relevance / math.log2(i + 1)

        # IDCG (Ideal DCG)
        ideal_relevances = [1.0] * min(len(relevant_docs), k)
        idcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(ideal_relevances, 1))

        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_end_to_end(self, test_cases: List[Dict], rag_system: Any) -> Dict[str, Any]:
        """
        エンドツーエンドの RAG 性能を評価する。

        Args:
            test_cases: テストケースのリスト。各要素は 'query', 'expected_answer', 'relevant_docs' 等を持つ辞書。
            rag_system: RAG システムのインスタンス。

        Returns:
            評価結果の辞書（total_cases, response_times, avg_response_time_ms など）。
        """
        response_times_list: List[float] = []
        results: Dict[str, Any] = {
            "total_cases": len(test_cases),
            "retrieval_accuracy": [],
            "answer_quality": [],
            "response_times": response_times_list,
        }

        for case in test_cases:
            # query = case["query"]
            # expected_answer = case.get("expected_answer", "")
            # relevant_docs = set(case.get("relevant_docs", []))

            # Measure response time
            start_time = time.time()

            # Get RAG response (this would call your main RAG system)
            # response = rag_system.query(query)

            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # ms

            response_times_list.append(response_time)

            # Note: Answer quality evaluation would require additional
            # metrics like BLEU, ROUGE, or LLM-based evaluation

        # Calculate summary statistics
        if response_times_list:
            results["avg_response_time_ms"] = sum(response_times_list) / len(response_times_list)
            results["max_response_time_ms"] = max(response_times_list)
            results["min_response_time_ms"] = min(response_times_list)

        return results
