"""
メインのハイブリッド RAG システム。

全コンポーネントを組み合わせ、エンドツーエンドの RAG 機能を提供する。
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .caching import QueryCache
from .chunking import Chunk, SemanticChunker
from .context import ContextBuilder, PromptBuilder
from .diversity import MMRSelector
from .evaluation import RAGLogger, RetrievalLog
from .indexing import HybridIndex
from .ingestion import DocumentProcessor, SectionDetector
from .query_expansion import QueryExpander
from .reranking import HybridReranker
from .retrieval import RRFRetriever, get_adaptive_candidate_multiplier
from .storage import DatabaseManager


class HybridRAGSystem:
    """全コンポーネントを統括するハイブリッド RAG システムのメインクラス。"""

    def __init__(
        self,
        db_path: str = "hybrid_rag.db",
        index_path: str = "indices",
        log_dir: str = "logs",
        dense_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
        max_chunk_size: int = 512,
        max_context_tokens: int = 4000,
        rrf_k: int = 60,
        retrieval_candidates_multiplier: int = 2,
        query_expander: Optional[QueryExpander] = None,
        enable_cache: bool = True,
        cache_size: int = 1000,
        cache_ttl_seconds: Optional[int] = 3600,
        # FAISS最適化パラメータ
        index_type: str = "flat",
        nlist: int = 100,
        hnsw_m: int = 32,
        # Cross-encoder最適化パラメータ
        rerank_batch_size: int = 32,
        # 追加最適化パラメータ
        sparse_index_type: str = "bm25",
        enable_embedding_cache: bool = True,
        embedding_cache_size: int = 10000,
        enable_early_filtering: bool = True,
        enable_async: bool = True,
        # MMRパラメータ
        enable_mmr: bool = False,
        mmr_lambda: float = 0.6,
    ):
        """
        ハイブリッド RAG システムを初期化する。

        Args:
            db_path: SQLite データベースのパス。
            index_path: インデックスを保存するディレクトリのパス。
            log_dir: ログを出力するディレクトリ。
            dense_model: Dense 埋め込み用の Sentence Transformers モデル名。
            rerank_model: 再ランキング用の Cross-encoder モデル名。
            max_chunk_size: チャンクあたりの最大文字数。
            max_context_tokens: コンテキストに含める最大トークン数。
            rrf_k: RRF 定数（小さいほど上位ランクを重視）。デフォルト 60。
            retrieval_candidates_multiplier: Dense/Sparse それぞれが返す候補数
                = rerank_top_k × この値。デフォルト 2。
            query_expander: クエリ拡張用の QueryExpander インスタンス。
                指定時は query() の冒頭で LLM によりキーワードを拡張し複数クエリで検索する。省略時は拡張なし。
            enable_cache: クエリ結果キャッシングを有効にするかどうか。デフォルト True。
            cache_size: キャッシュに保持する最大エントリ数。デフォルト 1000。
            cache_ttl_seconds: キャッシュエントリの有効期限（秒）。Noneの場合は無期限。デフォルト 3600（1時間）。
            index_type: FAISSインデックスのタイプ（"flat", "ivf", "hnsw"）。デフォルト "flat"。
            nlist: IVFインデックスのクラスタ数。index_type="ivf"時のみ有効。デフォルト 100。
            hnsw_m: HNSWインデックスのM値。index_type="hnsw"時のみ有効。デフォルト 32。
            rerank_batch_size: Cross-encoder再ランキングのバッチサイズ。デフォルト 32。
            sparse_index_type: Sparseインデックスのタイプ（"tfidf", "bm25"）。
                デフォルト "bm25"。
            enable_embedding_cache: 埋め込みキャッシュを有効にするかどうか。
                デフォルト True。
            embedding_cache_size: 埋め込みキャッシュのサイズ。デフォルト 10000。
            enable_early_filtering: 段階的フィルタリングを有効にするかどうか。
                デフォルト True。
            enable_async: Dense/Sparse検索を非同期並列実行するかどうか。
                デフォルト True。
            enable_mmr: MMR（Maximal Marginal Relevance）による多様性選択を
                有効にするかどうか。デフォルト False。
            mmr_lambda: MMRのλパラメータ（0.0〜1.0）。1.0に近いほど関連度重視、
                0.0に近いほど多様性重視。デフォルト 0.6。
        """
        # Initialize components
        self.db_manager = DatabaseManager(db_path)
        self.doc_processor = DocumentProcessor()
        self.section_detector = SectionDetector()
        self.chunker = SemanticChunker(max_chunk_size=max_chunk_size)
        self.hybrid_index = HybridIndex(
            dense_model_name=dense_model,
            index_type=index_type,
            nlist=nlist,
            hnsw_m=hnsw_m,
            sparse_index_type=sparse_index_type,
        )
        # 埋め込みキャッシュの設定を適用
        if hasattr(self.hybrid_index.dense_index, "enable_embedding_cache"):
            self.hybrid_index.dense_index.enable_embedding_cache = enable_embedding_cache
            if enable_embedding_cache:
                from .embedding_cache import EmbeddingCache

                self.hybrid_index.dense_index.embedding_cache = EmbeddingCache(
                    cache_size=embedding_cache_size
                )

        self.retriever: Optional[RRFRetriever] = None  # Set after build_index/load_index
        self.reranker = HybridReranker(
            cross_encoder_model=rerank_model,
            rerank_batch_size=rerank_batch_size,
        )
        self.context_builder = ContextBuilder(max_tokens=max_context_tokens)
        self.prompt_builder = PromptBuilder()
        self.logger = RAGLogger(log_dir)
        self.rrf_k = rrf_k
        self.retrieval_candidates_multiplier = retrieval_candidates_multiplier
        self.query_expander = query_expander
        self.enable_early_filtering = enable_early_filtering
        self.enable_async = enable_async

        # MMR設定
        self.enable_mmr = enable_mmr
        self.mmr_selector = (
            MMRSelector(
                embedding_model=self.hybrid_index.dense_index.model,
                lambda_param=mmr_lambda,
            )
            if enable_mmr
            else None
        )

        # Query cache
        self.enable_cache = enable_cache
        self.query_cache = (
            QueryCache(cache_size=cache_size, ttl_seconds=cache_ttl_seconds)
            if enable_cache
            else None
        )

        # Paths
        self.index_path = Path(index_path)
        self.index_path.mkdir(exist_ok=True)

        # State
        self.is_indexed = False

    def ingest_documents(
        self, file_paths: List[Union[str, Path]], rebuild_index: bool = True
    ) -> Dict[str, Any]:
        """
        ドキュメントをシステムに投入する。

        Args:
            file_paths: 投入するファイルパスのリスト。
            rebuild_index: 投入後にインデックスを再構築するかどうか。

        Returns:
            投入結果の辞書（processed_documents, failed_documents, total_chunks, index_rebuilt）。
        """
        print("Starting document ingestion...")

        all_chunks = []
        processed_docs = 0
        failed_docs = 0

        for file_path in file_paths:
            try:
                print(f"Processing: {file_path}")

                # Process document
                document = self.doc_processor.process_file(file_path)

                # Store document metadata
                self.db_manager.store_document(document)

                # Chunk document
                chunks = self.chunker.chunk_document(document)

                # Store chunks
                self.db_manager.store_chunks(chunks)

                all_chunks.extend(chunks)
                processed_docs += 1

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
                failed_docs += 1

        # Rebuild index if requested
        if rebuild_index and all_chunks:
            print("Building hybrid index...")
            self.build_index()

        stats = {
            "processed_documents": processed_docs,
            "failed_documents": failed_docs,
            "total_chunks": len(all_chunks),
            "index_rebuilt": rebuild_index and all_chunks,
        }

        print(f"Ingestion complete: {stats}")
        return stats

    def build_index(self, batch_size: int = 1000) -> None:
        """
        DB 内の全チャンクからバッチ処理でハイブリッドインデックスを構築する。

        Args:
            batch_size: 1 バッチあたりのチャンク数。デフォルト 1000。
        """
        print("Loading chunks from database...")

        # Get total chunk count
        total_chunks = self.db_manager.get_chunk_count()

        if total_chunks == 0:
            raise ValueError("No chunks found in database. Ingest documents first.")

        print(f"Processing {total_chunks} chunks in batches of {batch_size}...")

        # Process chunks in batches
        chunk_batches = []
        offset = 0

        while offset < total_chunks:
            # Get batch from database
            chunk_dicts = self.db_manager.get_chunks_batch(offset, batch_size)

            if not chunk_dicts:
                break

            # Convert to Chunk objects
            from .chunking import ChunkType

            batch_chunks = []
            for chunk_dict in chunk_dicts:
                chunk = Chunk(
                    content=chunk_dict["content"],
                    doc_id=chunk_dict["doc_id"],
                    section=chunk_dict["section"],
                    chunk_index=chunk_dict["chunk_index"],
                    chunk_type=ChunkType(chunk_dict["chunk_type"]),
                    source_path=chunk_dict.get("source_path", ""),
                    metadata=chunk_dict.get("metadata"),
                )
                batch_chunks.append(chunk)

            chunk_batches.append(batch_chunks)
            offset += batch_size

            # Show progress
            if offset % (batch_size * 10) == 0 or offset >= total_chunks:
                print(f"Loaded {min(offset, total_chunks)}/{total_chunks} chunks...")

        # Build hybrid index using batch processing
        self.hybrid_index.build_index_batch(chunk_batches, show_progress=True)

        # Save index
        self.hybrid_index.save(str(self.index_path))

        # Initialize retriever（キャッシュ有効化）
        self.retriever = RRFRetriever(
            self.hybrid_index,
            k=self.rrf_k,
            retrieval_candidates_multiplier=self.retrieval_candidates_multiplier,
            enable_cache=True,
            cache_size=1000,
            enable_async=self.enable_async,
        )

        self.is_indexed = True
        print(f"Index built successfully with {total_chunks} chunks!")

    def load_index(self) -> None:
        """
        ディスク上の既存インデックスを読み込む。

        Raises:
            FileNotFoundError: インデックスファイルが存在しない場合。
        """
        if not (self.index_path / "dense_index.faiss").exists():
            raise FileNotFoundError("Index not found. Build index first.")

        print("Loading existing index...")
        self.hybrid_index.load(str(self.index_path))
        self.retriever = RRFRetriever(
            self.hybrid_index,
            k=self.rrf_k,
            retrieval_candidates_multiplier=self.retrieval_candidates_multiplier,
            enable_cache=True,
            cache_size=1000,
            enable_async=self.enable_async,
        )
        self.is_indexed = True
        print("Index loaded successfully!")

    def query(
        self,
        query: str,
        top_k: int = 5,
        rerank_top_k: int = 10,
        include_metadata: bool = True,
        include_scores: bool = False,
        metadata_filters: Optional[Dict] = None,
        retrieval_candidates_multiplier: Optional[int] = None,
        content_keywords: Optional[List[str]] = None,
        use_adaptive_candidates: bool = False,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        RAG システムにクエリを実行する。

        Args:
            query: ユーザーのクエリ文字列。
            top_k: 再ランキング後に返す最終結果数。
            rerank_top_k: 再ランキング前に取得する候補数。大きいほど再ランキングの選択肢が増える。
            include_metadata: コンテキストにメタデータを含めるかどうか。
            include_scores: コンテキストに関連度スコアを含めるかどうか。
            metadata_filters: メタデータフィルタ（doc_id, source_path, section,
                chunk_type など出典条件）。
            retrieval_candidates_multiplier: 検索時の候補倍率の上書き。None で
                use_adaptive_candidates 時はクエリ長に応じて自動。
            content_keywords: 指定時、チャンク本文がどれか1つでも含むものだけに
                絞ってから再ランキング。ヒット率向上用。
            use_adaptive_candidates: True のとき、クエリ長に応じて候補倍率を
                2/3/4 で動的調整（retrieval_candidates_multiplier 未指定時）。
            use_cache: キャッシュを使用するかどうか。デフォルト True。

        Returns:
            クエリ結果の辞書（query, context, prompts, results, stats）。
        """
        if not self.is_indexed:
            try:
                self.load_index()
            except FileNotFoundError:
                raise ValueError("No index found. Ingest documents and build index first.")

        # キャッシュチェック
        if use_cache and self.query_cache is not None:
            cached_result = self.query_cache.get(
                query=query,
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                metadata_filters=metadata_filters,
                content_keywords=content_keywords,
                retrieval_candidates_multiplier=retrieval_candidates_multiplier,
            )
            if cached_result is not None:
                # キャッシュヒット: コピーを作成してfrom_cacheフラグを追加
                import copy

                result_copy = copy.deepcopy(cached_result)
                result_copy["stats"]["from_cache"] = True
                return result_copy

        start_time = time.time()

        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call build_index() or load_index() first.")

        # クエリ拡張: QueryExpander があれば LLM でキーワードを拡張し複数クエリに
        if self.query_expander is not None:
            queries = self.query_expander.expand(query)
        else:
            queries = [query]

        # 候補数の動的調整（4.1.2）
        if use_adaptive_candidates and retrieval_candidates_multiplier is None:
            retrieval_candidates_multiplier = get_adaptive_candidate_multiplier(query)

        # Retrieval phase（複数クエリの場合は retrieve_multi で RRF 統合）
        retrieval_start = time.time()
        if len(queries) > 1:
            results = self.retriever.retrieve_multi(
                queries,
                top_k=rerank_top_k,
                metadata_filters=metadata_filters,
                retrieval_candidates_multiplier=retrieval_candidates_multiplier,
            )
        else:
            results = self.retriever.retrieve(
                query,
                top_k=rerank_top_k,
                metadata_filters=metadata_filters,
                retrieval_candidates_multiplier=retrieval_candidates_multiplier,
                early_filtering=self.enable_early_filtering,
            )

        # Ensure content is available in metadata (4.1.3: 一括取得で N+1 回避)
        need_content = [(m["doc_id"], m["chunk_index"]) for m, _ in results if not m.get("content")]
        content_map = self.db_manager.get_contents_batch(need_content) if need_content else {}
        results_with_content = []
        for metadata, score in results:
            key = (metadata["doc_id"], metadata["chunk_index"])
            if key in content_map:
                metadata["content"] = content_map[key]
            results_with_content.append((metadata, score))
        results = results_with_content

        # Content keyword filter: 本文が指定キーワードのいずれかを含むチャンクだけに絞る
        if content_keywords:
            filtered = [
                (m, s)
                for m, s in results
                if (m.get("content") and any(kw in m["content"] for kw in content_keywords))
            ]
            if filtered:
                results = filtered
            # 1件も残らなければフィルターなしの結果のまま（落ちこぼれ防止）

        retrieval_time = (time.time() - retrieval_start) * 1000

        # Re-ranking phase
        rerank_start = time.time()
        reranked_results = self.reranker.rerank(query, results, top_k=top_k)
        rerank_time = (time.time() - rerank_start) * 1000

        # MMR phase (optional): 多様性を考慮した選択
        if self.enable_mmr and self.mmr_selector is not None and len(reranked_results) > top_k:
            mmr_start = time.time()
            reranked_results = self.mmr_selector.select(
                results=reranked_results, top_k=top_k, embeddings=None
            )
            mmr_time = (time.time() - mmr_start) * 1000
        else:
            mmr_time = 0.0

        # Context building
        context = self.context_builder.build_context(
            reranked_results, include_metadata=include_metadata, include_scores=include_scores
        )

        # Prompt building
        prompts = self.prompt_builder.build_prompt_with_citations(query, context)

        total_time = (time.time() - start_time) * 1000

        # Log retrieval
        log_entry = RetrievalLog(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            query=query,
            normalized_query=self.retriever.query_processor.normalize_query(query),
            dense_results_count=len(results),
            sparse_results_count=len(results),  # Same for RRF
            rrf_results_count=len(results),
            rerank_results_count=len(reranked_results),
            final_results_count=len(reranked_results),
            retrieval_time_ms=retrieval_time,
            rerank_time_ms=rerank_time,
            total_time_ms=total_time,
            dense_scores=[],  # Would need to track separately
            sparse_scores=[],
            rrf_scores=[score for _, score in results],
            rerank_scores=[score for _, score in reranked_results],
            metadata_filters=metadata_filters,
        )
        self.logger.log_retrieval(log_entry)

        # Also log to database
        self.db_manager.log_retrieval(
            query=query,
            normalized_query=log_entry.normalized_query,
            results_count=len(reranked_results),
            retrieval_time_ms=retrieval_time,
            rerank_time_ms=rerank_time,
            total_time_ms=total_time,
            metadata={"filters": metadata_filters},
        )

        result = {
            "query": query,
            "context": context,
            "prompts": prompts,
            "results": reranked_results,
            "stats": {
                "retrieval_time_ms": retrieval_time,
                "rerank_time_ms": rerank_time,
                "mmr_time_ms": mmr_time if self.enable_mmr else 0.0,
                "total_time_ms": total_time,
                "results_count": len(reranked_results),
                "from_cache": False,
            },
        }

        # キャッシュに保存
        if use_cache and self.query_cache is not None:
            self.query_cache.put(
                query=query,
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                result=result,
                metadata_filters=metadata_filters,
                content_keywords=content_keywords,
                retrieval_candidates_multiplier=retrieval_candidates_multiplier,
            )

        return result

    def get_system_stats(self) -> Dict[str, Any]:
        """
        システム全体の統計を取得する。

        Returns:
            データベース・検索・インデックス状態などを含む辞書。
        """
        db_stats = self.db_manager.get_database_stats()
        retrieval_stats = self.db_manager.get_retrieval_stats(24)

        stats = {
            "database": db_stats,
            "retrieval_24h": retrieval_stats,
            "index_status": self.is_indexed,
            "components": {
                "chunker_max_size": self.chunker.max_chunk_size,
                "context_max_tokens": self.context_builder.max_tokens,
                "dense_model": (
                    getattr(self.hybrid_index.dense_index.model, "_modules", {})
                    .get("0", {})
                    .auto_model.name_or_path
                    if hasattr(self.hybrid_index.dense_index, "model")
                    else "Unknown"
                ),
                "rerank_model": (
                    self.reranker.cross_encoder.model_name
                    if hasattr(self.reranker, "cross_encoder")
                    else "Unknown"
                ),
            },
        }

        # キャッシュ統計を追加
        if self.query_cache is not None:
            stats["cache"] = self.query_cache.get_stats()

        return stats

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """
        クエリキャッシュの統計情報を取得する。

        Returns:
            キャッシュ統計の辞書、またはキャッシュが無効の場合はNone。
        """
        if self.query_cache is None:
            return None
        return self.query_cache.get_stats()

    def clear_cache(self) -> None:
        """クエリキャッシュをクリアする。"""
        if self.query_cache is not None:
            self.query_cache.clear()
            print("Query cache cleared.")
        else:
            print("Query cache is not enabled.")

    def delete_document(self, doc_id: str, rebuild_index: bool = True) -> None:
        """
        ドキュメントを削除し、必要に応じてインデックスを再構築する。

        Args:
            doc_id: 削除するドキュメントの ID。
            rebuild_index: 削除後にインデックスを再構築するかどうか。デフォルト True。
        """
        self.db_manager.delete_document(doc_id)

        if rebuild_index:
            print(f"Document {doc_id} deleted. Rebuilding index...")
            self.build_index()

    def search_documents(self, query: str, doc_type: Optional[str] = None) -> List[Dict]:
        """
        コンテンツまたはメタデータでドキュメントを検索する。

        Args:
            query: 検索クエリ文字列。
            doc_type: 絞り込むドキュメントタイプ（指定時のみ）。省略可。

        Returns:
            ドキュメント単位の辞書のリスト（doc_id, source_path, max_score, chunk_count など）。
        """
        if not self.is_indexed:
            self.load_index()
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call build_index() or load_index() first.")

        results = self.retriever.retrieve(query, top_k=20)

        # Filter by document type if specified
        if doc_type:
            results = [
                (metadata, score)
                for metadata, score in results
                if metadata.get("doc_type") == doc_type
            ]

        # Group by document
        doc_results = {}
        for metadata, score in results:
            doc_id = metadata["doc_id"]
            if doc_id not in doc_results:
                doc_results[doc_id] = {
                    "doc_id": doc_id,
                    "source_path": metadata["source_path"],
                    "max_score": score,
                    "chunk_count": 0,
                }
            doc_results[doc_id]["max_score"] = max(doc_results[doc_id]["max_score"], score)
            doc_results[doc_id]["chunk_count"] += 1

        # Sort by max score
        sorted_docs = sorted(doc_results.values(), key=lambda x: x["max_score"], reverse=True)

        return sorted_docs
