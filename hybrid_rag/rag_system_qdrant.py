"""
Qdrant版ハイブリッド RAG システム。

FAISS+SQLite版と互換性のあるインターフェースを提供しつつ、
Qdrantの高速フィルタリングを活用。

使用方法:
    from hybrid_rag.rag_system_qdrant import QdrantHybridRAGSystem

    # FAISS版と同じインターフェース
    rag = QdrantHybridRAGSystem()
    rag.ingest_documents(["document.pdf"])
    result = rag.query("質問", top_k=5)
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .caching import QueryCache
from .chunking import Chunk, SemanticChunker
from .context import ContextBuilder, PromptBuilder
from .evaluation import RAGLogger, RetrievalLog
from .indexing_qdrant import QdrantHybridIndex
from .ingestion import DocumentProcessor, SectionDetector
from .query_expansion import QueryExpander
from .reranking import HybridReranker
from .retrieval import RRFRetriever, get_adaptive_candidate_multiplier
from .storage import DatabaseManager


class QdrantHybridRAGSystem:
    """
    Qdrantを使用したハイブリッド RAG システム。

    FAISS+SQLite版と比較した利点:
    - フィルタ検索が50-70%高速
    - メモリ使用量が30-50%削減可能
    - セットアップが簡単（pip install qdrant-client のみ）
    - Dockerやサーバー不要
    """

    def __init__(
        self,
        db_path: str = "hybrid_rag.db",
        qdrant_path: str = "./qdrant_data",
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
        enable_async: bool = True,
    ):
        """
        Qdrant版ハイブリッド RAG システムを初期化する。

        Args:
            db_path: SQLite データベースのパス（メタデータ管理用）。
            qdrant_path: Qdrantのデータ保存先（ローカルファイルパス）。
            log_dir: ログを出力するディレクトリ。
            dense_model: Dense 埋め込み用の Sentence Transformers モデル名。
            rerank_model: 再ランキング用の Cross-encoder モデル名。
            max_chunk_size: チャンクあたりの最大文字数。
            max_context_tokens: コンテキストに含める最大トークン数。
            rrf_k: RRF 定数。
            retrieval_candidates_multiplier: Dense/Sparse それぞれが返す候補数の倍率。
            query_expander: クエリ拡張用の QueryExpander インスタンス。
            enable_cache: クエリ結果キャッシングを有効にするかどうか。デフォルト True。
            cache_size: キャッシュに保持する最大エントリ数。デフォルト 1000。
            cache_ttl_seconds: キャッシュエントリの有効期限（秒）。Noneの場合は無期限。デフォルト 3600（1時間）。
            enable_async: Dense/Sparse検索を非同期並列実行するかどうか。デフォルト True。
        """
        # Initialize components
        self.db_manager = DatabaseManager(db_path)
        self.doc_processor = DocumentProcessor()
        self.section_detector = SectionDetector()
        self.chunker = SemanticChunker(max_chunk_size=max_chunk_size)

        # Qdrant版のハイブリッドインデックス
        self.hybrid_index = QdrantHybridIndex(
            dense_model_name=dense_model,
            qdrant_path=qdrant_path,
        )

        self.retriever: Optional[RRFRetriever] = None
        self.reranker = HybridReranker(cross_encoder_model=rerank_model)
        self.context_builder = ContextBuilder(max_tokens=max_context_tokens)
        self.prompt_builder = PromptBuilder()
        self.logger = RAGLogger(log_dir)
        self.rrf_k = rrf_k
        self.retrieval_candidates_multiplier = retrieval_candidates_multiplier
        self.query_expander = query_expander
        self.enable_async = enable_async

        # Query cache
        self.enable_cache = enable_cache
        self.query_cache = (
            QueryCache(cache_size=cache_size, ttl_seconds=cache_ttl_seconds)
            if enable_cache
            else None
        )

        # Paths
        self.qdrant_path = Path(qdrant_path)
        self.qdrant_path.mkdir(exist_ok=True)

        # State
        self.is_indexed = False

    def __del__(self):
        """デストラクタ: Qdrantクライアントを明示的にクローズしてクリーンアップエラーを防ぐ。"""
        self.close()

    def close(self):
        """
        Qdrantクライアントを明示的にクローズする。

        プログラム終了時のクリーンアップエラーを防ぐため、
        使用後に明示的に呼び出すことを推奨。
        """
        try:
            if hasattr(self, "hybrid_index") and hasattr(self.hybrid_index, "dense_index"):
                if hasattr(self.hybrid_index.dense_index, "client"):
                    # Pythonシャットダウン時のImportErrorを回避
                    import sys

                    if sys.meta_path is not None:
                        self.hybrid_index.dense_index.client.close()
        except Exception:
            # シャットダウン時のエラーは無視
            pass

    def ingest_documents(
        self,
        file_paths: List[Union[str, Path]],
        rebuild_index: bool = True,
        metadata: Optional[Union[Dict, List[Optional[Dict]]]] = None,
        skip_unchanged: bool = True,
    ) -> Dict[str, Any]:
        """
        ドキュメントをシステムに投入する。

        差分更新に対応しており、ファイル内容が変わっていないドキュメントはスキップする。
        任意のメタデータをドキュメント・チャンクに付与できる。

        Args:
            file_paths: 投入するファイルパスのリスト。
            rebuild_index: 投入後にインデックスを再構築するかどうか。
            metadata: 各ドキュメントに付与する追加メタデータ。
                - Dict を渡すと全ドキュメントに同じメタデータを付与。
                - List[Optional[Dict]] を渡すと file_paths と 1 対 1 で対応。
            skip_unchanged: True のとき、内容が変わっていないドキュメントをスキップする。
                デフォルト True。

        Returns:
            投入結果の辞書。
        """
        import hashlib as _hashlib

        print("Starting document ingestion...")

        all_chunks = []
        processed_docs = 0
        skipped_docs = 0
        failed_docs = 0

        for i, file_path in enumerate(file_paths):
            try:
                file_path = Path(file_path)
                print(f"Processing: {file_path}")

                if isinstance(metadata, list):
                    doc_meta = metadata[i] if i < len(metadata) else None
                elif isinstance(metadata, dict):
                    doc_meta = metadata
                else:
                    doc_meta = None

                document = self.doc_processor.process_file(file_path)
                if doc_meta:
                    document.metadata = {**(document.metadata or {}), **doc_meta}

                if skip_unchanged:
                    existing = self.db_manager.get_document(document.doc_id)
                    current_hash = _hashlib.md5(document.content.encode("utf-8")).hexdigest()
                    if existing and existing.get("content_hash") == current_hash:
                        print(f"  Skipped (unchanged): {file_path}")
                        skipped_docs += 1
                        continue

                self.db_manager.store_document(document)
                chunks = self.chunker.chunk_document(document)
                if doc_meta:
                    for chunk in chunks:
                        chunk.metadata = {**(chunk.metadata or {}), **doc_meta}
                self.db_manager.store_chunks(chunks)
                all_chunks.extend(chunks)
                processed_docs += 1

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
                failed_docs += 1

        if rebuild_index and (all_chunks or processed_docs > 0):
            print("Building Qdrant hybrid index...")
            self.build_index()

        stats = {
            "processed_documents": processed_docs,
            "skipped_documents": skipped_docs,
            "failed_documents": failed_docs,
            "total_chunks": len(all_chunks),
            "index_rebuilt": rebuild_index and bool(all_chunks or processed_docs > 0),
        }

        try:
            print(f"Ingestion complete: {stats}")
        except UnicodeEncodeError:
            print(
                f"Ingestion complete: processed={processed_docs}, "
                f"skipped={skipped_docs}, failed={failed_docs}, chunks={len(all_chunks)}"
            )
        return stats

    def build_index(self, batch_size: int = 1000) -> None:
        """
        DB 内の全チャンクからバッチ処理でQdrantインデックスを構築する。

        Args:
            batch_size: 1 バッチあたりのチャンク数。
        """
        print("Loading chunks from database...")

        total_chunks = self.db_manager.get_chunk_count()

        if total_chunks == 0:
            raise ValueError("No chunks found in database. Ingest documents first.")

        print(f"Processing {total_chunks} chunks in batches of {batch_size}...")

        # Process chunks in batches
        chunk_batches = []
        offset = 0

        while offset < total_chunks:
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

            if offset % (batch_size * 10) == 0 or offset >= total_chunks:
                print(f"Loaded {min(offset, total_chunks)}/{total_chunks} chunks...")

        # Build Qdrant hybrid index
        self.hybrid_index.build_index_batch(chunk_batches, show_progress=True)

        # Save the index (especially sparse index)
        self.hybrid_index.save(self.qdrant_path)

        # Initialize retriever（キャッシュ・非同期有効化）
        self.retriever = RRFRetriever(
            self.hybrid_index,
            k=self.rrf_k,
            retrieval_candidates_multiplier=self.retrieval_candidates_multiplier,
            enable_cache=True,
            cache_size=1000,
            enable_async=self.enable_async,
        )

        self.is_indexed = True
        print(f"Qdrant index built successfully with {total_chunks} chunks!")

    def load_index(self) -> None:
        """
        既存のQdrantインデックスを読み込む。

        Qdrantはファイルベースなので、自動的に読み込まれる。
        """
        print("Loading existing Qdrant index...")

        try:
            self.hybrid_index.load(str(self.qdrant_path))
            self.retriever = RRFRetriever(
                self.hybrid_index,
                k=self.rrf_k,
                retrieval_candidates_multiplier=self.retrieval_candidates_multiplier,
                enable_cache=True,
                cache_size=1000,
                enable_async=self.enable_async,
            )
            self.is_indexed = True
            print("Qdrant index loaded successfully!")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Qdrant index not found: {e}. Build index first.")

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

        Qdrantの単一ステージフィルタリングにより、
        metadata_filters使用時にFAISS版より50-70%高速。

        Args:
            query: ユーザーのクエリ文字列。
            top_k: 再ランキング後に返す最終結果数。
            rerank_top_k: 再ランキング前に取得する候補数。
            include_metadata: コンテキストにメタデータを含めるかどうか。
            include_scores: コンテキストに関連度スコアを含めるかどうか。
            metadata_filters: メタデータフィルタ（Qdrantで高速処理）。
            retrieval_candidates_multiplier: 検索時の候補倍率の上書き。None で
                use_adaptive_candidates 時はクエリ長に応じて自動。
            content_keywords: チャンク本文のキーワードフィルタ。
            use_adaptive_candidates: True のとき、クエリ長に応じて候補倍率を動的調整。
            use_cache: キャッシュを使用するかどうか。デフォルト True。

        Returns:
            クエリ結果の辞書。
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
            raise ValueError("Retriever not initialized.")

        # クエリ拡張
        if self.query_expander is not None:
            queries = self.query_expander.expand(query)
        else:
            queries = [query]

        if use_adaptive_candidates and retrieval_candidates_multiplier is None:
            retrieval_candidates_multiplier = get_adaptive_candidate_multiplier(query)

        # Retrieval phase（Qdrantの高速フィルタリングを活用）
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
            )

        # content が欠けている場合のみ一括取得（4.1.3）
        need_content = [(m["doc_id"], m["chunk_index"]) for m, _ in results if not m.get("content")]
        content_map = self.db_manager.get_contents_batch(need_content) if need_content else {}
        results_with_content = []
        for metadata, score in results:
            key = (metadata["doc_id"], metadata["chunk_index"])
            if key in content_map:
                metadata["content"] = content_map[key]
            results_with_content.append((metadata, score))
        results = results_with_content

        # Content keyword filter
        if content_keywords:
            filtered = [
                (m, s)
                for m, s in results
                if (m.get("content") and any(kw in m["content"] for kw in content_keywords))
            ]
            if filtered:
                results = filtered

        retrieval_time = (time.time() - retrieval_start) * 1000

        # Re-ranking phase
        rerank_start = time.time()
        reranked_results = self.reranker.rerank(query, results, top_k=top_k)
        rerank_time = (time.time() - rerank_start) * 1000

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
            sparse_results_count=len(results),
            rrf_results_count=len(results),
            rerank_results_count=len(reranked_results),
            final_results_count=len(reranked_results),
            retrieval_time_ms=retrieval_time,
            rerank_time_ms=rerank_time,
            total_time_ms=total_time,
            dense_scores=[],
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
            metadata={"filters": metadata_filters, "backend": "qdrant"},
        )

        result = {
            "query": query,
            "context": context,
            "prompts": prompts,
            "results": reranked_results,
            "stats": {
                "retrieval_time_ms": retrieval_time,
                "rerank_time_ms": rerank_time,
                "total_time_ms": total_time,
                "results_count": len(reranked_results),
                "backend": "qdrant",
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
        """システム全体の統計を取得する。"""
        db_stats = self.db_manager.get_database_stats()
        retrieval_stats = self.db_manager.get_retrieval_stats(24)

        stats = {
            "database": db_stats,
            "retrieval_24h": retrieval_stats,
            "index_status": self.is_indexed,
            "backend": "qdrant",
            "qdrant_path": str(self.qdrant_path),
            "components": {
                "chunker_max_size": self.chunker.max_chunk_size,
                "context_max_tokens": self.context_builder.max_tokens,
                "dense_model": "paraphrase-multilingual-MiniLM-L12-v2",
                "rerank_model": "BAAI/bge-reranker-v2-m3",
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
            rebuild_index: 削除後にインデックスを再構築するかどうか。
        """
        self.db_manager.delete_document(doc_id)

        if rebuild_index:
            print(f"Document {doc_id} deleted. Rebuilding Qdrant index...")
            self.build_index()

    def search_documents(self, query: str, doc_type: Optional[str] = None) -> List[Dict]:
        """
        コンテンツまたはメタデータでドキュメントを検索する。

        Args:
            query: 検索クエリ文字列。
            doc_type: 絞り込むドキュメントタイプ。

        Returns:
            ドキュメント単位の辞書のリスト。
        """
        if not self.is_indexed:
            self.load_index()
        if self.retriever is None:
            raise ValueError("Retriever not initialized.")

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
