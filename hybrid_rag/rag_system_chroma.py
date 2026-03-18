"""
ChromaDB版ハイブリッド RAG システム。

FAISS+SQLite / Qdrant と互換性のあるインターフェースを提供。
ChromaDB で Dense ベクトルを永続化し、Sparse は TF-IDF で同一の RRF + 再ランキングパイプラインを使用。

使用方法:
    from hybrid_rag.rag_system_chroma import ChromaHybridRAGSystem

    rag = ChromaHybridRAGSystem(chroma_path="./chroma_data")
    rag.ingest_documents(["document.pdf"])
    result = rag.query("質問", top_k=5)
    rag.close()
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .caching import QueryCache
from .chunking import Chunk, SemanticChunker
from .context import ContextBuilder, PromptBuilder
from .evaluation import RAGLogger, RetrievalLog
from .indexing_chroma import ChromaHybridIndex
from .ingestion import DocumentProcessor, SectionDetector
from .query_expansion import QueryExpander
from .reranking import HybridReranker
from .retrieval import RRFRetriever, get_adaptive_candidate_multiplier
from .storage import DatabaseManager


class ChromaHybridRAGSystem:
    """
    ChromaDB を使用したハイブリッド RAG システム。

    - Dense: ChromaDB（永続化・コサイン類似度）
    - Sparse: TF-IDF（FAISS/Qdrant と同一）
    - RRF + Cross-encoder 再ランキングは共通
    """

    def __init__(
        self,
        db_path: str = "hybrid_rag.db",
        chroma_path: str = "./chroma_data",
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
    ):
        self.db_manager = DatabaseManager(db_path)
        self.doc_processor = DocumentProcessor()
        self.section_detector = SectionDetector()
        self.chunker = SemanticChunker(max_chunk_size=max_chunk_size)

        self.hybrid_index = ChromaHybridIndex(
            dense_model_name=dense_model,
            chroma_path=chroma_path,
        )

        self.retriever: Optional[RRFRetriever] = None
        self.reranker = HybridReranker(cross_encoder_model=rerank_model)
        self.context_builder = ContextBuilder(max_tokens=max_context_tokens)
        self.prompt_builder = PromptBuilder()
        self.logger = RAGLogger(log_dir)
        self.rrf_k = rrf_k
        self.retrieval_candidates_multiplier = retrieval_candidates_multiplier
        self.query_expander = query_expander

        self.enable_cache = enable_cache
        self.query_cache = (
            QueryCache(cache_size=cache_size, ttl_seconds=cache_ttl_seconds)
            if enable_cache
            else None
        )

        self.chroma_path = Path(chroma_path)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.is_indexed = False

    def __del__(self):
        self.close()

    def close(self):
        """クリーンアップ。ChromaDB PersistentClient は特にクローズ不要だが互換性のため。"""
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
            print("Building Chroma hybrid index...")
            self.build_index()

        stats = {
            "processed_documents": processed_docs,
            "skipped_documents": skipped_docs,
            "failed_documents": failed_docs,
            "total_chunks": len(all_chunks),
            "index_rebuilt": rebuild_index and bool(all_chunks or processed_docs > 0),
        }
        print(f"Ingestion complete: {stats}")
        return stats

    def build_index(self, batch_size: int = 1000) -> None:
        """DB 内の全チャンクから Chroma + Sparse インデックスを構築する。"""
        print("Loading chunks from database...")
        total_chunks = self.db_manager.get_chunk_count()
        if total_chunks == 0:
            raise ValueError("No chunks found in database. Ingest documents first.")

        print(f"Processing {total_chunks} chunks in batches of {batch_size}...")
        chunk_batches = []
        offset = 0

        while offset < total_chunks:
            chunk_dicts = self.db_manager.get_chunks_batch(offset, batch_size)
            if not chunk_dicts:
                break
            from .chunking import ChunkType

            batch_chunks = [
                Chunk(
                    content=d["content"],
                    doc_id=d["doc_id"],
                    section=d["section"],
                    chunk_index=d["chunk_index"],
                    chunk_type=ChunkType(d["chunk_type"]),
                    source_path=d.get("source_path", ""),
                    metadata=d.get("metadata"),
                )
                for d in chunk_dicts
            ]
            chunk_batches.append(batch_chunks)
            offset += batch_size
            if offset % (batch_size * 10) == 0 or offset >= total_chunks:
                print(f"Loaded {min(offset, total_chunks)}/{total_chunks} chunks...")

        self.hybrid_index.build_index_batch(chunk_batches, show_progress=True)
        self.hybrid_index.save(str(self.chroma_path))

        self.retriever = RRFRetriever(
            self.hybrid_index,
            k=self.rrf_k,
            retrieval_candidates_multiplier=self.retrieval_candidates_multiplier,
            enable_cache=True,
            cache_size=1000,
        )
        self.is_indexed = True
        print(f"Chroma index built successfully with {total_chunks} chunks!")

    def load_index(self) -> None:
        """既存の Chroma + Sparse インデックスを読み込む。"""
        print("Loading existing Chroma index...")
        try:
            self.hybrid_index.load(str(self.chroma_path))
            self.retriever = RRFRetriever(
                self.hybrid_index,
                k=self.rrf_k,
                retrieval_candidates_multiplier=self.retrieval_candidates_multiplier,
                enable_cache=True,
                cache_size=1000,
            )
            self.is_indexed = True
            print("Chroma index loaded successfully!")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Chroma index not found: {e}. Build index first.")

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
        """RAG クエリを実行する。"""
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
                import copy
                result_copy = copy.deepcopy(cached_result)
                result_copy["stats"]["from_cache"] = True
                return result_copy

        start_time = time.time()
        if self.retriever is None:
            raise ValueError("Retriever not initialized.")

        if self.query_expander is not None:
            queries = self.query_expander.expand(query)
        else:
            queries = [query]

        if use_adaptive_candidates and retrieval_candidates_multiplier is None:
            retrieval_candidates_multiplier = get_adaptive_candidate_multiplier(query)

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

        if content_keywords:
            filtered = [
                (m, s)
                for m, s in results
                if (m.get("content") and any(kw in m["content"] for kw in content_keywords))
            ]
            if filtered:
                results = filtered

        retrieval_time = (time.time() - retrieval_start) * 1000

        rerank_start = time.time()
        reranked_results = self.reranker.rerank(query, results, top_k=top_k)
        rerank_time = (time.time() - rerank_start) * 1000

        context = self.context_builder.build_context(
            reranked_results, include_metadata=include_metadata, include_scores=include_scores
        )
        prompts = self.prompt_builder.build_prompt_with_citations(query, context)
        total_time = (time.time() - start_time) * 1000

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
            rrf_scores=[s for _, s in results],
            rerank_scores=[s for _, s in reranked_results],
            metadata_filters=metadata_filters,
        )
        self.logger.log_retrieval(log_entry)
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
                "total_time_ms": total_time,
                "results_count": len(reranked_results),
                "from_cache": False,
            },
        }

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

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """クエリキャッシュの統計情報を取得する。"""
        if self.query_cache is None:
            return None
        return self.query_cache.get_stats()

    def clear_cache(self) -> None:
        """クエリキャッシュをクリアする。"""
        if self.query_cache is not None:
            self.query_cache.clear()
        else:
            print("Query cache is not enabled.")

    def get_system_stats(self) -> Dict[str, Any]:
        """システム統計を返す。"""
        db_stats = self.db_manager.get_database_stats()
        retrieval_stats = self.db_manager.get_retrieval_stats(24)
        return {
            "database": db_stats,
            "retrieval_24h": retrieval_stats,
            "index_status": self.is_indexed,
            "backend": "chroma",
        }
