"""
PostgreSQL(pgvector) 版ハイブリッド RAG システム。

FAISS / Qdrant / Chroma と同じインターフェースで、
Dense を Postgres(pgvector)、Sparse を TF-IDF とするバックエンド。
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .chunking import Chunk, SemanticChunker
from .context import ContextBuilder, PromptBuilder
from .evaluation import RAGLogger, RetrievalLog
from .indexing_postgres import PostgresHybridIndex
from .ingestion import DocumentProcessor, SectionDetector
from .query_expansion import QueryExpander
from .reranking import HybridReranker
from .retrieval import RRFRetriever, get_adaptive_candidate_multiplier
from .storage import DatabaseManager


class PostgresHybridRAGSystem:
    """
    Postgres(pgvector) を Dense ベクトルストアとして使用するハイブリッド RAG システム。

    - Dense: Postgres + pgvector
    - Sparse: TF-IDF
    - RRF + Cross-encoder 再ランクは他バックエンドと共通
    """

    def __init__(
        self,
        db_path: str = "hybrid_rag.db",
        index_path: str = "./postgres_indices",
        log_dir: str = "logs",
        dense_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
        max_chunk_size: int = 512,
        max_context_tokens: int = 4000,
        rrf_k: int = 60,
        retrieval_candidates_multiplier: int = 2,
        query_expander: Optional[QueryExpander] = None,
    ):
        self.db_manager = DatabaseManager(db_path)
        self.doc_processor = DocumentProcessor()
        self.section_detector = SectionDetector()
        self.chunker = SemanticChunker(max_chunk_size=max_chunk_size)

        self.hybrid_index = PostgresHybridIndex(
            dense_model_name=dense_model,
        )

        self.retriever: Optional[RRFRetriever] = None
        self.reranker = HybridReranker(cross_encoder_model=rerank_model)
        self.context_builder = ContextBuilder(max_tokens=max_context_tokens)
        self.prompt_builder = PromptBuilder()
        self.logger = RAGLogger(log_dir)
        self.rrf_k = rrf_k
        self.retrieval_candidates_multiplier = retrieval_candidates_multiplier
        self.query_expander = query_expander

        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.is_indexed = False

    def ingest_documents(
        self, file_paths: List[Union[str, Path]], rebuild_index: bool = True
    ) -> Dict[str, Any]:
        print("Starting document ingestion...")
        all_chunks: List[Chunk] = []
        processed_docs = 0
        failed_docs = 0

        for file_path in file_paths:
            try:
                print(f"Processing: {file_path}")
                document = self.doc_processor.process_file(file_path)
                self.db_manager.store_document(document)
                chunks = self.chunker.chunk_document(document)
                self.db_manager.store_chunks(chunks)
                all_chunks.extend(chunks)
                processed_docs += 1
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
                failed_docs += 1

        if rebuild_index and all_chunks:
            print("Building Postgres hybrid index...")
            self.build_index()

        stats = {
            "processed_documents": processed_docs,
            "failed_documents": failed_docs,
            "total_chunks": len(all_chunks),
            "index_rebuilt": rebuild_index and bool(all_chunks),
        }
        print(f"Ingestion complete: {stats}")
        return stats

    def build_index(self, batch_size: int = 1000) -> None:
        print("Loading chunks from database...")
        total_chunks = self.db_manager.get_chunk_count()
        if total_chunks == 0:
            raise ValueError("No chunks found in database. Ingest documents first.")

        print(f"Processing {total_chunks} chunks in batches of {batch_size}...")
        chunk_batches: List[List[Chunk]] = []
        offset = 0

        while offset < total_chunks:
            chunk_dicts = self.db_manager.get_chunks_batch(offset, batch_size)
            if not chunk_dicts:
                break

            from .chunking import ChunkType

            batch_chunks: List[Chunk] = []
            for d in chunk_dicts:
                batch_chunks.append(
                    Chunk(
                        content=d["content"],
                        doc_id=d["doc_id"],
                        section=d["section"],
                        chunk_index=d["chunk_index"],
                        chunk_type=ChunkType(d["chunk_type"]),
                        source_path=d.get("source_path", ""),
                        metadata=d.get("metadata"),
                    )
                )
            chunk_batches.append(batch_chunks)
            offset += batch_size

            if offset % (batch_size * 10) == 0 or offset >= total_chunks:
                print(f"Loaded {min(offset, total_chunks)}/{total_chunks} chunks...")

        self.hybrid_index.build_index_batch(chunk_batches, show_progress=True)
        self.hybrid_index.save(str(self.index_path))

        self.retriever = RRFRetriever(
            self.hybrid_index,
            k=self.rrf_k,
            retrieval_candidates_multiplier=self.retrieval_candidates_multiplier,
            enable_cache=True,
            cache_size=1000,
        )
        self.is_indexed = True
        print(f"Postgres index built successfully with {total_chunks} chunks!")

    def load_index(self) -> None:
        print("Loading existing Postgres hybrid index...")
        try:
            self.hybrid_index.load(str(self.index_path))
            self.retriever = RRFRetriever(
                self.hybrid_index,
                k=self.rrf_k,
                retrieval_candidates_multiplier=self.retrieval_candidates_multiplier,
                enable_cache=True,
                cache_size=1000,
            )
            self.is_indexed = True
            print("Postgres index loaded successfully!")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Postgres index not found: {e}. Build index first.")

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
    ) -> Dict[str, Any]:
        if not self.is_indexed:
            try:
                self.load_index()
            except FileNotFoundError:
                raise ValueError("No index found. Ingest documents and build index first.")

        if self.retriever is None:
            raise ValueError("Retriever not initialized.")

        start_time = time.time()

        # クエリ拡張
        if self.query_expander is not None:
            queries = self.query_expander.expand(query)
        else:
            queries = [query]

        if use_adaptive_candidates and retrieval_candidates_multiplier is None:
            retrieval_candidates_multiplier = get_adaptive_candidate_multiplier(query)

        # Retrieval
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

        # content が無い場合は一括取得（4.1.3）
        need_content = [(m["doc_id"], m["chunk_index"]) for m, _ in results if not m.get("content")]
        content_map = self.db_manager.get_contents_batch(need_content) if need_content else {}
        results_with_content = []
        for metadata, score in results:
            key = (metadata["doc_id"], metadata["chunk_index"])
            if key in content_map:
                metadata["content"] = content_map[key]
            results_with_content.append((metadata, score))
        results = results_with_content

        # content_keywords フィルタ
        if content_keywords:
            filtered = [
                (m, s)
                for m, s in results
                if (m.get("content") and any(kw in m["content"] for kw in content_keywords))
            ]
            if filtered:
                results = filtered

        retrieval_time = (time.time() - retrieval_start) * 1000

        # 再ランク
        rerank_start = time.time()
        reranked_results = self.reranker.rerank(query, results, top_k=top_k)
        rerank_time = (time.time() - rerank_start) * 1000

        # コンテキスト & プロンプト
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

        return {
            "query": query,
            "context": context,
            "prompts": prompts,
            "results": reranked_results,
            "stats": {
                "retrieval_time_ms": retrieval_time,
                "rerank_time_ms": rerank_time,
                "total_time_ms": total_time,
                "results_count": len(reranked_results),
            },
        }
