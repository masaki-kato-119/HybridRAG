.. Hybrid RAG documentation master file

Hybrid RAG ドキュメント
=======================

Dense（ベクトル）と Sparse（キーワード）を組み合わせたハイブリッド検索による
RAG（Retrieval-Augmented Generation）の実装です。

概要
----

- **ingestion**: PDF / Markdown / HTML / テキストの取り込み
- **chunking**: セマンティックチャンキング（見出し・コード・表を考慮）
- **indexing**: Dense (FAISS / Qdrant / ChromaDB / pgvector) + Sparse (TF-IDF)
- **retrieval**: RRF（Reciprocal Rank Fusion）によるハイブリッド検索
- **reranking**: Cross-encoder による再ランキング
- **context**: トークン上限を考慮したコンテキスト構築
- **storage**: SQLite によるメタデータ・チャンク保存
- **evaluation**: 検索評価メトリクス（Precision@k, Recall@k, MRR, NDCG）

クイックスタート
----------------

.. code-block:: python

   from hybrid_rag import create_rag_system

   # バックエンドを指定して RAG を生成（faiss / qdrant / chroma / postgres）
   rag = create_rag_system(backend="faiss")
   rag.ingest_documents(["document.pdf"])
   result = rag.query("質問文", top_k=5)

目次
----

.. toctree::
   :maxdepth: 2
   :caption: パッケージ概要

   modules/overview

.. toctree::
   :maxdepth: 2
   :caption: API リファレンス

   modules/rag_system
   modules/rag_system_factory
   modules/retrieval
   modules/reranking
   modules/indexing
   modules/chunking
   modules/ingestion
   modules/context
   modules/storage
   modules/evaluation
   modules/query_expansion
   modules/backends
   modules/indexing_qdrant
   modules/indexing_chroma
   modules/indexing_postgres

索引と検索
----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
