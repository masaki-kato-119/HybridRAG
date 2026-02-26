.. _overview:

パッケージ概要
==============

**hybrid_rag** は、密ベクトル検索（Dense）とスパースキーワード検索（Sparse）を
RRF（Reciprocal Rank Fusion）で統合し、Cross-encoder で再ランキングする RAG システムです。

バックエンド
------------

- **faiss**: FAISS + SQLite（標準版）
- **qdrant**: Qdrant ローカルストレージ（高速フィルタリング）
- **chroma**: ChromaDB（永続化ベクトル DB）
- **postgres**: PostgreSQL + pgvector

共通インターフェース
--------------------

いずれのバックエンドも次の API で利用できます。

- :meth:`ingest_documents` … ドキュメント投入
- :meth:`build_index` / :meth:`load_index` … インデックス構築・読み込み
- :meth:`query` … 検索＋再ランク＋コンテキスト生成
- :meth:`get_system_stats` … 統計取得

インデックス実装はバックエンドごとに次のモジュールがあります。

- :doc:`indexing` … FAISS（標準版）
- :doc:`indexing_qdrant` … Qdrant 用
- :doc:`indexing_chroma` … ChromaDB 用
- :doc:`indexing_postgres` … PostgreSQL (pgvector) 用

詳細は :doc:`rag_system` および :doc:`rag_system_factory` を参照してください。
