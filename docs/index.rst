.. Hybrid RAG documentation master file

Hybrid RAG ドキュメント
=======================

Dense（ベクトル）と Sparse（キーワード）を組み合わせたハイブリッド検索による
RAG（Retrieval-Augmented Generation）システムのドキュメントです。

概要
----

- **取り込み (ingestion)**: PDF / Markdown / HTML / テキストの読み込み
- **チャンキング (chunking)**: 見出し・コード・表を考慮したセマンティック分割
- **インデックス (indexing)**: Dense (FAISS) + Sparse (TF-IDF)
- **検索 (retrieval)**: RRF（Reciprocal Rank Fusion）によるハイブリッド検索
- **再ランキング (reranking)**: Cross-encoder による関連度の再計算
- **コンテキスト (context)**: トークン上限を考慮したプロンプト用コンテキスト構築
- **ストレージ (storage)**: SQLite によるメタデータ・チャンク保存
- **評価 (evaluation)**: Precision@k, Recall@k, MRR, NDCG などのメトリクス

クイックスタート
----------------

.. code-block:: python

   from hybrid_rag import HybridRAGSystem

   rag = HybridRAGSystem()
   rag.ingest_documents(["document.pdf"])
   rag.build_index()
   result = rag.query("質問文")
   print(result["context"])
   print(result["prompts"])

API リファレンス
----------------

.. toctree::
   :maxdepth: 2

   api

索引
----

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
