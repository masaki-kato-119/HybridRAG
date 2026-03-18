# ハイブリッドRAGシステム

Dense（密）検索とSparse（疎）検索を組み合わせ、Reciprocal Rank Fusion（RRF）とCross-encoder再ランキングを実装した現代的なRetrieval-Augmented Generation（RAG）システムです。

## 言語対応（英語・日本語）

- **デフォルトモデル**: Dense / Reranker ともマルチ言語対応（50+言語）。英語・日本語のクエリとドキュメントの両方で利用可能です。
- **チャンク分割**: 日本語の句点（。．！？）を文境界として認識し、空白が少ない言語では文字数ベースのオーバーラップを使用します。
- **入出力**: 内部・ファイルとも UTF-8 を前提としています。

## 機能

- **多形式ドキュメント投入**: PDF、Markdown、HTML、テキストファイル対応（日本語PDFは PyMuPDF を優先使用）
- **差分更新**: ファイル内容が変わっていないドキュメントをスキップし、変更分だけ再処理
- **カスタムメタデータ**: 任意のキーをドキュメント・チャンクに付与可能
- **セマンティックチャンク**: ドキュメント構造に基づく知的テキスト分割
- **ハイブリッド検索**: Dense（ベクトル）検索とSparse（BM25）検索の組み合わせ
- **Reciprocal Rank Fusion**: 高度な結果統合戦略
- **Cross-encoder再ランキング**: 関連性スコアの改善
- **MMR多様性選択**: 重複・冗長なチャンクを削減し、コンテキストの情報量を向上
- **クエリ結果キャッシング**: LRU戦略による高速クエリ応答（80-95%高速化）
- **埋め込みキャッシング**: クエリ埋め込みの再利用（90-99%高速化）
- **クエリ拡張**: LLM（OpenAI 等）によるキーワード拡張と複数クエリ RRF 統合
- **包括的ログ**: パフォーマンス追跡と評価指標
- **SQLiteストレージ**: 効率的なメタデータとチャンク管理

## アーキテクチャ

1. **データ投入**: 正規化を含む多形式ドキュメント処理（差分更新対応）
2. **セマンティックチャンク**: 構造認識テキスト分割
3. **ハイブリッドインデックス**: FAISS（Dense）+ BM25（Sparse）インデックス
4. **クエリ処理**: 正規化・キーワード抽出・LLMクエリ拡張
5. **RRF検索**: ランク融合による並列Dense/Sparse検索
6. **再ランキング**: Cross-encoderベースの関連性スコアリング
7. **コンテキスト構築**: トークン認識コンテキスト構築
8. **評価・ログ**: 包括的パフォーマンス追跡

## インストール

```bash
# コアのみ（FAISS バックエンド）
pip install hybrid-rag

# バックエンドを追加する場合
pip install hybrid-rag[qdrant]    # Qdrant
pip install hybrid-rag[chroma]    # ChromaDB
pip install hybrid-rag[postgres]  # PostgreSQL (pgvector)
pip install hybrid-rag[all]       # 全バックエンド

# 開発用
pip install hybrid-rag[dev]
```

ソースから直接インストールする場合:
```bash
pip install -r requirements.txt
```

## ドキュメント（Sphinx）

```bash
pip install sphinx sphinx-rtd-theme
python -m sphinx -b html docs/source docs/build
```

生成された HTML は `docs/build/index.html` を開いて閲覧できます。

## クイックスタート

```python
from hybrid_rag import create_rag_system

# バックエンドを指定（"faiss" / "qdrant" / "chroma" / "postgres"）
rag = create_rag_system(backend="faiss")

# ドキュメントの投入
rag.ingest_documents(["document1.pdf", "document2.md"])

# クエリ
result = rag.query("機械学習とは何ですか？", top_k=5)
print(result['context'])
print(result['prompts'])
```

環境変数でバックエンドを切り替えることもできます:
```python
import os
os.environ["RAG_BACKEND"] = "qdrant"
rag = create_rag_system()
```

## バックエンド

| 項目 | FAISS | Qdrant | ChromaDB | PostgreSQL |
|------|-------|--------|----------|------------|
| フィルタ検索 | 後処理 | 高速（単一ステージ） | 後処理 | 後処理 |
| サーバー | 不要 | 不要 | 不要 | 必要 |
| スケール | 〜2000万 | 〜5000万 | 中規模 | 大規模 |
| 追加インストール | なし | `qdrant-client` | `chromadb` | `psycopg[binary] pgvector` |

```python
# Qdrant版（使用後は close() を呼ぶ）
rag = create_rag_system(backend="qdrant", qdrant_path="./qdrant_data")
rag.ingest_documents(["doc.pdf"])
result = rag.query("質問", top_k=5)
rag.close()

# ChromaDB版
rag = create_rag_system(backend="chroma", chroma_path="./chroma_data")

# PostgreSQL版（要 Docker 等で Postgres 起動）
rag = create_rag_system(backend="postgres", index_path="./postgres_indices")
```

## 差分更新

`ingest_documents` はデフォルトで差分更新を行います。ファイルの内容（MD5ハッシュ）が前回と同じであればスキップされます。

```python
# 初回投入
rag.ingest_documents(["doc1.pdf", "doc2.pdf"])
# → processed=2, skipped=0

# 再実行（doc1.pdf は変更なし、doc2.pdf は更新済み）
rag.ingest_documents(["doc1.pdf", "doc2.pdf"])
# → processed=1, skipped=1

# 差分更新を無効にして強制再処理
rag.ingest_documents(["doc1.pdf"], skip_unchanged=False)
```

## カスタムメタデータ

任意のキーをドキュメントとチャンクに付与できます。付与したメタデータは `metadata_filters` での絞り込みに使えます。

```python
# 全ドキュメントに共通のメタデータを付与
rag.ingest_documents(
    ["manual_v1.pdf", "manual_v2.pdf"],
    metadata={"category": "manual", "lang": "ja"}
)

# ファイルごとに異なるメタデータを付与
rag.ingest_documents(
    ["chapter1.pdf", "chapter2.pdf"],
    metadata=[
        {"chapter": 1, "topic": "overview"},
        {"chapter": 2, "topic": "installation"},
    ]
)

# 付与したメタデータでフィルタリング
result = rag.query("インストール方法", metadata_filters={"topic": "installation"})
```

## クエリ拡張

LLM でクエリをキーワード群に展開し、複数クエリで検索して RRF で統合します。Sparse 検索の用語不一致による空振りを減らします。

```python
import os
from hybrid_rag import create_rag_system
from hybrid_rag.query_expansion import QueryExpander

expander = QueryExpander(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    # ドメイン固有のプロンプトに差し替え可能
    system_prompt="あなたは医療文書の検索アシスタントです。..."
)

rag = create_rag_system(backend="qdrant", query_expander=expander)
result = rag.query("副作用について", top_k=10)
```

## MMR多様性選択

再ランキング後に MMR を適用し、似た内容のチャンクが top_k に固まるのを防ぎます。

```python
rag = create_rag_system(
    backend="faiss",
    enable_mmr=True,
    mmr_lambda=0.6,  # 1.0: 関連度重視 / 0.0: 多様性重視
)
```

## キャッシング

```python
rag = create_rag_system(
    enable_cache=True,           # クエリ結果キャッシュ（デフォルト: True）
    cache_size=1000,
    cache_ttl_seconds=3600,
    enable_embedding_cache=True, # 埋め込みキャッシュ（デフォルト: True）
    embedding_cache_size=10000,
)

result = rag.query("質問", top_k=5)
print(result['stats']['from_cache'])  # True/False

stats = rag.get_cache_stats()
print(f"ヒット率: {stats['hit_rate']:.1%}")

rag.clear_cache()
```

## 高度な設定

```python
rag = create_rag_system(
    backend="faiss",
    # モデル
    dense_model="paraphrase-multilingual-MiniLM-L12-v2",
    rerank_model="BAAI/bge-reranker-v2-m3",
    # チャンク
    max_chunk_size=512,
    max_context_tokens=4000,
    # FAISSインデックス（大規模データ向け）
    index_type="hnsw",   # "flat" / "ivf" / "hnsw"
    hnsw_m=32,
    # Sparse
    sparse_index_type="bm25",  # "bm25" / "tfidf"
    # 検索
    rrf_k=60,
    retrieval_candidates_multiplier=2,
    rerank_batch_size=32,
    enable_early_filtering=True,
    enable_async=True,
)
```

**FAISSインデックスタイプの目安**:
- `flat`: 〜10万チャンク（デフォルト、最高精度）
- `hnsw`: 10万〜100万チャンク（高速・高精度）
- `ivf`: 100万チャンク以上（超大規模向け）

## 検索パラメータ

```python
result = rag.query(
    "質問",
    top_k=5,              # 最終返却チャンク数
    rerank_top_k=10,      # 再ランキング前の候補数
    metadata_filters={"doc_id": "manual_001"},  # 出典フィルタ
    content_keywords=["キーワード1", "キーワード2"],  # 本文キーワードフィルタ
    use_adaptive_candidates=True,  # クエリ長に応じて候補数を自動調整
    use_cache=True,
)
```

**`metadata_filters` は出典条件用**（`doc_id`, `source_path`, `section`, `chunk_type`, カスタムメタデータキー）です。内容の関連度フィルタには `content_keywords` を使ってください。

## ドキュメント管理

```python
# 特定ドキュメントの削除
rag.delete_document("doc_001", rebuild_index=True)

# ドキュメント検索
docs = rag.search_documents("キーワード")
for d in docs:
    print(d["doc_id"], d["source_path"])

# システム統計
stats = rag.get_system_stats()
print(f"ドキュメント数: {stats['database']['total_documents']}")
print(f"チャンク数: {stats['database']['total_chunks']}")
```

## 全データの削除

| バックエンド | 削除対象 |
|-------------|---------|
| faiss | `hybrid_rag.db` + `indices/` |
| qdrant | `hybrid_rag.db` + `qdrant_data/` |
| chroma | `hybrid_rag.db` + `chroma_data/` |
| postgres | `hybrid_rag.db` + `postgres_indices/` + Postgres テーブル |

```bash
# FAISS版
rm -f hybrid_rag.db && rm -rf indices

# Qdrant版
rm -f hybrid_rag.db && rm -rf qdrant_data
```

## 評価

```python
from hybrid_rag.evaluation import RAGEvaluator

evaluator = RAGEvaluator()

# 検索精度の評価
metrics = evaluator.evaluate_retrieval(
    queries_and_ground_truth=[
        {"query": "機械学習とは", "relevant_docs": ["doc_001", "doc_002"]},
    ],
    retrieval_function=lambda q, k: rag.query(q, top_k=k)["results"],
    k_values=[1, 3, 5, 10],
)
print(f"MRR: {metrics.mrr:.3f}")
print(f"NDCG@5: {metrics.ndcg_at_k[5]:.3f}")

# エンドツーエンド評価（レスポンスタイム計測）
results = evaluator.evaluate_end_to_end(
    test_cases=[{"query": "質問1"}, {"query": "質問2"}],
    rag_system=rag,
)
print(f"平均レスポンス: {results['avg_response_time_ms']:.1f}ms")
```

## コード品質

```bash
black .
isort .
flake8 .
mypy hybrid_rag/
```

## ファイル構造

```
hybrid_rag/
├── __init__.py
├── ingestion.py           # ドキュメント処理
├── chunking.py            # セマンティックチャンク
├── indexing.py            # Dense & Sparse インデックス（FAISS）
├── indexing_bm25.py       # BM25 インデックス
├── indexing_qdrant.py     # Qdrant インデックス
├── indexing_chroma.py     # ChromaDB インデックス
├── indexing_postgres.py   # PostgreSQL インデックス
├── retrieval.py           # RRF 検索
├── reranking.py           # Cross-encoder 再ランキング
├── caching.py             # クエリ結果キャッシング
├── embedding_cache.py     # 埋め込みキャッシング
├── context.py             # コンテキスト構築
├── diversity.py           # MMR 多様性選択
├── evaluation.py          # ログ・評価
├── storage.py             # SQLite データベース
├── query_expansion.py     # クエリ拡張（LLM）
├── rag_system.py          # メインオーケストレータ（FAISS）
├── rag_system_qdrant.py   # Qdrant 版
├── rag_system_chroma.py   # ChromaDB 版
├── rag_system_postgres.py # PostgreSQL 版
└── rag_system_factory.py  # バックエンド切り替えファクトリー
```

## ライセンス

MIT License
