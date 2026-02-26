# ハイブリッドRAGシステム

Dense（密）検索とSparse（疎）検索を組み合わせ、Reciprocal Rank Fusion（RRF）とCross-encoder再ランキングを実装した現代的なRetrieval-Augmented Generation（RAG）システムです。

## 言語対応（英語・日本語）

- **デフォルトモデル**: Dense / Reranker ともマルチ言語対応（50+言語）。英語・日本語のクエリとドキュメントの両方で利用可能です。
- **チャンク分割**: 日本語の句点（。．！？）を文境界として認識し、空白が少ない言語では文字数ベースのオーバーラップを使用します。
- **入出力**: 内部・ファイルとも UTF-8 を前提としています。

## 機能

- **多形式ドキュメント投入**: PDF、Markdown、HTML、テキストファイル対応（日本語PDFは PyMuPDF を優先使用）
- **セマンティックチャンク**: ドキュメント構造に基づく知的テキスト分割
- **ハイブリッド検索**: Dense（ベクトル）検索とSparse（BM25風）検索の組み合わせ
- **Reciprocal Rank Fusion**: 高度な結果統合戦略
- **Cross-encoder再ランキング**: 関連性スコアの改善
- **包括的ログ**: パフォーマンス追跡と評価指標
- **SQLiteストレージ**: 効率的なメタデータとチャンク管理
- **レポート生成**: RAG の検索結果を OpenAI API に渡して Markdown レポートを生成（`test.py` で例：機能一覧）

## アーキテクチャ

システムは仕様要件に従って構築されています：

1. **データ投入**: 正規化を含む多形式ドキュメント処理
2. **セマンティックチャンク**: 構造認識テキスト分割
3. **ハイブリッドインデックス**: FAISS（Dense）+ TF-IDF（Sparse）インデックス
4. **クエリ処理**: 正規化とキーワード抽出
5. **RRF検索**: ランク融合による並列Dense/Sparse検索
6. **再ランキング**: Cross-encoderベースの関連性スコアリング
7. **コンテキスト構築**: トークン認識コンテキスト構築
8. **評価・ログ**: 包括的パフォーマンス追跡

## インストール

```bash
# 依存関係のインストール
pip install -r requirements.txt

# test.py で OpenAI レポート生成を使う場合（要 OPENAI_API_KEY）
# openai は requirements.txt に含まれています

# 開発用
pip install -r requirements-dev.txt
```

## ドキュメント（Sphinx）

API リファレンスを Sphinx でビルドできます。

```bash
pip install sphinx sphinx-rtd-theme
python -m sphinx -b html docs/source docs/build
```

生成された HTML は ``docs/build/index.html`` を開いて閲覧できます。詳細は ``docs/README.md`` を参照してください。

## クイックスタート

### 推奨：ファクトリー関数で簡単切り替え

```python
from hybrid_rag import create_rag_system

# FAISS版を使用（デフォルト）
rag = create_rag_system(backend="faiss")

# または Qdrant版を使用（高速フィルタリング）
rag = create_rag_system(backend="qdrant")

# または ChromaDB版を使用（実験・比較用）
rag = create_rag_system(backend="chroma")

# または PostgreSQL (pgvector) 版を使用（要 Docker 等で Postgres 起動）
rag = create_rag_system(backend="postgres")

# 環境変数でも切り替え可能
import os
os.environ["RAG_BACKEND"] = "qdrant"
rag = create_rag_system()  # Qdrant版が使われる

# 使い方はどちらも同じ
rag.ingest_documents(["document1.pdf", "document2.md"])
result = rag.query("機械学習とは何ですか？", top_k=5)

print("コンテキスト:", result['context'])
print("プロンプト:", result['prompts'])
```

### 標準版（FAISS + SQLite）- 直接インポート

```python
from hybrid_rag.rag_system import HybridRAGSystem

# システムの初期化（デフォルトはマルチ言語モデル）
rag = HybridRAGSystem()

# ドキュメントの投入
rag.ingest_documents(["document1.pdf", "document2.md"])

# システムへのクエリ
result = rag.query("機械学習とは何ですか？", top_k=5)

print("コンテキスト:", result['context'])
print("プロンプト:", result['prompts'])
```

### Qdrant版（高速フィルタリング）- 直接インポート

**Qdrant版の利点**:
- フィルタ検索が50-70%高速（単一ステージフィルタリング）
- メモリ使用量が30-50%削減可能（量子化使用時）
- セットアップが簡単（`pip install qdrant-client` のみ）
- Dockerやサーバー不要（ローカルファイルベース）

```bash
# Qdrantクライアントのインストール
pip install qdrant-client
```

```python
from hybrid_rag.rag_system_qdrant import QdrantHybridRAGSystem

# Qdrant版システムの初期化（FAISS版と同じインターフェース）
rag = QdrantHybridRAGSystem(
    qdrant_path="./qdrant_data"  # ローカルファイルパス（サーバー不要）
)

# 使い方は標準版と同じ
rag.ingest_documents(["document1.pdf", "document2.md"])
result = rag.query("機械学習とは何ですか？", top_k=5)

# 使用後は明示的にクローズ（終了時のエラーを防ぐ）
rag.close()
```

**Qdrant版のサンプル実行**:
```bash
python example_qdrant.py
```

### ChromaDB版（実験・比較用）- ファクトリーで選択

**ChromaDB版の特徴**:
- Dense に ChromaDB（永続化・コサイン類似度）、Sparse は TF-IDF で他バックエンドと同一
- RRF + Cross-encoder 再ランキングは FAISS/Qdrant と共通
- 実験や他バックエンドとのパフォーマンス比較に利用可能

```bash
pip install chromadb
```

```python
from hybrid_rag import create_rag_system

rag = create_rag_system(backend="chroma", chroma_path="./chroma_data")
rag.ingest_documents(["document1.pdf", "document2.md"])
result = rag.query("機械学習とは何ですか？", top_k=5)
```

### PostgreSQL (pgvector) 版 - ファクトリーで選択

**PostgreSQL 版の特徴**:
- Dense に PostgreSQL + pgvector、Sparse は TF-IDF（他バックエンドと同一）
- RRF + Cross-encoder 再ランキングまで FAISS/Qdrant/Chroma と同じパイプライン
- 既存の PostgreSQL 環境や大規模データ向け

```bash
# 依存（Postgres は Docker 等で別途起動）
pip install psycopg[binary] pgvector

# 例: docker-compose で Postgres を起動
docker compose up -d
```

```python
from hybrid_rag import create_rag_system

# backend="postgres" で PostgresHybridRAGSystem を利用（接続先は indexing_postgres のデフォルト）
rag = create_rag_system(backend="postgres", index_path="./postgres_indices")
rag.ingest_documents(["document1.pdf", "document2.md"])
result = rag.query("機械学習とは何ですか？", top_k=5)
```

直接インポートする場合:
```python
from hybrid_rag.rag_system_postgres import PostgresHybridRAGSystem

rag = PostgresHybridRAGSystem(index_path="./postgres_indices")
rag.ingest_documents(["document1.pdf", "document2.md"])
result = rag.query("質問", top_k=5)
```

**バックエンドの比較・パフォーマンス測定**:
```bash
# FAISS と Qdrant を比較
python example_backend_comparison.py --backend both

# 全バックエンドを比較（FAISS / Qdrant / ChromaDB / PostgreSQL）
# 要: chromadb, psycopg, pgvector。Postgres は docker compose 等で起動しておく
python example_backend_comparison.py --backend all

# 単体で実行
python example_backend_comparison.py --backend faiss
python example_backend_comparison.py --backend qdrant
python example_backend_comparison.py --backend chroma
python example_backend_comparison.py --backend postgres
```

## 使用例

### test.py：RAG ＋ OpenAI でレポート生成

`test.py` は RAG で「機能を全て上げて」を検索し、取得したコンテキストを OpenAI API（gpt-5.2）に渡して **機能一覧** を Markdown レポートとして生成します。

```bash
# 環境変数 OPENAI_API_KEY を設定してから実行
pip install openai   # 未導入の場合
python test.py
```

- レポートは `機能一覧_レポート.md` に保存されます。
- `OPENAI_API_KEY` が未設定または `openai` が未インストールの場合はレポート生成をスキップし、RAG のコンテキスト・プロンプトのみ出力します。

### 高度な設定

```python
rag = HybridRAGSystem(
    db_path="custom_rag.db",
    index_path="custom_indices",
    dense_model="paraphrase-multilingual-MiniLM-L12-v2",  # デフォルト（マルチ言語）
    rerank_model="BAAI/bge-reranker-v2-m3",               # デフォルト（マルチ言語）
    max_chunk_size=1024,   # より大きなチャンク
    max_context_tokens=8000 # より多くのコンテキスト
)
# 英語のみで軽量にしたい場合: dense_model="all-MiniLM-L6-v2", rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
```

### メタデータフィルタリング

```python
# ドキュメントタイプでフィルタ
result = rag.query(
    "機械学習の概念",
    metadata_filters={"doc_type": "pdf"}
)

# ソースでフィルタ
result = rag.query(
    "ニューラルネットワーク", 
    metadata_filters={"source_path": "ml_textbook.pdf"}
)
```

### クエリ拡張（Query Expansion）

LLM（OpenAI 等）で元クエリを「検索用キーワード群」に変換し、複数クエリで検索して RRF で統合する。技術用語・同義語を 3〜5 個抽出する。Sparse 検索の用語不一致による空振りを減らし、検索精度を向上させる。

```python
import os
from hybrid_rag import HybridRAGSystem
from hybrid_rag.query_expansion import QueryExpander

# API キーを渡して QueryExpander を用意（省略時は拡張なし）
expander = QueryExpander(api_key=os.environ.get("OPENAI_API_KEY"), model="gpt-4o-mini")

rag = HybridRAGSystem(query_expander=expander)
# 既存どおり ingest / build_index したあと
result = rag.query("ドアが開かない", top_k=10)
```

- クライアントを渡す場合: `QueryExpander(client=openai.OpenAI(), model="gpt-4o-mini")`
- 拡張を使わない場合: `query_expander` は省略（デフォルト None）

## コード品質

リントとフォーマットの実行：

```bash
# コードフォーマット
black .
isort .

# スタイルチェック
flake8 .

# 型チェック
mypy hybrid_rag/
```

## Sphinx（APIドキュメント）

API ドキュメントを HTML でビルドする（`requirements-dev.txt` に sphinx を含む）：

```bash
# 開発用依存関係を入れている前提
pip install -r requirements-dev.txt

# プロジェクトルートからビルド（出力: docs/_build/html/）
sphinx-build -b html docs docs/_build
```

- ビルド後は `docs/_build/html/index.html` をブラウザで開くとドキュメントを閲覧できます。
- **テーマ**: Read the Docs 風の `sphinx_rtd_theme` を使用（左サイドバー＋本文）。`docs/conf.py` の `html_theme` で変更可能（例: `furo`、`pydata_sphinx_theme` など）。
- 見た目を変えたい場合は `docs/_static/custom.css` に CSS を追加すると反映されます。
- ソースは `docs/` 配下の `.rst`（ReStructuredText）。`docs/conf.py` で autodoc / Napoleon により `hybrid_rag` の docstring から API を生成しています。

## パフォーマンス

システムは以下に最適化されています：
- **高速検索**: 並列Dense/Sparse検索
- **メモリ効率**: チャンク処理とFAISSインデックス
- **スケーラビリティ**: 適切なインデックスを持つSQLiteストレージ
- **精度**: RRF + Cross-encoder再ランキング

現代的なハードウェアでの典型的なパフォーマンス：
- インデックス作成: 約1000チャンク/秒
- 検索: クエリあたり約50-100ms
- 再ランキング: クエリあたり約10-50ms（モデルに依存）

## 設定

### モデル（マルチ言語対応がデフォルト）

- **Dense埋め込み**: デフォルト `paraphrase-multilingual-MiniLM-L12-v2`（50+言語、日本語含む）
- **再ランキング**: デフォルト `BAAI/bge-reranker-v2-m3`（マルチ言語対応）

日本語・多言語ドキュメントでは上記のまま利用することを推奨します。英語のみで軽量にしたい場合の例：
- Dense: `all-MiniLM-L6-v2`
- 再ランキング: `cross-encoder/ms-marco-MiniLM-L-6-v2`

より高品質なマルチ言語 Dense の例: `paraphrase-multilingual-mpnet-base-v2`（768次元、やや重い）。  
**注意**: Dense モデルを変更した場合は、既存のインデックスでは次元が合わないため **再投入（ingest）＋インデックス再構築** が必要です。

### チャンクパラメータ

- `max_chunk_size`: チャンクあたりの最大文字数（デフォルト: 512）
- `overlap_size`: チャンク間のオーバーラップ（デフォルト: 50）

### 検索パラメータ

- `top_k`: 返す最終結果数（プロンプトに含めるチャンク数）
- `rerank_top_k`: 再ランキング前に取得する候補数（大きいほど関連チャンクが拾いやすい）
- RRFパラメータ `k`: ランク融合を制御（デフォルト: 60。小さいほど上位ランクを重視）
- `retrieval_candidates_multiplier`: Dense/Sparse それぞれが返す候補数 = rerank_top_k × この値（デフォルト: 2。4 などに増やすと候補が増え、関連度の高いチャンクが選ばれやすくなる）
- **メタデータフィルター** (`metadata_filters`): **出典条件のみ**（例: `doc_id`, `source_path`, `section`, `chunk_type`）。「質問と内容が関連しているか」のフィルターには使えません。そのためメタフィルターをかけてもヒット率は上がりません。
- **コンテンツキーワード** (`content_keywords`): チャンク**本文**に指定語のいずれかが含まれるものだけに絞ってから再ランキング。ヒット率・精度向上用。

## システム統計

包括的なシステム統計の取得：

```python
stats = rag.get_system_stats()
print(f"ドキュメント数: {stats['database']['total_documents']}")
print(f"チャンク数: {stats['database']['total_chunks']}")
print(f"平均クエリ時間: {stats['retrieval_24h']['avg_total_time']}ms")
```

## データの削除・初期化

### 特定ドキュメントの削除（全バックエンド共通）

1 件のドキュメントとそのチャンクだけを消したい場合は、`delete_document(doc_id)` を使います。削除後にインデックスを再構築するため、検索結果からも即反映されます。

```python
from hybrid_rag import create_rag_system

rag = create_rag_system(backend="qdrant")  # faiss / chroma / postgres でも同じ API

# doc_id を指定して削除（次回検索用にインデックスも再構築）
rag.delete_document("doc_001", rebuild_index=True)
```

`doc_id` が分からない場合は、`search_documents()` で確認できます。

```python
docs = rag.search_documents("キーワード")
for d in docs:
    print(d["doc_id"], d.get("source_path"))
```

### 全データを消して初期状態に戻す

バックエンドごとに、**メタデータ・チャンク用の DB** と **ベクトル／インデックス用の保存先** が異なります。すべて消す場合は、**RAG を利用しているプロセスを止めたうえで**、下表のファイル／ディレクトリを削除してください。

| バックエンド | 削除する対象 | 備考 |
|--------------|----------------|------|
| **faiss** | `db_path`（デフォルト: `hybrid_rag.db`）<br>＋ `index_path` 配下（デフォルト: `indices/`） | SQLite ファイル 1 つと、`indices/` 内の FAISS・TF-IDF 等のファイル |
| **qdrant** | `db_path`（デフォルト: `hybrid_rag.db`）<br>＋ `qdrant_path`（デフォルト: `./qdrant_data`） | SQLite と Qdrant のローカル保存ディレクトリ |
| **chroma** | `db_path`（デフォルト: `hybrid_rag.db`）<br>＋ `chroma_path`（デフォルト: `./chroma_data`） | SQLite と ChromaDB の保存ディレクトリ |
| **postgres** | `db_path`（デフォルト: `hybrid_rag.db`）<br>＋ `index_path` 配下（Sparse 等。デフォルト: `./postgres_indices`）<br>＋ **PostgreSQL 内の該当テーブル** | SQLite と Sparse 用ファイル。Dense は Postgres のテーブル（例: `rag_embeddings_pg`）を削除または TRUNCATE |

**例（PowerShell）:**

```powershell
# FAISS 版のデータをすべて消す
Remove-Item hybrid_rag.db -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force indices -ErrorAction SilentlyContinue

# Qdrant 版のデータをすべて消す
Remove-Item hybrid_rag.db -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force qdrant_data -ErrorAction SilentlyContinue

# Chroma 版のデータをすべて消す
Remove-Item hybrid_rag.db -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force chroma_data -ErrorAction SilentlyContinue
```

**例（Bash）:**

```bash
# FAISS 版
rm -f hybrid_rag.db && rm -rf indices

# Qdrant 版
rm -f hybrid_rag.db && rm -rf qdrant_data

# Chroma 版
rm -f hybrid_rag.db && rm -rf chroma_data
```

**PostgreSQL 版**で Dense のベクトルも消す場合は、Postgres に接続してテーブルを空にします（テーブル名は `indexing_postgres` の実装に依存します。例: `rag_embeddings_pg`）。

```sql
-- 例: テーブルを空にする
TRUNCATE TABLE rag_embeddings_pg;
```

その後、SQLite（`hybrid_rag.db`）と `postgres_indices/` を削除すれば、Postgres 版も初期状態に戻せます。

削除後は、再度 `ingest_documents()` と `build_index()`（または `load_index()` なしで ingest からやり直し）でデータを投入してください。

## ファイル構造

```
hybrid_rag/
├── __init__.py
├── ingestion.py      # ドキュメント処理
├── chunking.py       # セマンティックチャンク
├── indexing.py       # Dense & Sparseインデックス
├── retrieval.py      # RRF検索
├── reranking.py      # Cross-encoder再ランキング
├── context.py        # コンテキスト構築
├── evaluation.py     # ログ・評価
├── storage.py        # SQLiteデータベース
└── rag_system.py     # メインオーケストレータ
```

## 貢献

1. 既存のコードスタイル（Black + isort）に従う
2. 必要に応じてドキュメントを更新する

## ドキュメント改定

- **言語対応**: デフォルトをマルチ言語モデルに統一。英語・日本語のクエリ／ドキュメント両対応を明記。
- **test.py**: RAG 検索結果を OpenAI API で「ドアの機能一覧」Markdown レポートにまとめるフローを追加。
- **設定・検索パラメータ**: デフォルトモデル名、`top_k` / `rerank_top_k` の調整例を現行コードに合わせて更新。

## ライセンス

MIT License - 詳細はLICENSEファイルを参照してください。