import os

from hybrid_rag import create_rag_system
from hybrid_rag.query_expansion import QueryExpander

# システムの初期化（クエリ拡張あり: LLM でキーワードを拡張して複数クエリで検索）
expander = QueryExpander(api_key=os.environ.get("OPENAI_API_KEY"), model="gpt-4o-mini")
rag = create_rag_system(
    backend="qdrant",
    query_expander=expander,
    #enable_mmr=True,      # MMRを有効化
    #mmr_lambda=0.6,       # λパラメータ（0.0〜1.0）    
)  # または "faiss"

# ドキュメントの投入（初回またはインデックスを作り直すときはコメントを外す）
# 注: Qdrant版に移行した場合、スパースインデックスも再構築が必要
rag.ingest_documents(["xxxx.pdf"])

# システムへのクエリ（クエリ拡張によりキーワードが広がり、複数クエリで RRF 検索）
result = rag.query("機能を全て上げて", top_k=80, rerank_top_k=120)

# RAG結果から OpenAI API で機能一覧レポートを生成
def generate_door_report_with_openai(
    prompts: dict, report_path: str = "機能一覧_レポート.md"
) -> str:
    """RAGのプロンプトをOpenAI APIに送り、機能一覧レポートを生成してファイルに保存する。"""
    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI API を使うには pip install openai を実行してください。")
        return ""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("環境変数 OPENAI_API_KEY を設定してください。")
        return ""

    system_report = """あなたは取扱説明書の内容を整理する技術ライターです。
与えられたコンテキスト（チャンク）のみに基づき、「機能一覧」をMarkdown形式のレポートとして作成してください。

ルール:
- コンテキストに書かれている関連の機能だけを列挙する
- 各項目では必要に応じて参照元のチャンク番号を記載する（例: Chunk 3 参照）
- 見出しで分類する
- コンテキストにない機能は書かない"""

    user_report = (
        prompts.get("user", "")
        + "\n\n---\n\n上記の Context と Question に基づき、機能一覧をMarkdownレポートとして作成してください。"
    )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": system_report},
            {"role": "user", "content": user_report},
        ],
        temperature=0.3,
    )
    report_text = response.choices[0].message.content

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"レポートを保存しました: {report_path}")
    return report_text

report_path = "機能一覧_レポート.md"
report = generate_door_report_with_openai(result["prompts"], report_path=report_path)
if report:
    print("\n--- レポート (先頭2000文字) ---\n")
    print(report[:2000] + ("..." if len(report) > 2000 else ""))

# Qdrantクライアントを明示的にクローズ（終了時のエラーを防ぐ）
rag.close()
