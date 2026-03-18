"""
クエリ拡張モジュール。

LLM（OpenAI 等）を用いて元クエリを検索用キーワード群に変換し、
複数クエリで検索して RRF で統合するための拡張クエリを生成する。
"""

import re
from typing import Any, List, Optional

DEFAULT_SYSTEM_PROMPT = """あなたは検索アシスタントです。
ユーザーの質問から、ドキュメントの索引や本文で使われていそうな「技術用語」や「同義語」を3〜5個抽出してください。
出力形式: カンマ区切りのキーワードのみ。説明や改行は不要です。"""


class QueryExpander:
    """
    LLM を用いて元クエリから検索用キーワードを抽出し、拡張クエリリストを返す。

    デフォルトでは技術用語・同義語を 3〜5 個抽出する汎用プロンプトを使用する。
    system_prompt を渡すことでドメイン固有の指示に差し替え可能。
    クライアント未設定や API エラー時は元クエリのみを返す。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        max_keywords: int = 5,
        system_prompt: Optional[str] = None,
    ) -> None:
        """
        クエリ拡張器を初期化する。

        Args:
            api_key: OpenAI API キー。省略時は環境変数 OPENAI_API_KEY を参照しない
                （client を渡す場合は不要）。
            client: OpenAI 互換のクライアント（openai.OpenAI() のインスタンス等）。
                省略時は api_key からクライアントを生成する。
            model: 使用するモデル名。低コストの gpt-4o-mini を推奨。
            max_keywords: 抽出するキーワードの最大個数。3〜5 が適正。
            system_prompt: キーワード抽出に使うシステムプロンプト。
                省略時は DEFAULT_SYSTEM_PROMPT を使用。ドメイン固有の指示に差し替え可能。
        """
        self._client: Optional[Any] = None
        if client is not None:
            self._client = client
        elif api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=api_key)
            except ImportError:
                self._client = None
        else:
            # api_key も client も渡されていない場合はクライアントを作らない。
            # 環境変数 OPENAI_API_KEY を使う場合は呼び出し側で api_key=os.environ.get("OPENAI_API_KEY") を渡す。
            self._client = None

        self.model = model
        self.max_keywords = max(1, min(max_keywords, 10))
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT

    def expand(self, original_query: str) -> List[str]:
        """
        元クエリを LLM で拡張し、[元クエリ, キーワード1, キーワード2, ...] を返す。

        Args:
            original_query: ユーザーの元のクエリ文字列。

        Returns:
            元クエリを先頭にした検索用クエリのリスト。拡張不可時は [original_query] のみ。
        """
        if not original_query or not original_query.strip():
            return [original_query]

        if self._client is None:
            return [original_query]

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": original_query.strip()},
                ],
                max_tokens=150,
                temperature=0.3,
            )
        except Exception:
            return [original_query]

        content = None
        if response.choices:
            content = response.choices[0].message.content
        if not content or not content.strip():
            return [original_query]

        # カンマ区切りでパースし、空白除去・空でないものを最大 max_keywords 個まで
        keywords = [w.strip() for w in re.split(r"[,，、\n]+", content.strip()) if w.strip()][
            : self.max_keywords
        ]

        # 元クエリを先頭にしたリストを返す（重複は残す。検索で RRF がまとめる）
        return [original_query] + keywords
