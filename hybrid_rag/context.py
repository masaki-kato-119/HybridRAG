"""
コンテキスト構築モジュール。

トークン上限と重複除去を考慮したコンテキストの組み立てを行う。
"""

import re
from typing import Dict, List, Optional, Tuple


class ContextBuilder:
    """検索結果チャンクからトークン管理付きでコンテキストを構築する。"""

    def __init__(self, max_tokens: int = 4000, tokens_per_char: float = 0.25):
        """
        コンテキストビルダーを初期化する。

        Args:
            max_tokens: コンテキストに許容する最大トークン数。
            tokens_per_char: 文字あたりの概算トークン数（逆数が 1 文字あたりトークン数）。
        """
        self.max_tokens = max_tokens
        self.tokens_per_char = tokens_per_char
        self.max_chars = int(max_tokens / tokens_per_char)

    def build_context(
        self,
        results: List[Tuple[Dict, float]],
        include_metadata: bool = True,
        include_scores: bool = False,
    ) -> str:
        """
        検索結果からコンテキスト文字列を構築する。

        Args:
            results: (メタデータ, スコア) のタプルのリスト。
            include_metadata: 最小限のメタデータを含めるかどうか。
            include_scores: 関連度スコアを含めるかどうか。

        Returns:
            整形されたコンテキスト文字列。
        """
        if not results:
            return ""

        # Remove duplicates while preserving order
        unique_results = self._remove_duplicates(results)

        # Build context within token limits
        context_parts = []
        current_length = 0

        for i, (metadata, score) in enumerate(unique_results, 1):
            chunk_text = metadata["content"].strip()

            # Format chunk
            chunk_formatted = self._format_chunk(
                chunk_text, i, metadata, score, include_metadata, include_scores
            )

            # Check if adding this chunk would exceed limits
            chunk_length = len(chunk_formatted)
            if current_length + chunk_length > self.max_chars:
                # Try to fit a truncated version
                remaining_chars = self.max_chars - current_length - 100  # Buffer
                if remaining_chars > 200:  # Minimum useful chunk size
                    truncated_text = chunk_text[:remaining_chars] + "..."
                    chunk_formatted = self._format_chunk(
                        truncated_text, i, metadata, score, include_metadata, include_scores
                    )
                    context_parts.append(chunk_formatted)
                break

            context_parts.append(chunk_formatted)
            current_length += chunk_length

        return "\n\n".join(context_parts)

    def _remove_duplicates(self, results: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        """
        内容の重複するチャンクを除去する。

        Args:
            results: (メタデータ, スコア) のタプルのリスト。

        Returns:
            重複を除いた (メタデータ, スコア) のリスト。順序は維持する。
        """
        unique_results = []
        seen_contents = set()

        for metadata, score in results:
            content = metadata["content"]

            # Simple deduplication based on exact content match
            content_normalized = re.sub(r"\s+", " ", content.strip().lower())

            if content_normalized not in seen_contents:
                seen_contents.add(content_normalized)
                unique_results.append((metadata, score))

        return unique_results

    def _format_chunk(
        self,
        content: str,
        index: int,
        metadata: Dict,
        score: float,
        include_metadata: bool,
        include_scores: bool,
    ) -> str:
        """
        1 チャンクをコンテキスト用に整形する。

        Args:
            content: チャンクの本文。
            index: チャンク番号（1 始まり）。
            metadata: チャンクのメタデータ辞書。
            score: 関連度スコア。
            include_metadata: メタデータをヘッダに含めるかどうか。
            include_scores: スコアをヘッダに含めるかどうか。

        Returns:
            整形されたチャンク文字列（ヘッダ + 本文）。
        """
        formatted_parts = []

        # Add chunk header
        header = f"[Chunk {index}]"

        # Add minimal metadata if requested
        if include_metadata:
            meta_parts = []
            if "source_path" in metadata:
                source = metadata["source_path"].split("/")[-1]  # Just filename
                meta_parts.append(f"Source: {source}")
            if "section" in metadata and metadata["section"] != "main":
                section = metadata["section"][:30]  # Truncate long section names
                meta_parts.append(f"Section: {section}")

            if meta_parts:
                header += f" ({', '.join(meta_parts)})"

        # Add score if requested
        if include_scores:
            header += f" [Score: {score:.3f}]"

        formatted_parts.append(header)
        formatted_parts.append(content)

        return "\n".join(formatted_parts)

    def get_context_stats(self, results: List[Tuple[Dict, float]]) -> Dict:
        """
        コンテキスト構築に関する統計を返す。

        Args:
            results: (メタデータ, スコア) のタプルのリスト。

        Returns:
            統計の辞書（total_chunks, unique_chunks, total_chars, estimated_tokens など）。
        """
        if not results:
            return {"total_chunks": 0, "unique_chunks": 0, "total_chars": 0, "estimated_tokens": 0}

        unique_results = self._remove_duplicates(results)
        context = self.build_context(unique_results, include_metadata=False)

        stats = {
            "total_chunks": len(results),
            "unique_chunks": len(unique_results),
            "duplicates_removed": len(results) - len(unique_results),
            "total_chars": len(context),
            "estimated_tokens": int(len(context) * self.tokens_per_char),
            "within_token_limit": int(len(context) * self.tokens_per_char) <= self.max_tokens,
            "chunks_included": len([chunk for chunk in context.split("[Chunk ") if chunk.strip()])
            - 1,
        }

        return stats


class PromptBuilder:
    """検索コンテキストを用いた LLM 用プロンプトを組み立てる。"""

    def __init__(self) -> None:
        self.system_prompt_template = (
            "You are a helpful AI assistant that answers questions "
            "based on the provided context. \n\n"
            "Guidelines:\n"
            "- Use the retrieved context as your primary source of information\n"
            "- If the context doesn't contain enough information to answer "
            "the question, say so clearly\n"
            "- Provide specific references to the source material when possible\n"
            "- Keep your response focused and relevant to the question\n"
            "- Do not make up information that isn't in the context"
        )

    def build_prompt(
        self, query: str, context: str, system_prompt: Optional[str] = None
    ) -> Dict[str, str]:
        """
        LLM 用のプロンプトを組み立てる。

        Args:
            query: ユーザーの質問文。
            context: 検索して整形したコンテキスト文字列。
            system_prompt: 任意のシステムプロンプト。省略時はデフォルトを使用。

        Returns:
            'system' と 'user' キーを持つプロンプト辞書。
        """
        if system_prompt is None:
            system_prompt = self.system_prompt_template

        user_prompt = f"""Context:
{context}

Question: {query}

Please answer the question based on the provided context."""

        return {"system": system_prompt, "user": user_prompt}

    def build_prompt_with_citations(
        self, query: str, context: str, system_prompt: Optional[str] = None
    ) -> Dict[str, str]:
        """
        出典引用を促すプロンプトを組み立てる。

        Args:
            query: ユーザーの質問文。
            context: 検索して整形したコンテキスト文字列。
            system_prompt: 任意のシステムプロンプト。省略時は引用用デフォルトを使用。

        Returns:
            'system' と 'user' キーを持つプロンプト辞書。
        """
        citation_system_prompt = (
            "You are a helpful AI assistant that answers questions "
            "based on the provided context.\n\n"
            "Guidelines:\n"
            "- Use the retrieved context as your primary source of information\n"
            "- Always cite your sources using the chunk numbers "
            '(e.g., "According to Chunk 1...")\n'
            "- If the context doesn't contain enough information, say so clearly\n"
            "- Provide specific references to the source material\n"
            "- Keep your response focused and relevant to the question\n"
            "- Do not make up information that isn't in the context"
        )

        if system_prompt is None:
            system_prompt = citation_system_prompt

        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Please answer the question based on the provided context. "
            "Make sure to cite the specific chunks you're referencing in your answer."
        )

        return {"system": system_prompt, "user": user_prompt}
