"""
Semantic Chunking Module
Handles intelligent text chunking based on semantic boundaries.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from .ingestion import Document


class ChunkType(Enum):
    """チャンクの種類。テキスト・コード・表を区別する。"""

    TEXT = "text"  # 通常のテキスト段落
    CODE = "code"  # コードブロック
    TABLE = "table"  # 表


@dataclass
class Chunk:
    """ドキュメントから切り出したチャンクとメタデータ。"""

    content: str
    """チャンクの本文。"""
    doc_id: str
    """所属ドキュメント ID。"""
    section: str
    """セクション名（見出しなど）。"""
    chunk_index: int
    """ドキュメント内のチャンク通し番号。"""
    chunk_type: ChunkType
    """チャンクの種類（TEXT / CODE / TABLE）。"""
    source_path: str
    """元ファイルのパス。"""
    metadata: Optional[Dict] = None
    """任意の追加メタデータ。"""


class SemanticChunker:
    """
    ドキュメント構造に基づくセマンティックチャンキング。

    見出し・段落を考慮し、コードブロックと表は別チャンクとして抽出する。
    最大文字数超過時は文境界または文字数で分割し、オーバーラップを付与する。
    """

    def __init__(self, max_chunk_size: int = 512, overlap_size: int = 50):
        """
        Args:
            max_chunk_size: 1チャンクの最大文字数。
            overlap_size: チャンク間のオーバーラップ（単語数または文字数）。
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

        # Code block patterns
        self.code_patterns = [
            r"```[\s\S]*?```",  # Markdown code blocks
            r"<code>[\s\S]*?</code>",  # HTML code blocks
            r"<pre>[\s\S]*?</pre>",  # HTML pre blocks
        ]

        # Table patterns
        self.table_patterns = [
            r"\|.*\|.*\n\|[-\s\|]*\|.*\n(\|.*\|.*\n)*",  # Markdown tables
            r"<table>[\s\S]*?</table>",  # HTML tables
        ]

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        ドキュメントをセマンティックチャンクに分割する。

        Args:
            document: 取り込み済み Document。

        Returns:
            チャンクのリスト。chunk_index は 0 から振り直される。
        """
        chunks = []

        # First, extract special content (code, tables)
        special_chunks = self._extract_special_content(document)
        chunks.extend(special_chunks)

        # Remove special content from main text
        cleaned_content = self._remove_special_content(document.content)

        # Chunk the remaining text by paragraphs and headings
        text_chunks = self._chunk_by_structure(cleaned_content, document)
        chunks.extend(text_chunks)

        # Re-index chunks
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        return chunks

    def _extract_special_content(self, document: Document) -> List[Chunk]:
        """
        コードブロックと表を別チャンクとして抽出する。max_chunk_size を超える場合は分割する。

        Args:
            document: 対象ドキュメント。

        Returns:
            コード・表チャンクのリスト。
        """
        chunks = []
        content = document.content

        # Extract code blocks
        for pattern in self.code_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
            for match in matches:
                code_content = match.group(0).strip()
                for part in self._split_by_size(code_content, ChunkType.CODE):
                    chunk = Chunk(
                        content=part,
                        doc_id=document.doc_id,
                        section="code_block",
                        chunk_index=0,
                        chunk_type=ChunkType.CODE,
                        source_path=document.source_path,
                    )
                    chunks.append(chunk)

        # Extract tables
        for pattern in self.table_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
            for match in matches:
                table_content = match.group(0).strip()
                for part in self._split_by_size(table_content, ChunkType.TABLE):
                    chunk = Chunk(
                        content=part,
                        doc_id=document.doc_id,
                        section="table",
                        chunk_index=0,
                        chunk_type=ChunkType.TABLE,
                        source_path=document.source_path,
                    )
                    chunks.append(chunk)

        return chunks

    def _split_by_size(self, text: str, chunk_type: ChunkType) -> List[str]:
        """
        テキストを max_chunk_size 以下に分割する。断片間にオーバーラップを付与する。

        Args:
            text: 分割対象のテキスト。
            chunk_type: チャンクの種類（未使用だがインターフェース統一用）。

        Returns:
            分割されたテキストのリスト。
        """
        if len(text) <= self.max_chunk_size:
            return [text] if text else []
        parts = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))
            parts.append(text[start:end])
            start = end - self.overlap_size if end < len(text) else len(text)
        return parts

    def _remove_special_content(self, content: str) -> str:
        """
        コンテンツからコードブロックと表を除去する。

        Args:
            content: 対象テキスト。

        Returns:
            除去後のテキスト。
        """
        cleaned = content

        # Remove code blocks
        for pattern in self.code_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE | re.DOTALL)

        # Remove tables
        for pattern in self.table_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE | re.DOTALL)

        # Clean up extra whitespace
        cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned)
        return cleaned.strip()

    def _chunk_by_structure(self, content: str, document: Document) -> List[Chunk]:
        """
        見出し・段落などの構造に基づいてテキストをチャンクに分割する。

        Args:
            content: チャンク化するテキスト。
            document: 元ドキュメント（doc_id, source_path 等に使用）。

        Returns:
            テキストチャンクのリスト。
        """
        chunks = []

        # Split by double newlines (paragraphs)
        paragraphs = content.split("\n\n")

        current_chunk = ""
        current_section = "main"

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Check if this is a heading
            if self._is_heading(paragraph):
                # If we have accumulated content, save it as a chunk
                if current_chunk.strip():
                    chunk = self._create_text_chunk(
                        current_chunk.strip(), document, current_section
                    )
                    chunks.append(chunk)
                    current_chunk = ""

                # Update section name
                current_section = paragraph[:50]  # Use first 50 chars as section name

            # Add paragraph to current chunk
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph

            # Check if chunk is getting too large
            if len(potential_chunk) > self.max_chunk_size:
                if current_chunk.strip():
                    # Save current chunk
                    chunk = self._create_text_chunk(
                        current_chunk.strip(), document, current_section
                    )
                    chunks.append(chunk)

                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + paragraph
                    # オーバーラップ+段落がまだ長すぎる場合は文/サイズで分割
                    if len(current_chunk) > self.max_chunk_size:
                        sub_chunks = self._split_large_paragraph(
                            current_chunk, document, current_section
                        )
                        chunks.extend(sub_chunks[:-1])
                        current_chunk = sub_chunks[-1].content if sub_chunks else ""
                else:
                    # Single paragraph is too large, split it
                    sub_chunks = self._split_large_paragraph(paragraph, document, current_section)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1].content if sub_chunks else ""
            else:
                current_chunk = potential_chunk

        # Add remaining content as final chunk
        if current_chunk.strip():
            chunk = self._create_text_chunk(current_chunk.strip(), document, current_section)
            chunks.append(chunk)

        return chunks

    def _is_heading(self, text: str) -> bool:
        """
        テキストが見出しかどうかを判定する。

        Args:
            text: 判定する文字列。

        Returns:
            見出しなら True、そうでなければ False。
        """
        heading_patterns = [
            r"^#{1,6}\s+",  # Markdown headings
            r"^\d+\.\s+",  # Numbered sections
            r"^[A-Z][^.!?]*$",  # All caps or title case without punctuation
        ]

        for pattern in heading_patterns:
            if re.match(pattern, text.strip()):
                return True

        return False

    def _create_text_chunk(self, content: str, document: Document, section: str) -> Chunk:
        """
        メタデータ付きのテキストチャンクを生成する。

        Args:
            content: チャンク本文。
            document: 元ドキュメント。
            section: セクション名。

        Returns:
            Chunk インスタンス。
        """
        return Chunk(
            content=content,
            doc_id=document.doc_id,
            section=section,
            chunk_index=0,  # Will be set later
            chunk_type=ChunkType.TEXT,
            source_path=document.source_path,
        )

    def _get_overlap_text(self, text: str) -> str:
        """
        現在チャンクの末尾からオーバーラップ用テキストを取得する。
        日本語など空白が少ない場合は文字数ベースのオーバーラップを使用する。

        Args:
            text: 対象テキスト。

        Returns:
            オーバーラップに使う末尾の文字列。
        """
        words = text.split()
        if len(words) > self.overlap_size:
            overlap_words = words[-self.overlap_size :]
            return " ".join(overlap_words) + " "
        # 日本語など空白が少ない場合: 末尾 overlap_size 文字（約50文字）を重複させる
        overlap_chars = min(self.overlap_size * 2, len(text))  # 約50〜100文字
        if len(text) <= overlap_chars:
            return text
        return text[-overlap_chars:]

    def _split_large_paragraph(
        self, paragraph: str, document: Document, section: str
    ) -> List[Chunk]:
        """
        長い段落を小さなチャンクに分割する。
        文境界（日本語の 。．！？ を含む）で分割。1 文が max_chunk_size を超える場合は文字数で分割。

        Args:
            paragraph: 分割する段落テキスト。
            document: 元ドキュメント。
            section: セクション名。

        Returns:
            チャンクのリスト。
        """
        chunks = []
        # 日本語の句点・疑問符・感嘆符も区切りに使う
        sentences = re.split(r"[.!?。．！？]+", paragraph)

        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sep = "。" if re.search(r"[\u3000-\u303f\u3040-\u9fff]", sentence) else ". "
            potential_chunk = (current_chunk + sep + sentence) if current_chunk else sentence

            if len(potential_chunk) > self.max_chunk_size and current_chunk:
                chunk = self._create_text_chunk(current_chunk.strip(), document, section)
                chunks.append(chunk)
                current_chunk = sentence
            else:
                current_chunk = potential_chunk

        if current_chunk.strip():
            # 1文が max_chunk_size を超える場合は文字数で分割（フォールバック）
            if len(current_chunk) <= self.max_chunk_size:
                chunks.append(self._create_text_chunk(current_chunk.strip(), document, section))
            else:
                for part in self._split_by_size(current_chunk, ChunkType.TEXT):
                    if part.strip():
                        chunks.append(self._create_text_chunk(part.strip(), document, section))
        return chunks
