"""
Data Ingestion Module
Handles PDF, Markdown, HTML, and Text file processing.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import markdown
import PyPDF2
from bs4 import BeautifulSoup


@dataclass
class Document:
    """
    取り込み済みドキュメントの表現。

    Attributes:
        content: 抽出されたテキスト全文。
        doc_id: 一意のドキュメント ID。
        source_path: 元ファイルのパス。
        doc_type: 拡張子に基づくタイプ（pdf, md, html, txt）。
        metadata: 任意の追加メタデータ。
    """

    content: str
    doc_id: str
    source_path: str
    doc_type: str
    metadata: Optional[Dict] = None


class DocumentProcessor:
    """
    PDF / Markdown / HTML / テキストファイルを処理する。

    対応形式: .pdf, .md, .html, .txt。
    PDF は PyMuPDF があればそれを使用し、なければ PyPDF2 にフォールバックする。
    """

    def __init__(self) -> None:
        self.supported_formats = {".pdf", ".md", ".html", ".txt"}

    def process_file(self, file_path: Union[str, Path]) -> Document:
        """
        単一ファイルを処理して Document を返す。

        Args:
            file_path: ファイルパス（str または Path）。

        Returns:
            正規化済みテキストとメタデータを持つ Document。

        Raises:
            ValueError: 未対応の拡張子の場合。
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        doc_id = self._generate_doc_id(file_path)

        if file_path.suffix.lower() == ".pdf":
            content = self._process_pdf(file_path)
        elif file_path.suffix.lower() == ".md":
            content = self._process_markdown(file_path)
        elif file_path.suffix.lower() == ".html":
            content = self._process_html(file_path)
        else:  # .txt
            content = self._process_text(file_path)

        # Apply normalization
        content = self._normalize_text(content)

        return Document(
            content=content,
            doc_id=doc_id,
            source_path=str(file_path),
            doc_type=file_path.suffix.lower()[1:],  # Remove the dot
        )

    def _generate_doc_id(self, file_path: Path) -> str:
        """
        ファイルパスから一意のドキュメント ID を生成する。

        Args:
            file_path: 対象ファイルのパス。

        Returns:
            一意のドキュメント ID 文字列。
        """
        return f"{file_path.stem}_{hash(str(file_path)) % 10000:04d}"

    def _process_pdf(self, file_path: Path) -> str:
        """
        PDF ファイルからテキストを抽出する。

        PyMuPDF (pymupdf) があれば優先使用（日本語・CJK に有利）。なければ PyPDF2 にフォールバック。
        PyPDF2 は日本語が文字化けしやすい。

        Args:
            file_path: PDF ファイルのパス。

        Returns:
            抽出したテキスト文字列。
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            fitz = None
        if fitz is not None:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
                text += "\n"
        return text

    def _process_markdown(self, file_path: Path) -> str:
        """
        Markdown ファイルを処理してプレーンテキストを取得する。

        Args:
            file_path: Markdown ファイルのパス。

        Returns:
            抽出したテキスト文字列。
        """
        with open(file_path, "r", encoding="utf-8") as file:
            md_content = file.read()

        # Convert to HTML first to extract plain text
        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text()

    def _process_html(self, file_path: Path) -> str:
        """
        HTML ファイルからテキストを抽出する。

        Args:
            file_path: HTML ファイルのパス。

        Returns:
            抽出したテキスト文字列。
        """
        with open(file_path, "r", encoding="utf-8") as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text()

    def _process_text(self, file_path: Path) -> str:
        """
        プレーンテキストファイルを読み込む。

        Args:
            file_path: テキストファイルのパス。

        Returns:
            ファイル内容の文字列。
        """
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def _normalize_text(self, text: str) -> str:
        """
        空白と改行を整理してテキストを正規化する。

        Args:
            text: 対象テキスト。

        Returns:
            正規化後のテキスト。
        """
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove excessive line breaks
        text = re.sub(r"\n\s*\n", "\n\n", text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text


class SectionDetector:
    """
    見出しパターンに基づいてセクション境界を検出する。

    Markdown の # 見出し、アンダーライン見出し、番号付き見出しを検出する。
    """

    def __init__(self) -> None:
        # Patterns for different heading styles
        self.heading_patterns = [
            r"^#{1,6}\s+(.+)$",  # Markdown headings
            r"^(.+)\n[=-]+$",  # Underlined headings
            r"^\d+\.\s+(.+)$",  # Numbered sections
        ]

    def detect_sections(self, text: str) -> List[Dict]:
        """
        テキストからセクション境界を検出する。

        Args:
            text: 解析対象のテキスト。

        Returns:
            各要素が title, start_line, end_line, content を持つ辞書のリスト。
        """
        lines = text.split("\n")
        sections = []
        current_section = {"title": "Introduction", "start_line": 0, "content": ""}

        for i, line in enumerate(lines):
            is_heading = False
            heading_text: Optional[str] = None

            for pattern in self.heading_patterns:
                match = re.match(pattern, line.strip(), re.MULTILINE)
                if match:
                    is_heading = True
                    heading_text = str(match.group(1)).strip()
                    break

            if is_heading and heading_text:
                # Save previous section
                if str(current_section["content"]).strip():
                    current_section["end_line"] = i - 1
                    sections.append(current_section.copy())

                # Start new section
                current_section = {"title": heading_text, "start_line": i, "content": ""}
            else:
                current_section["content"] = str(current_section["content"]) + line + "\n"

        # Add the last section
        if str(current_section["content"]).strip():
            current_section["end_line"] = len(lines) - 1
            sections.append(current_section)

        return sections
