# Configuration file for the Sphinx documentation builder.
#
# Hybrid RAG - Sphinx ドキュメント設定

import os
import sys

# プロジェクトルートをパスに追加（hybrid_rag を import するため）
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
project = "Hybrid RAG"
copyright = "Hybrid RAG Project"
author = "Hybrid RAG"
release = "1.0.0"
version = "1.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
]

templates_path = ["_templates"]
exclude_patterns = []
language = "ja"

# Napoleon: Google / NumPy スタイルの docstring を解釈
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# オプション依存のモジュールをドキュメントビルド時に mock（psycopg 未導入環境でもビルド可能）
autodoc_mock_imports = ["psycopg", "pgvector"]

# autodoc
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "Hybrid RAG ドキュメント"
html_short_title = "Hybrid RAG"
