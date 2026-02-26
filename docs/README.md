# Hybrid RAG ドキュメント（Sphinx）

## ビルド方法

```bash
# 依存（Sphinx とテーマ）
pip install sphinx sphinx-rtd-theme

# HTML を生成（プロジェクトルートで実行）
python -m sphinx -b html docs/source docs/build
```

生成された HTML は `docs/build/index.html` をブラウザで開いて閲覧できます。

## クリーンビルド

```bash
# キャッシュ削除してから再ビルド
rm -rf docs/build
python -m sphinx -b html docs/source docs/build
```

## ソース構成

- `source/conf.py` … Sphinx 設定（autodoc, Napoleon, テーマ）
- `source/index.rst` … トップページと toctree
- `source/modules/*.rst` … 各モジュールの API リファレンス（autodoc 使用）
