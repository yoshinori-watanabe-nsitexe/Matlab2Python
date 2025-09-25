# ⚠️ Heuristic translator:

- 8割方のよくある文法・関数を雑に置換します（if/for/while/function/switch、演算子、コメントなど）。
- 1-based→0-basedの添字・行列/要素演算の違いは完全には解決しません。必要に応じて手で修正してください。
- MATLAB の * は行列積、.* は要素積です。本スクリプトはデフォルトで * → @ に変換します（--no-matmul で抑止）。
- 複雑なインデクシングや cell / struct、クラス、GUI は非対応 or 簡易対応です。

## 使い方:
    python matlab2python.py input.m -o output.py
    # 文字列として変換
    python -c "import matlab2python, sys; print(matlab2python.convert_str(open('in.m').read())[0])"

## 問題点
- 'を全部.Tに変換してしまう
- コメントが消えてしまうことが多い
- matlab特有の関数への対応は不十分
- matplotlib関係の変換が甘い
 
ので

機能制限版  matlab2python_simple.py

matlab関数のpython実装utils.py
