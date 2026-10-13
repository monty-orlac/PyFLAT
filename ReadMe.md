# Python で学ぶ有限要素法 サンプルコード

[Zenn にて公開中の「Python で学ぶ有限要素法」](https://zenn.dev/monty/articles/adcc984d5233c3) で紹介している有限要素法のサンプルコードです。Step.1 ～ Step.7 に分けて、教育用の2次元有限要素法コードを少しずつ整理・機能拡張しています。

各記事には下記からアクセスすることもできます。

- [第1回 トラス解析とアセンブリの仕組み](https://zenn.dev/monty/articles/f2d053cfdcc2da)
- [第2回 オブジェクト指向によるソルバーと要素の整理](https://zenn.dev/monty/articles/db117a60d8ec55)
- [第3回 疎行列ソルバーの追加](https://zenn.dev/monty/articles/7bcaae30307bca)
- [第4回 要素実装の抽象化](https://zenn.dev/monty/articles/0f98da7a19adce)
- [第5回 平面ひずみ要素の追加](https://zenn.dev/monty/articles/03c4f84b915b0f)
- [第6回 ファイル分割とクラス階層の整理](https://zenn.dev/monty/articles/45c1417bd1d3da)
- [第7回 変位境界条件と要素辺荷重の追加](https://zenn.dev/monty/articles/012e3b0a31b500)
- [第8回 今後の発展性について](https://zenn.dev/monty/articles/da847e275a252d)

本プログラムは MIT ライセンスのもとに公開されています。詳しくは [LICENSE](LICENSE) ファイル、もしくは[こちら](https://opensource.org/licenses/mit-license.php)をご参照ください。

## 動作環境

- Python 3.12 以上
- NumPy
- SciPy

ライブラリは pip でインストールできます。

```bash
pip install numpy scipy
```

## 使い方

### 1. プログラムのダウンロード
#### Git を使う場合

下記コマンドを実行してリポジトリを取得します。

```bash
$ git clone https://github.com/monty-orlac/PyFLAT.git
$ cd PyFLAT
```

#### Git を使わない場合
1. 本ページ上部にある緑色の **「<> Code」** ボタンをクリックします。
2. 表示されたメニューから **「Download ZIP」** を選びます。
3. ダウンロードした ZIP ファイルを好きな場所に展開します。
   展開すると `PyFLAT-main` というフォルダができます。
4. ターミナル (Windows ではコマンドプロンプトや PowerShell) を開き、展開したフォルダに移動します。

```bash
$ cd <展開したフォルダのパス>/PyFLAT-main
```

### 2. プログラムの実行

実行したいステップの pyflat.py を実行してください。例えば Step.1 のプログラムは下記の通り実行できます。

```bash
$ python Step01/pyflat.py
```

環境によっては python の代わりに python3 (Windows では py) で実行してください。解析モデル (節点や要素、荷重境界条件などの定義) は各ステップの pyflat.py の中に直接書かれています。

## フォルダ構成

```
.
├── Step01/  ...  第1回記事に対応したプログラム
├── Step02/  ...  第2回記事に対応したプログラム
├── Step03/  ...  第3回記事に対応したプログラム
├── Step04/  ...  第4回記事に対応したプログラム
├── Step05/  ...  第5回記事に対応したプログラム
├── Step06/  ...  第6回記事に対応したプログラム
└── Step07/  ...  第7回～第8回記事に対応したプログラム
```

各ステップのフォルダは独立しており、それぞれ単体で実行できます。

## 各ステップの内容

| ステップ | 内容 | 解析例 |
|---|---|---|
| Step01 | 2次元トラスの最小構成の FEM プログラム | 2次元トラス |
| Step02 | 連立一次方程式の解法 (行列ソルバー) を別モジュールに分離 | 2次元トラス |
| Step03 | 疎行列を用いた直接法ソルバーを追加 | 2次元トラス |
| Step04 | 自由度番号の自動割り当てと、要素の共通化 | 2次元トラス |
| Step05 | 平面ひずみ要素 (三角形1次/2次要素、四角形1次要素) を追加 | 平面ひずみ問題 |
| Step06 | 要素・材料・荷重条件をモジュールに整理 | 平面ひずみ問題 |
| Step07 | 分布荷重・強制変位と、複数の荷重ケースの解析に対応 | 平面ひずみ問題 |

## 実行例

`Step01` を実行すると、節点変位と要素応力が表示されます。

```
$ python Step01/pyflat.py
Nodal displacement result:
     nid          x-disp          y-disp
       1    0.000000e+00    0.000000e+00
       2    3.306878e-03   -5.537468e-03
       3    1.653439e-03   -1.004154e-02
       4    5.516528e-19   -5.537468e-03
       5    3.306878e-03    0.000000e+00

Element stress result:
     eid           str11
       1    -3.275688e+08
       2     1.736111e+08
       ...
```

## 注意事項

- 本コードは有限要素法の仕組みを理解するための教育用です。実務の設計・解析への使用は想定していません。
- 結果の正確性について保証はできません。ご利用は自己責任でお願いします。
