import dataclasses
import numpy as np
import numpy.typing as npt
import scipy.linalg

# Numpy型ヒントのエイリアスを作成
# 便宜上MatrixTypeとVectorTypeに分けたが、内部的な取り扱いは一緒
type MatrixType = npt.NDArray[np.float64]
type VectorType = npt.NDArray[np.float64]

# np.ndarrayの出力フォーマットのカスタマイズ
np.set_printoptions(linewidth=200, formatter={
    "float_kind": lambda val: f"{val:10.3e}",
    "int_kind": lambda val: f"{val:8d}", })


# **********************************************************
#   解析モデルを格納するための構造体
# **********************************************************
@dataclasses.dataclass
class Node:
    coord: VectorType   # 節点座標 [x, y]
    dof: list[int]      # 各方向の全体自由度番号 [x, y]（-1は変位ゼロで拘束）


@dataclasses.dataclass
class PropertyTRUSS:
    young: float        # ヤング率
    area: float         # 断面積


@dataclasses.dataclass
class ElementTRUSS:
    nodes: list[Node]           # 構成節点（トラスは始点・終点の2節点）
    property: PropertyTRUSS     # 材料・断面特性

    # 要素の自由度番号リスト（lm）を [節点0のx, 節点0のy, 節点1のx, 節点1のy] の順で返す
    def get_lm(self) -> list[int]:
        return [dof for node in self.nodes for dof in node.dof]

    # 構成節点の座標を行列にまとめて返す（1行が1節点の [x, y]）
    def get_coord_matrix(self) -> MatrixType:
        return np.vstack([it.coord for it in self.nodes], )


@dataclasses.dataclass
class PointLoad:
    node: Node      # 荷重を載荷する節点
    dof: int        # 載荷方向（0: x方向, 1: y方向）
    value: float    # 荷重値


# **********************************************************
#   各種処理を行うための関数
# **********************************************************
def calc_stiffness_matrix_truss(coords: MatrixType, young: float, area: float) -> MatrixType:
    # 要素軸方向ベクトルと要素長を求め、単位方向ベクトルに正規化する
    ldir: VectorType = coords[1, :] - coords[0, :]
    lenxy: float = float(np.linalg.norm(ldir))
    ldir /= lenxy
    # Bマトリクスの計算
    bee: MatrixType = np.array([[-ldir[0], -ldir[1], +ldir[0], +ldir[1]], ]) / lenxy

    # 要素剛性行列の計算
    matrix_local: MatrixType = bee.transpose() * young @ bee
    matrix_local *= area * lenxy
    return matrix_local


def calc_result_truss(coords: MatrixType, young: float, area: float, ans_elem: VectorType) -> list[VectorType]:
    # Bマトリクスの計算までは要素剛性行列の計算と全く同じ
    ldir: VectorType = coords[1, :] - coords[0, :]
    lenxy: float = float(np.linalg.norm(ldir))
    ldir /= lenxy
    bee: MatrixType = np.array([[-ldir[0], -ldir[1], +ldir[0], +ldir[1]], ]) / lenxy

    # 軸応力の計算
    stress: float = float(bee.transpose()[:, 0].dot(ans_elem) * young)
    return [np.array([stress, ]), ]


def assemble_matrix(matrix_global: MatrixType, lm: list[int], matrix_local: MatrixType) -> None:
    # 要素剛性行列のサイズは自由度数（=lmの長さ）と一致するはず
    assert matrix_local.shape == (len(lm), len(lm))

    # 要素剛性行列の各成分を、lm が示す全体自由度番号の位置へ加算していく
    for il, _ig in enumerate(lm):
        if _ig < 0:         # 拘束自由度（-1）は全体行列に存在しないのでスキップ
            continue
        for jl, _jg in enumerate(lm[0:il+1]):
            if _jg < 0:     # 同上
                continue
            # 対称行列の下三角部分のみ格納するため、行番号 ig >= 列番号 jg に振り分ける
            ig: int = max(_ig, _jg)
            jg: int = min(_ig, _jg)
            matrix_global[ig, jg] += matrix_local[il, jl]
    return


# **********************************************************
#   メインプログラム
# **********************************************************
def main():

    # ******************************************************
    #   解析モデルの定義
    # ******************************************************

    neq: int = 7    # 全体の自由度数（拘束されていない自由度の総数）

    # 節点：座標と自由度番号を定義（-1 は変位ゼロで拘束された自由度）
    all_nodes: list[Node] = [
        Node(np.array((0.000000, 0.000000)), [-1, -1]),
        Node(np.array((1.000000, 1.600000)), [ 0,  1]),
        Node(np.array((2.000000, 0.000000)), [ 2,  3]),
        Node(np.array((3.000000, 1.600000)), [ 4,  5]),
        Node(np.array((4.000000, 0.000000)), [ 6, -1]),
    ]

    # 材料・断面特性（全要素で共通：ヤング率 210 GPa、断面積 4.5e-4 m^2）
    prop: PropertyTRUSS = PropertyTRUSS(210.0e+9, 4.5000e-4)

    # 要素：それぞれ始点・終点となる2節点で構成
    all_elems: list[ElementTRUSS] = [
        ElementTRUSS([all_nodes[0], all_nodes[1]], prop),
        ElementTRUSS([all_nodes[0], all_nodes[2]], prop),
        ElementTRUSS([all_nodes[1], all_nodes[2]], prop),
        ElementTRUSS([all_nodes[1], all_nodes[3]], prop),
        ElementTRUSS([all_nodes[2], all_nodes[3]], prop),
        ElementTRUSS([all_nodes[2], all_nodes[4]], prop),
        ElementTRUSS([all_nodes[3], all_nodes[4]], prop),
    ]

    # 荷重：3番目の節点の y方向に -250 kN を載荷
    all_loads: list[PointLoad] = [
        PointLoad(all_nodes[2], 1, -250000.0)
    ]

    # ******************************************************
    #   全体剛性行列の構築
    # ******************************************************
    matrix_global: MatrixType = np.zeros((neq, neq))

    # 全要素の要素剛性行列を計算し、全体剛性行列へ足し込む
    for elem in all_elems:
        coords: MatrixType = elem.get_coord_matrix()
        matrix_local: MatrixType = calc_stiffness_matrix_truss(coords, elem.property.young, elem.property.area)
        assemble_matrix(matrix_global, elem.get_lm(), matrix_local)

    # 荷重ベクトル（右辺ベクトル rhs）を組み立てる
    rhs: VectorType = np.zeros(neq)
    for load in all_loads:
        rhs[load.node.dof[load.dof]] += load.value

    # ******************************************************
    #   行列の求解
    # ******************************************************
    # 全体剛性行列は対称正定値なので、Cholesky分解により求解する
    factorized = scipy.linalg.cho_factor(matrix_global, lower=True, overwrite_a=True)

    rhs = scipy.linalg.cho_solve(factorized, rhs, overwrite_b=True)

    # ******************************************************
    #   結果表示
    # ******************************************************
    # 節点変位（拘束された自由度は 0.0 として表示）
    print("Nodal displacement result:")
    print(" {:>7s} {:>15s} {:>15s}".format("nid", "x-disp", "y-disp"))
    for label, node in enumerate(all_nodes, start=1):
        rhs_node: list[float] = [rhs[idof] if idof >= 0 else 0.0 for idof in node.dof]
        print(" {:7d} {:15.6e} {:15.6e}".format(label, rhs_node[0], rhs_node[1]))
    print()

    # 要素応力
    print("Element stress result:")
    print(" {:>7s} {:>15s}".format("eid", "str11"))
    for label, elem in enumerate(all_elems, start=1):
        coords: MatrixType = elem.get_coord_matrix()
        ans_elem: VectorType = np.array([rhs[i] if i >= 0 else 0.0 for i in elem.get_lm()])
        results: list[VectorType] = calc_result_truss(coords, elem.property.young, elem.property.area, ans_elem)
        for result in results:
            print(f" {label:7d} " + "".join([" {:15.6e}".format(v) for v in result]))
    print()


if __name__ == "__main__":
    main()
