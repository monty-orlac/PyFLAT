import collections.abc as cabc
import numpy as np
import numpy.typing as npt

import linalg
import fe

# Numpy型ヒントのエイリアスを作成
# 便宜上MatrixTypeとVectorTypeに分けたが、内部的な取り扱いは一緒
type MatrixType = npt.NDArray[np.float64]
type VectorType = npt.NDArray[np.float64]


# **********************************************************
#   各種処理を行うための関数
# **********************************************************
def reorder(all_nodes: cabc.Iterable[fe.Node]) -> int:
    neq: int = 0
    for node in all_nodes:
        for i in range(len(node.dof)):
            if node.dof[i] < -1:
                node.dof[i] = neq
                neq += 1
    return neq


# **********************************************************
#   メインプログラム
# **********************************************************
def main():

    # ******************************************************
    #   解析モデルの定義
    # ******************************************************
    all_nodes: list[fe.Node] = [
        fe.Node(np.array(( 0.0000,  0.0000)), [-1, -1]),
        fe.Node(np.array((10.0000,  0.0000)), [-2, -1]),
        fe.Node(np.array((10.0000, 10.0000)), [-2, -2]),
        fe.Node(np.array(( 0.0000, 10.0000)), [-2, -2]),
        fe.Node(np.array(( 2.5000,  1.5000)), [-2, -2]),
        fe.Node(np.array(( 7.0000,  2.0000)), [-2, -2]),
        fe.Node(np.array(( 8.5000,  7.0000)), [-2, -2]),
        fe.Node(np.array(( 2.2000,  7.2000)), [-2, -2]),
    ]

    prop: fe.PropertyBase = fe.PropertyPE(1.4000e10, 0.3, 0.8)

    all_elems: list[fe.ElementBase] = [
        fe.ElementPEQUAD4(prop, [all_nodes[0], all_nodes[1], all_nodes[5], all_nodes[4]]),
        fe.ElementPEQUAD4(prop, [all_nodes[1], all_nodes[2], all_nodes[6], all_nodes[5]]),
        fe.ElementPEQUAD4(prop, [all_nodes[2], all_nodes[3], all_nodes[7], all_nodes[6]]),
        fe.ElementPEQUAD4(prop, [all_nodes[3], all_nodes[0], all_nodes[4], all_nodes[7]]),
        fe.ElementPEQUAD4(prop, [all_nodes[4], all_nodes[5], all_nodes[6], all_nodes[7]]),
    ]

    all_conds: list[fe.ConditionBase] = [
        fe.PointLoad(all_nodes[2], 1, 10000.0),
        fe.PointLoad(all_nodes[3], 1, 10000.0),
    ]

    solver_type: str = "DIRECT"

    # ******************************************************
    #   全体剛性行列の構築
    # ******************************************************
    neq: int = reorder(all_nodes)

    matrix_shape: linalg.IShape = linalg.SOLVER_TYPES[solver_type].get_instance(neq)
    for elem in all_elems:
        matrix_shape.assemble(elem.get_lm())

    # 全要素の要素剛性行列を計算し、全体剛性行列へ足し込む
    matrix_builder: linalg.IBuilder = matrix_shape.allocate()
    for elem in all_elems:
        matrix_local: MatrixType = elem.calc_stiffness_matrix()
        matrix_builder.assemble(elem.get_lm(), matrix_local)

    matrix_solver: linalg.ISolver = matrix_builder.complete()

    # 荷重ベクトル（右辺ベクトル rhs）を組み立てる
    rhs: VectorType = np.zeros(neq)
    for cond in all_conds:
        if isinstance(cond, fe.ConditionLoad):
            cond.apply(rhs)

    # ******************************************************
    #   行列の求解
    # ******************************************************
    # 求解のための準備を行う
    matrix_factorized: linalg.IFactorized = matrix_solver.factorize()

    # 求解後の rhs は変位ベクトルとなる
    rhs: VectorType = matrix_factorized.solve(rhs)

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
    print(" {:>7s} {:>3s} {:>15s} {:>15s} {:>15s} {:>15s}".format("eid", "ip", "str11", "str22", "str33", "str12"))
    for label, elem in enumerate(all_elems, start=1):
        ans_elem: VectorType = np.array([rhs[i] if i >= 0 else 0.0 for i in elem.get_lm()])
        results: cabc.Sequence[VectorType] = elem.calc_result(ans_elem)
        for ip, result in enumerate(results, start=1):
            print(f" {label:7d} {ip:3}" + "".join([" {:15.6e}".format(v) for v in result]))
    print()


if __name__ == "__main__":
    main()
