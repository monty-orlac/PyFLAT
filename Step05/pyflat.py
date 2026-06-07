import abc
import typing
import dataclasses
import collections.abc as cabc
import numpy as np
import numpy.typing as npt

import linalg

# Numpy型ヒントのエイリアスを作成
# 便宜上MatrixTypeとVectorTypeに分けたが、内部的な取り扱いは一緒
type MatrixType = npt.NDArray[np.float64]
type VectorType = npt.NDArray[np.float64]


# **********************************************************
#   解析モデルを格納するための構造体
# **********************************************************
@dataclasses.dataclass
class Node:
    coord: VectorType               # 節点座標 [x, y]
    dof: cabc.MutableSequence[int]  # 各方向の全体自由度番号 [x, y]（-1は変位ゼロで拘束）


@dataclasses.dataclass
class PropertyTRUSS:
    young: float        # ヤング率
    area: float         # 断面積


@dataclasses.dataclass
class PropertyPE:
    young: float
    rpois: float
    thickness: float

    def get_dee_pe_solver(self) -> MatrixType:
        nu = self.rpois
        coeff: float = self.young / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return np.array([
            [1.0-nu, nu,     0.0],
            [nu,     1.0-nu, 0.0],
            [0.0,    0.0,    0.5-nu],
        ]) * coeff

    def get_dee_pe_result(self) -> MatrixType:
        nu = self.rpois
        coeff: float = self.young / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return np.array([
            [1.0-nu, nu,     0.0],
            [nu,     1.0-nu, 0.0],
            [nu,     nu,     0.0],
            [0.0,    0.0,    0.5-nu],
        ]) * coeff


@dataclasses.dataclass
class ElementBase(metaclass=abc.ABCMeta):
    nodes: cabc.Sequence[Node]  # 構成節点

    NNODE: typing.ClassVar[int]

    def __post_init__(self):
        assert len(self.nodes) == self.NNODE
        return

    # 要素の自由度番号リスト（lm）を [節点0のx, 節点0のy, 節点1のx, 節点1のy] の順で返す
    def get_lm(self) -> cabc.Sequence[int]:
        return [dof for node in self.nodes for dof in node.dof]

    # 構成節点の座標を行列にまとめて返す（1行が1節点の [x, y]）
    def get_coord_matrix(self) -> MatrixType:
        return np.vstack([it.coord for it in self.nodes], )

    @abc.abstractmethod
    def calc_stiffness_matrix(self) -> MatrixType:
        ...

    @abc.abstractmethod
    def calc_result(self, ans_elem: VectorType) -> cabc.Sequence[VectorType]:
        ...


@dataclasses.dataclass
class ElementTRUSS(ElementBase):
    property: PropertyTRUSS

    NNODE: typing.ClassVar[int] = 2

    def calc_stiffness_matrix(self) -> MatrixType:
        matrix_local = calc_element_truss(
            self.get_coord_matrix(),
            self.property.young,
            self.property.area)
        assert isinstance(matrix_local, np.ndarray)
        return matrix_local

    def calc_result(self, ans_elem: VectorType) -> cabc.Sequence[VectorType]:
        result = calc_element_truss(
            self.get_coord_matrix(),
            self.property.young,
            self.property.area,
            ans_elem)
        assert isinstance(result, list)
        return result


@dataclasses.dataclass
class ElementPETRIA3(ElementBase):
    property: PropertyPE

    NNODE: typing.ClassVar[int] = 3

    def calc_stiffness_matrix(self) -> MatrixType:
        matrix_local = calc_element_plane2d_full(
            self.get_coord_matrix(),
            self.property.get_dee_pe_solver(),
            self.property.thickness)
        assert isinstance(matrix_local, np.ndarray)
        return matrix_local

    def calc_result(self, ans_elem: VectorType) -> cabc.Sequence[VectorType]:
        result = calc_element_plane2d_full(
            self.get_coord_matrix(),
            self.property.get_dee_pe_result(),
            self.property.thickness,
            ans_elem)
        assert isinstance(result, list)
        return result


@dataclasses.dataclass
class ElementPEQUAD4(ElementBase):
    property: PropertyPE

    NNODE: typing.ClassVar[int] = 4

    def calc_stiffness_matrix(self) -> MatrixType:
        matrix_local = calc_element_plane2d_full(
            self.get_coord_matrix(),
            self.property.get_dee_pe_solver(),
            self.property.thickness)
        assert isinstance(matrix_local, np.ndarray)
        return matrix_local

    def calc_result(self, ans_elem: VectorType) -> cabc.Sequence[VectorType]:
        result = calc_element_plane2d_full(
            self.get_coord_matrix(),
            self.property.get_dee_pe_result(),
            self.property.thickness,
            ans_elem)
        assert isinstance(result, list)
        return result


@dataclasses.dataclass
class ElementPETRIA6(ElementBase):
    property: PropertyPE

    NNODE: typing.ClassVar[int] = 6

    def calc_stiffness_matrix(self) -> MatrixType:
        matrix_local = calc_element_plane2d_full(
            self.get_coord_matrix(),
            self.property.get_dee_pe_solver(),
            self.property.thickness)
        assert isinstance(matrix_local, np.ndarray)
        return matrix_local

    def calc_result(self, ans_elem: VectorType) -> cabc.Sequence[VectorType]:
        result = calc_element_plane2d_full(
            self.get_coord_matrix(),
            self.property.get_dee_pe_result(),
            self.property.thickness,
            ans_elem)
        assert isinstance(result, list)
        return result


@dataclasses.dataclass
class PointLoad:
    node: Node      # 荷重を載荷する節点
    dof: int        # 載荷方向（0: x方向, 1: y方向）
    value: float    # 荷重値


# **********************************************************
#   各種処理を行うための関数
# **********************************************************
def calc_element_truss(
        coords: MatrixType, young: float, area: float,
        ans_elem: VectorType | None = None,
        ) -> MatrixType | cabc.Sequence[VectorType]:
    # 要素軸方向ベクトルと要素長を求め、単位方向ベクトルに正規化する
    ldir: VectorType = coords[1, :] - coords[0, :]
    lenxy: float = float(np.linalg.norm(ldir))
    ldir /= lenxy
    # Bマトリクスの計算
    bee: MatrixType = np.array([[-ldir[0], -ldir[1], +ldir[0], +ldir[1]], ]) / lenxy

    # 要素剛性行列の計算
    if ans_elem is None:
        matrix_local: MatrixType = bee.transpose() * young @ bee
        matrix_local *= area * lenxy
        return matrix_local

    # 軸応力の計算
    stress: float = float(bee.transpose()[:, 0].dot(ans_elem) * young)
    return [np.array([stress, ]), ]


def _gauss_info_2d(nnode: int, nord: int) -> cabc.Iterator[tuple[float, float, float]]:
    if nnode == 3:
        if nord == 0:
            yield (1.0/3.0, 1.0/3.0, 1.0/2.0)
            return
    elif nnode == 4:
        if nord == 0:
            yield (-0.5773502692, -0.5773502692, 1.0)
            yield ( 0.5773502692, -0.5773502692, 1.0)
            yield (-0.5773502692,  0.5773502692, 1.0)
            yield ( 0.5773502692,  0.5773502692, 1.0)
            return
        elif nord == -1:
            yield (0.0, 0.0, 4.0)
            return
    elif nnode == 6:
        if nord == 0:
            yield (1.0/6.0, 1.0/6.0, 1.0/6.0)
            yield (4.0/6.0, 1.0/6.0, 1.0/6.0)
            yield (1.0/6.0, 4.0/6.0, 1.0/6.0)
            return
    raise AssertionError(f"Unknown (nnode, nord)=({nnode}, {nord}) in {__file__}._gauss_info_2d().")
    return


def _shape_derivs_2d(nnode: int, xi: float, eta: float) -> MatrixType:
    match nnode:
        case 3:
            return np.array([
                [-1.0, +1.0,  0.0],
                [-1.0,  0.0, +1.0], ])
        case 4:
            return np.array([
                [-0.25 * (1.0 - eta), +0.25 * (1.0 - eta), +0.25 * (1.0 + eta), -0.25 * (1.0 + eta)],
                [-0.25 * (1.0 - xi),  -0.25 * (1.0 + xi),  +0.25 * (1.0 + xi),  +0.25 * (1.0 - xi)], ])
        case 6:
            fn1: float = 1.0 - xi - eta
            fn2: float = xi
            fn3: float = eta
            return np.array([
                [-4.0*fn1+1.0, 4.0*fn2-1.0, 0.0,         +4.0*(fn1-fn2), +4.0*fn3, -4.0*fn3,],
                [-4.0*fn1+1.0, 0.0,         4.0*fn3-1.0, -4.0*fn2,       +4.0*fn2, +4.0*(fn1-fn3),],
            ])
        case _:
            pass
    raise AssertionError(f"Unknown nnode={nnode} in {__file__}._shape_derivs_2d().")


def calc_element_plane2d_full(coords: MatrixType, dee: MatrixType, thickness: float, ans_elem: VectorType | None = None) -> MatrixType | list[VectorType]:

    nnode: int = coords.shape[0]
    kstiff: MatrixType = np.zeros((nnode*2, nnode*2))
    stress_ip: list[VectorType] = []

    # Gauss積分点ごとにループを回す
    for (xi, eta, weight) in _gauss_info_2d(nnode, 0):
        # 形状関数の自然座標微分 dN/dxi, dN/deta の取得 (size: 2 x NNODE)
        #   dslocal[0,i] = dN_i/dxi
        #   dslocal[1,i] = dN_i/deta
        dslocal: MatrixType = _shape_derivs_2d(nnode, xi, eta)

        # ヤコビ行列 [J] (size: 2 x 2)
        jac: MatrixType = dslocal @ coords
        jdet: float = np.linalg.det(jac)
        jinv = np.linalg.inv(jac)

        # 形状関数の物理座標微分 dN/dx, dN/dy の計算 (size: 2 x NNODE)
        dsglobal: MatrixType = jinv @ dslocal

        # Bマトリクスの組み立て (size: 3 x (NNODE*2))
        bee: MatrixType = np.zeros((3, nnode*2))
        for i in range(nnode):
            i0: int = 2 * i + 0
            i1: int = i0 + 1
            bee[0, i0] = dsglobal[0, i]
            bee[1, i1] = dsglobal[1, i]
            bee[2, i0] = dsglobal[1, i]
            bee[2, i1] = dsglobal[0, i]

        # 要素剛性行列が与えられている場合は計算
        if ans_elem is None:
            # 要素剛性行列の計算
            kstiff += bee.transpose() @ dee @ bee * (jdet * thickness * weight)
        else:
            stress: VectorType = dee @ bee @ ans_elem
            stress_ip.append(stress)

    if ans_elem is None:
        return kstiff
    return stress_ip


def reorder(all_nodes: cabc.Iterable[Node]) -> int:
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
    all_nodes: list[Node] = [
        Node(np.array(( 0.0000,  0.0000)), [-1, -1]),
        Node(np.array((10.0000,  0.0000)), [-2, -1]),
        Node(np.array((10.0000, 10.0000)), [-2, -2]),
        Node(np.array(( 0.0000, 10.0000)), [-2, -2]),
        Node(np.array(( 2.5000,  1.5000)), [-2, -2]),
        Node(np.array(( 7.0000,  2.0000)), [-2, -2]),
        Node(np.array(( 8.5000,  7.0000)), [-2, -2]),
        Node(np.array(( 2.2000,  7.2000)), [-2, -2]),
    ]

    prop: PropertyPE = PropertyPE(1.4000e10, 0.3, 0.8)

    all_elems: list[ElementBase] = [
        ElementPEQUAD4([all_nodes[0], all_nodes[1], all_nodes[5], all_nodes[4]], prop),
        ElementPEQUAD4([all_nodes[1], all_nodes[2], all_nodes[6], all_nodes[5]], prop),
        ElementPEQUAD4([all_nodes[2], all_nodes[3], all_nodes[7], all_nodes[6]], prop),
        ElementPEQUAD4([all_nodes[3], all_nodes[0], all_nodes[4], all_nodes[7]], prop),
        ElementPEQUAD4([all_nodes[4], all_nodes[5], all_nodes[6], all_nodes[7]], prop),
    ]

    all_loads: list[PointLoad] = [
        PointLoad(all_nodes[2], 1, 10000.0),
        PointLoad(all_nodes[3], 1, 10000.0),
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
    for load in all_loads:
        rhs[load.node.dof[load.dof]] += load.value

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
