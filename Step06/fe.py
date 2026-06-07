import abc
import typing
import dataclasses
import collections.abc as cabc
import numpy as np
import numpy.typing as npt

import fe_lib as lib

# Numpy型ヒントのエイリアスを作成
# 便宜上MatrixTypeとVectorTypeに分けたが、内部的な取り扱いは一緒
type MatrixType = npt.NDArray[np.float64]
type VectorType = npt.NDArray[np.float64]


@dataclasses.dataclass
class Node:
    coord: VectorType               # 節点座標 [x, y]
    dof: cabc.MutableSequence[int]  # 各方向の全体自由度番号 [x, y]（-1は変位ゼロで拘束）


class PropertyBase:
    __slots__ = []


class IPropertyTRUSS(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_dee_truss(self) -> float:
        ...

    @abc.abstractmethod
    def get_area(self) -> float:
        ...


class IPropertyPE(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_dee_pe_solver(self) -> MatrixType:
        ...

    @abc.abstractmethod
    def get_dee_pe_result(self) -> MatrixType:
        ...

    @abc.abstractmethod
    def get_thickness(self) -> float:
        ...


class PropertyTRUSS(PropertyBase, IPropertyTRUSS):
    __slots__ = ["_young", "_area"]

    def __init__(self, young: float, area: float):
        self._young: float = young
        self._area: float = area
        return

    def get_dee_truss(self) -> float:
        return self._young

    def get_area(self) -> float:
        return self._area


class PropertyPE(PropertyBase, IPropertyPE):
    __slots__ = ["_young", "_rpois", "_thickness"]

    def __init__(self, young: float, rpois: float, thickness: float):
        self._young: float = young
        self._rpois: float = rpois
        self._thickness: float = thickness
        return

    def get_dee_pe_solver(self) -> MatrixType:
        nu = self._rpois
        coeff: float = self._young / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return np.array([
            [1.0-nu, nu,     0.0],
            [nu,     1.0-nu, 0.0],
            [0.0,    0.0,    0.5-nu],
        ]) * coeff

    def get_dee_pe_result(self) -> MatrixType:
        nu = self._rpois
        coeff: float = self._young / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return np.array([
            [1.0-nu, nu,     0.0],
            [nu,     1.0-nu, 0.0],
            [nu,     nu,     0.0],
            [0.0,    0.0,    0.5-nu],
        ]) * coeff

    def get_thickness(self) -> float:
        return self._thickness


class ElementBase(metaclass=abc.ABCMeta):
    __slots__ = ["_prop", "_nodes"]

    NNODE: typing.ClassVar[int]

    def __init__(self, prop: PropertyBase, nodes: cabc.Sequence[Node]):
        assert len(nodes) == self.NNODE
        self._prop: PropertyBase = prop
        self._nodes: cabc.Sequence[Node] = nodes
        return

    # 要素の自由度番号リスト（lm）を [節点0のx, 節点0のy, 節点1のx, 節点1のy] の順で返す
    def get_lm(self) -> cabc.Sequence[int]:
        return [dof for node in self._nodes for dof in node.dof]

    # 構成節点の座標を行列にまとめて返す（1行が1節点の [x, y]）
    def get_coord_matrix(self) -> MatrixType:
        return np.vstack([it.coord for it in self._nodes], )

    @abc.abstractmethod
    def calc_stiffness_matrix(self) -> MatrixType:
        ...

    @abc.abstractmethod
    def calc_result(self, ans_elem: VectorType) -> cabc.Sequence[VectorType]:
        ...


class ElementTRUSS(ElementBase):
    __slots__ = []

    NNODE: typing.ClassVar[int] = 2

    def calc_stiffness_matrix(self) -> MatrixType:
        assert isinstance(self._prop, IPropertyTRUSS)
        matrix_local = lib.calc_element_truss(
            self.get_coord_matrix(),
            self._prop.get_dee_truss(),
            self._prop.get_area())
        assert isinstance(matrix_local, np.ndarray)
        return matrix_local

    def calc_result(self, ans_elem: VectorType) -> cabc.Sequence[VectorType]:
        assert isinstance(self._prop, IPropertyTRUSS)
        result = lib.calc_element_truss(
            self.get_coord_matrix(),
            self._prop.get_dee_truss(),
            self._prop.get_area(),
            ans_elem)
        assert isinstance(result, list)
        return result


class ElementPETRIA3(ElementBase):
    __slots__ = []

    NNODE: typing.ClassVar[int] = 3

    def calc_stiffness_matrix(self) -> MatrixType:
        assert isinstance(self._prop, IPropertyPE)
        matrix_local = lib.calc_element_plane2d_full(
            self.get_coord_matrix(),
            self._prop.get_dee_pe_solver(),
            self._prop.get_thickness())
        assert isinstance(matrix_local, np.ndarray)
        return matrix_local

    def calc_result(self, ans_elem: VectorType) -> cabc.Sequence[VectorType]:
        assert isinstance(self._prop, IPropertyPE)
        result = lib.calc_element_plane2d_full(
            self.get_coord_matrix(),
            self._prop.get_dee_pe_result(),
            self._prop.get_thickness(),
            ans_elem)
        assert isinstance(result, list)
        return result


class ElementPEQUAD4(ElementBase):
    __slots__ = []

    NNODE: typing.ClassVar[int] = 4

    def calc_stiffness_matrix(self) -> MatrixType:
        assert isinstance(self._prop, IPropertyPE)
        matrix_local = lib.calc_element_plane2d_full(
            self.get_coord_matrix(),
            self._prop.get_dee_pe_solver(),
            self._prop.get_thickness())
        assert isinstance(matrix_local, np.ndarray)
        return matrix_local

    def calc_result(self, ans_elem: VectorType) -> cabc.Sequence[VectorType]:
        assert isinstance(self._prop, IPropertyPE)
        result = lib.calc_element_plane2d_full(
            self.get_coord_matrix(),
            self._prop.get_dee_pe_result(),
            self._prop.get_thickness(),
            ans_elem)
        assert isinstance(result, list)
        return result


class ElementPETRIA6(ElementBase):
    __slots__ = []

    NNODE: typing.ClassVar[int] = 6

    def calc_stiffness_matrix(self) -> MatrixType:
        assert isinstance(self._prop, IPropertyPE)
        matrix_local = lib.calc_element_plane2d_full(
            self.get_coord_matrix(),
            self._prop.get_dee_pe_solver(),
            self._prop.get_thickness())
        assert isinstance(matrix_local, np.ndarray)
        return matrix_local

    def calc_result(self, ans_elem: VectorType) -> cabc.Sequence[VectorType]:
        assert isinstance(self._prop, IPropertyPE)
        result = lib.calc_element_plane2d_full(
            self.get_coord_matrix(),
            self._prop.get_dee_pe_result(),
            self._prop.get_thickness(),
            ans_elem)
        assert isinstance(result, list)
        return result


class ConditionBase(metaclass=abc.ABCMeta):
    __slots__ = []


class ConditionLoad(ConditionBase, metaclass=abc.ABCMeta):
    __slots__ = []

    @abc.abstractmethod
    def apply(self, rhs: VectorType) -> None:
        ...


class PointLoad(ConditionLoad):
    __slots__ = ["__node", "__dof", "__value"]

    def __init__(self, node: Node, dof: int, value: float):
        self.__node: Node = node
        self.__dof: int = dof
        self.__value: float = value
        return

    def apply(self, rhs: VectorType) -> None:
        idof: int = self.__node.dof[self.__dof]
        if idof >= 0:
            rhs[idof] += self.__value
        return
