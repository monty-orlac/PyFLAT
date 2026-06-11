import typing
import numpy as np
import numpy.typing as npt
import scipy.linalg
import abc

# Numpy型ヒントのエイリアスを作成
# 便宜上MatrixTypeとVectorTypeに分けたが、内部的な取り扱いは一緒
type MatrixType = npt.NDArray[np.float64]
type VectorType = npt.NDArray[np.float64]

# np.ndarrayの出力フォーマットのカスタマイズ
np.set_printoptions(linewidth=200, formatter={
    "float_kind": lambda val: f"{val:10.3e}",
    "int_kind": lambda val: f"{val:8d}", })


# **********************************************************************
# **********************************************************************
# **
# **    行列ソルバーのフレームワーク定義
# **
# **********************************************************************
# **********************************************************************

# **********************************************************
#  IFactorized: 分解済みの行列を保持するクラス
# **********************************************************
class IFactorized(metaclass=abc.ABCMeta):
    __slots__ = []

    @property
    @abc.abstractmethod
    def size(self) -> int:
        ...

    @abc.abstractmethod
    def solve(self, rhs: VectorType) -> VectorType:
        ...


# **********************************************************
#  ISolver: 構築済み（分解前）の全体剛性行列を保持するクラス
# **********************************************************
class ISolver(metaclass=abc.ABCMeta):
    __slots__ = []

    @property
    @abc.abstractmethod
    def size(self) -> int:
        ...

    @abc.abstractmethod
    def factorize(self) -> IFactorized:
        ...


# **********************************************************
#  IBuilder: 全体剛性行列の構築を行うクラス
# **********************************************************
class IBuilder(metaclass=abc.ABCMeta):
    __slots__ = []

    @property
    @abc.abstractmethod
    def size(self) -> int:
        ...

    @abc.abstractmethod
    def add_value(self, i: int, j: int, value: float) -> None:
        ...

    def assemble(self, lm: list[int], matrix_local: MatrixType) -> None:
        assert matrix_local.shape == (len(lm), len(lm))

        for il, _ig in enumerate(lm):
            if _ig < 0:
                continue
            for jl, _jg in enumerate(lm[0:il+1]):
                if _jg < 0:
                    continue
                ig: int = max(_ig, _jg)
                jg: int = min(_ig, _jg)
                self.add_value(ig, jg, matrix_local[(il, jl)])
        return

    @abc.abstractmethod
    def complete(self) -> ISolver:
        ...


# **********************************************************
#  IShape: 非ゼロ構造を決定する段階の行列データを格納するクラス
# **********************************************************
class IShape(metaclass=abc.ABCMeta):
    __slots__ = []

    @abc.abstractmethod
    def assemble(self, lm: list[int]) -> None:
        ...

    @abc.abstractmethod
    def allocate(self) -> IBuilder:
        ...

    @classmethod
    @abc.abstractmethod
    def get_instance(cls, nsize: int) -> typing.Self:
        ...


# **********************************************************************
# **********************************************************************
# **
# **    Scipyの密行列ソルバーを使った実装 (Cholesky分解)
# **
# **********************************************************************
# **********************************************************************

class ScipyDenseFactorized(IFactorized):
    __slots__ = ["__nsize", "__factorized"]

    @property
    def size(self) -> int:
        return self.__nsize

    def __init__(self, nsize: int, factorized: tuple[MatrixType, bool]):
        self.__nsize: int = nsize
        self.__factorized: tuple[MatrixType, bool] = factorized
        return

    def solve(self, rhs: VectorType) -> VectorType:
        ans: VectorType = scipy.linalg.cho_solve(self.__factorized, rhs, overwrite_b=True)
        return ans


# **********************************************************
class ScipyDenseSolver(ISolver):
    __slots__ = ["__matrix"]

    @property
    def size(self) -> int:
        return self.__matrix.shape[0]

    def __init__(self, matrix: MatrixType):
        self.__matrix: MatrixType = matrix
        return

    def factorize(self) -> IFactorized:
        factorized: tuple[MatrixType, bool] = scipy.linalg.cho_factor(self.__matrix, lower=True, overwrite_a=True)  # type: ignore
        return ScipyDenseFactorized(self.size, factorized)


# **********************************************************
class ScipyDenseBuilder(IBuilder):
    __slots__ = ["__matrix"]

    @property
    def size(self) -> int:
        return self.__matrix.shape[0]

    def __init__(self, nsize: int):
        self.__matrix: MatrixType = np.zeros((nsize, nsize))
        return

    def add_value(self, i: int, j: int, value: float) -> None:
        assert i >= j
        self.__matrix[i, j] += value
        return

    def complete(self) -> ISolver:
        return ScipyDenseSolver(self.__matrix)


# **********************************************************
class ScipyDenseShape(IShape):
    __slots__ = ["__nsize"]

    def __init__(self, nsize: int):
        self.__nsize: int = nsize
        return

    def assemble(self, lm: list[int]) -> None:
        return

    def allocate(self) -> IBuilder:
        return ScipyDenseBuilder(self.__nsize)

    @classmethod
    def get_instance(cls, nsize: int) -> typing.Self:
        return cls(nsize)
