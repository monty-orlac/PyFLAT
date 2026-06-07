import collections.abc as cabc
import numpy as np
import numpy.typing as npt

# Numpy型ヒントのエイリアスを作成
# 便宜上MatrixTypeとVectorTypeに分けたが、内部的な取り扱いは一緒
type MatrixType = npt.NDArray[np.float64]
type VectorType = npt.NDArray[np.float64]


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
