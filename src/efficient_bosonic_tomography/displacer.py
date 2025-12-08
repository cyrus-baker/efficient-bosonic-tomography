from typing import List
import jax
import jax.numpy as jnp


import dynamiqs as dq
import equinox as eqx

from deprecated import deprecated


class Displacer(eqx.Module):  # 总得来说是根据之前学长提出来的方法去计算位移算符，输入n，得到输出的n阶位移算符
    evals: jnp.ndarray
    evecs: jnp.ndarray
    range: jnp.ndarray
    n: int = eqx.field(static=True)  # 一个常数

    def __init__(self, n):
        # The off-diagonal of the real-symmetric similar matrix T.
        sym = jnp.sqrt(jnp.arange(1, n))  # 从0到n-1的数组，加一个根号
        self.n = n
        # Solve the eigensystem.

        # construct a tri-diagonal matrix
        _mat = jnp.diag(sym, 1) + jnp.diag(sym, -1)  # 1表示上对角线，-1表示下对角线

        self.evals, self.evecs = jax.scipy.linalg.eigh(_mat)  # 特征值与特征向量
        self.range = jnp.arange(n)  # 从0到n-1的数组

    @jax.jit
    def displace(self, alpha):
        r = jnp.abs(alpha) + 1e-10

        eit = alpha / r  # abbreviation for exp(i*theta)  #相位

        Dk_diag = (1 / (1j * eit)) ** self.range  # 相位序列
        Dk_inv_diag = (1j * eit) ** self.range  # 另一个序列

        Dk_diag = jnp.nan_to_num(Dk_diag, nan=1.0)  # 将non转换为1
        Dk_inv_diag = jnp.nan_to_num(Dk_inv_diag, nan=1.0)  # 将non转换为1

        exp_diag = jnp.exp(-1j * r * self.evals)

        Da = (
            (Dk_inv_diag.reshape(-1, 1) * self.evecs)
            @ jnp.diag(exp_diag)
            @ (jnp.conj(self.evecs.T) * Dk_diag)
        )

        return Da



@deprecated(reason="Use Alpha2Row instead", version="0.1.0")
def alpha2row(alpha_list, rho_h=None, N=10):
    # 在 JIT 编译之前进行 rho_h 的判断
    if rho_h is None:
        rho_h = dq.basis_dm(N, 0)  # 生成N维的0态的密度矩阵
    a = dq.destroy(N).to_jax()  # 生成N维的湮灭算符

    # 将主要的计算逻辑 jit 编译
    @jax.jit
    def core(alpha):
        def displacement(alpha):
            return jax.scipy.linalg.expm(alpha * a.conj().T - alpha.conj() * a)
            # 注意，这里的alpha输入是一个数组，而不是一个数，因此要用conj()

        Da = displacement(alpha)
        _oper = Da @ rho_h @ Da.conj().T
        # 这里把密度矩阵左右都用位移算符作用了一下

        # 矢量化 _oper
        oper_vec = _oper.T.flatten(order="F")  # 按列优先进行扁平化
        return oper_vec

    return jax.vmap(core)(alpha_list)
    # vmap是一个非常神奇的函数，它会把输入分开，分别送入函数，再把结果拼起来。这里以N=40为例讲解
    # 函数core中的alpha是一个数，是1600个复数中的一个，每个数都执行了core函数，然后再把结果拼起来
    # 输出是40*40，再通过flatten拍扁为（1,1600），然后拼起来是（1600，1600)，即每一列（或者行，要再看看）表示网格中的一个结果
    # 即一列对应于一个值相应的位移算符


class Alpha2Row(eqx.Module):  # 定义一个类
    rho_h: jnp.ndarray
    N: int = eqx.field(static=True)
    displacer: eqx.Module
    N_compute: int | None = eqx.field(static=True)

    def __init__(self, rho_h=None, N=5, N_compute=None):  # 这个是类的初始化函数
        self.N = N
        self.N_compute = N_compute if N_compute is not None else N
        if rho_h is None:
            rho_h = dq.basis_dm(N_compute, 0).to_jax()  # 生成N维的0态的密度矩阵
        self.rho_h = jnp.array(rho_h)  # 转换变量为jnp.array

        self.displacer = Displacer(N_compute)  # 生成位移算符，维度为N_compute

    # 将主要的计算逻辑 jit 编译
    @jax.jit
    def __call__(self, alpha_vec):
        def core(alpha):
            Da = self.displacer.displace(
                alpha
            )  # 调用了前面Displacer类中的displace函数，即生成了一个前面所说的位移算符
            _oper = (Da @ self.rho_h @ Da.conj().T)[: self.N, : self.N]  # 截断到N维

            # 矢量化 _oper
            oper_vec = _oper.T.flatten(order="F")  # 二维到一维
            return oper_vec

        return jax.vmap(core)(alpha_vec)


# 这个类中，N是截断维度，N_compute是计算维度，输出的结果为噪声情况情况下的测量算符，即13号文章的20式


class Alpha2RowMultiMode(eqx.Module):  # 多模的A
    rho_h: jnp.ndarray | List = eqx.field(static=True)
    N_single: int = eqx.field(static=True)
    num_modes: int = eqx.field(static=True)
    N_compute: int | None = eqx.field(static=True)
    displacer: eqx.Module

    def __init__(self, rho_h=None, N_single=5, num_modes=2, N_compute=None):
        self.N_single = N_single
        self.N_compute = N_compute if N_compute is not None else N_single
        self.num_modes = num_modes
        if rho_h is None:
            rho_h = [dq.basis_dm(N_compute, 0).to_jax()]  # 生成N维的0态的密度矩阵
            for _ in range(num_modes - 1):  # 从0开始，所以-1
                rho_h.append(dq.basis_dm(N_compute, 0).to_jax())
        self.rho_h = rho_h
        # if isinstance(rho_h, List): # 判断是否是列表
        #     self.rho_h = rho_h
        # else:
        #     self.rho_h = [rho_h]
        # 将num_modes - 1个密度矩阵一个个贴起来，并把rho_h转换为列表形式。

        # self.rho_h = jnp.array(rho_h)

        self.displacer = Displacer(N_compute)  # 生成位移算符，维度为N_compute

    # @jax.jit
    def __call__(self, alpha_vec):
        def core(alpha_multi):
            Da = [self.displacer.displace(alpha_single) for alpha_single in alpha_multi]
            # 调用了前面Displacer类中的displace函数，即生成了一个前面所说的位移算符

            # Da_pi = Da[0]
            res = (Da[0] @ self.rho_h[0] @ Da[0].conj().T)[
                : self.N_single, : self.N_single
            ]  # 可以试试把rho_h换为Parity函数  # 这个似乎是第0个密度矩阵？求第0个的投影算符
            for i, ds in enumerate(Da[1:]):  # 从第i个开始到最后一个
                # Da_pi = jnp.kron(Da_pi, ds)
                res = jnp.kron(
                    res,
                    (ds @ self.rho_h[i + 1] @ ds.conj().T)[
                        : self.N_single, : self.N_single
                    ],
                )  # 每一个做截断，然后乘出投影算符，直积起来

            # _oper = Da_pi @ self.rho_h @ Da_pi.conj().T
            _oper = res

            # 矢量化 _oper
            oper_vec = _oper.T.flatten(order="F")  # 矢量化
            return oper_vec

        return jax.vmap(core)(alpha_vec)

@deprecated(reason="Use Alpha2RowMultiMode instead", version="0.1.0")
def alpha2row_multimode(alpha_vec, rho_h=None, N_single=5, num_modes=2):
    if rho_h is None:
        rho_h = dq.basis_dm(N_single, 0).to_jax()
        for _ in range(num_modes - 1):
            rho_h = jnp.kron(rho_h, dq.basis_dm(N_single, 0).to_jax())
    a_list = [
        dq.tensor(
            *[
                dq.destroy(N_single).to_jax() if j == i else dq.eye(N_single).to_jax()
                for j in range(num_modes)
            ]
        ).to_jax()
        for i in range(num_modes)
    ]

    @jax.jit
    def core(alpha_multi):
        def displacement(alpha_single, a):
            return jax.scipy.linalg.expm(
                alpha_single * a.conj().T - alpha_single.conj() * a
            )

        Da = [
            displacement(alpha_single, a)
            for alpha_single, a in zip(alpha_multi, a_list)
        ]
        Da_pi = jnp.eye(N_single**num_modes)
        for ds in Da:
            Da_pi = Da_pi @ ds
        _oper = Da_pi @ rho_h @ Da_pi.conj().T

        # 矢量化 _oper
        oper_vec = _oper.T.flatten(order="F")
        return oper_vec

    return jax.vmap(core)(alpha_vec)


class Alpha2RowMultiModeNoNoise(eqx.Module):
    N_single: int = eqx.field(static=True)
    num_modes: int = eqx.field(static=True)

    def __init__(self, N_single=5, num_modes=2):
        self.N_single = N_single
        self.num_modes = num_modes

    @jax.jit
    def __call__(self, alpha_vec):
        def _coherent2(alpha, N):
            n_array = jnp.arange(0, N, dtype=jnp.float64)
            sqrtn = jnp.sqrt(jax.scipy.special.factorial(n_array))

            alpha_n_array = jnp.power(alpha, n_array) / sqrtn

            return jnp.reshape(alpha_n_array, (N, 1)) * jnp.exp(
                -0.5 * jnp.abs(alpha) ** 2
            )

        def _coherent(alpha, N):
            sqrtn = jnp.sqrt(jnp.arange(0, N, dtype=jnp.complex128))
            sqrtn = sqrtn.at[0].set(1)  # Get rid of divide by zero warning
            data = alpha / sqrtn

            data = data.at[0].set(jnp.exp(-(abs(alpha) ** 2) / 2.0))

            state = jnp.cumprod(data)

            return jnp.reshape(state, (N, 1))

        def core(alpha_multi):
            alpha_state_list = jax.vmap(_coherent, in_axes=(0, None))(
                alpha_multi, self.N_single
            )

            final_state = alpha_state_list[0]
            for i in range(1, self.num_modes):
                final_state = jnp.kron(final_state, alpha_state_list[i])

            _oper = final_state @ final_state.conj().T
            # 矢量化 _oper,vec( (|alpha><alpha|).T)
            oper_vec = _oper.T.flatten(order="F")
            return oper_vec

        return jax.vmap(core)(alpha_vec)


@deprecated(reason="Use Alpha2RowMultiModeNoNoise instead", version="0.1.0")
def alpha2row_multimode_nonoise(alpha_vec, N_single=5, num_modes=2):
    def _coherent2(alpha, N):
        n_array = jnp.arange(0, N, dtype=jnp.float64)
        sqrtn = jnp.sqrt(jax.vmap(jax.scipy.special.factorial)(n_array))

        alpha_n_array = jnp.power(alpha, n_array) / sqrtn

        return jnp.reshape(alpha_n_array, (N, 1)) * jnp.exp(-0.5 * jnp.abs(alpha) ** 2)

    @jax.jit
    def core(alpha_multi):
        alpha_state_list = jax.vmap(_coherent2, in_axes=(0, None))(
            alpha_multi, N_single
        )

        final_state = alpha_state_list[0]
        for i in range(1, num_modes):
            final_state = jnp.kron(final_state, alpha_state_list[i])

        _oper = final_state @ final_state.conj().T
        # 矢量化 _oper,vec( (|alpha><alpha|).T)
        oper_vec = _oper.T.flatten(order="F")
        return oper_vec

    return jax.vmap(core)(alpha_vec)


# 尝试写wigner函数的
# 单模版
class Alpha2RowWigner(eqx.Module):  # 定义一个类
    rho_h: jnp.ndarray
    N: int = eqx.field(static=True)
    displacer: eqx.Module
    N_compute: int | None = eqx.field(static=True)

    def __init__(self, rho_h=None, N=5, N_compute=None):  # 这个是类的初始化函数
        self.N = N
        self.N_compute = N_compute if N_compute is not None else N
        if rho_h is None:
            rho_h = dq.basis_dm(N_compute, 0).to_jax()  # 生成N维的0态的密度矩阵
        self.rho_h = jnp.array(rho_h)  # 转换变量为jnp.array

        self.displacer = Displacer(N_compute)  # 生成位移算符，维度为N_compute

    # 将主要的计算逻辑 jit 编译
    @jax.jit
    def __call__(self, alpha_vec):
        def core(alpha):
            Da = self.displacer.displace(
                alpha
            )  # 调用了前面Displacer类中的displace函数，即生成了一个前面所说的位移算符
            Parity = jnp.diag(jnp.exp(1j * jnp.pi * jnp.arange(self.N_compute)))
            _oper = (Da @ Parity @ Da.conj().T)[: self.N, : self.N]  # 截断到N维

            # 矢量化 _oper
            oper_vec = _oper.T.flatten(order="F")  # 二维到一维
            return oper_vec

        return jax.vmap(core)(alpha_vec)


# 这个类中，N是截断维度，N_compute是计算维度，输出的结果为噪声情况情况下的测量算符，即13号文章的20式


# 多模版
class Alpha2RowMultiModeWigner(eqx.Module):  # 多模的A
    rho_h: jnp.ndarray | List = eqx.field(static=True)
    N_single: int = eqx.field(static=True)
    num_modes: int = eqx.field(static=True)
    N_compute: int | None = eqx.field(static=True)
    displacer: eqx.Module
    parity: jnp.ndarray = eqx.field(static=True)

    def __init__(self, rho_h=None, N_single=5, num_modes=2, N_compute=None):
        self.N_single = N_single
        self.N_compute = N_compute if N_compute is not None else N_single
        self.num_modes = num_modes
        if rho_h is None:
            rho_h = [dq.basis_dm(N_compute, 0).to_jax()]  # 生成N维的0态的密度矩阵
            for _ in range(num_modes - 1):  # 从0开始，所以-1
                rho_h.append(dq.basis_dm(N_compute, 0).to_jax())
        if isinstance(rho_h, List):  # 判断是否是列表
            self.rho_h = rho_h
        else:
            self.rho_h = [rho_h]
        # 将num_modes - 1个密度矩阵一个个贴起来，并把rho_h转换为列表形式。

        # self.rho_h = jnp.array(rho_h)

        self.displacer = Displacer(N_compute)  # 生成位移算符，维度为N_compute
        self.parity = jnp.diag(jnp.exp(1j * jnp.pi * jnp.arange(self.N_compute)))

    # @jax.jit
    def __call__(self, alpha_vec):
        def core(alpha_multi):
            Da = [self.displacer.displace(alpha_single) for alpha_single in alpha_multi]
            # 调用了前面Displacer类中的displace函数，即生成了一个前面所说的位移算符

            Parity = self.parity

            # Da_pi = Da[0]
            res = (Da[0] @ Parity @ Da[0].conj().T)[
                : self.N_single, : self.N_single
            ]  # 可以试试把rho_h换为Parity函数  # 这个似乎是第0个密度矩阵？求第0个的投影算符
            for i, ds in enumerate(Da[1:]):  # 从第i个开始到最后一个
                # Da_pi = jnp.kron(Da_pi, ds)
                res = jnp.kron(
                    res,
                    (ds @ Parity @ ds.conj().T)[: self.N_single, : self.N_single],
                )  # 每一个做截断，然后乘出投影算符，直积起来

            # _oper = Da_pi @ self.rho_h @ Da_pi.conj().T
            _oper = res

            # 矢量化 _oper
            oper_vec = jnp.array(_oper.T.flatten(order="F"))  # 矢量化
            return oper_vec * (2 / jnp.pi) ** self.num_modes

        return jax.vmap(core)(alpha_vec)

