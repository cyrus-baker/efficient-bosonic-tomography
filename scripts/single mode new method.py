import jax

from qutip import coherent, basis, fidelity, plot_fock_distribution, thermal_dm, Qobj  # noqa: F401

# from qutip.qobj import Qobj
import numpy as np
import jax.numpy as jnp
import time
import pandas as pd

import cvxpy as cp


from efficient_bosonic_tomography.generate_data import generate_qfunc
from efficient_bosonic_tomography.displacer import Alpha2Row
from math import ceil

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
# enable compilation cache
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")


if __name__ == "__main__":
    num_iterations = 10  # 循环次数
    z0 = np.zeros(num_iterations)
    z1 = np.zeros(num_iterations)
    z2 = np.zeros(num_iterations)
    z3 = np.zeros(num_iterations)
    z4 = np.zeros(num_iterations)
    z5 = np.zeros(num_iterations)

    for p in range(num_iterations):
        # 把要迭代的变量用p表达即可
        start_time1 = time.time()
        # 参数输入与初态
        grid = 20  # 网格数目
        limit = 4  # phase space limit
        N = 10 + p  # N的值
        compute = 1.5
        N = int(N)

        xlim = [-limit, limit]
        ylim = [-limit, limit]
        x_points = grid
        y_points = grid

        # state = (coherent(N, 2) + 1j * coherent(N, -2)).unit()
        # state = (basis(N, 0) + basis(N, 4) + basis(N, 8)).unit()

        state = (coherent(N, 2) + coherent(N, -2)).unit()  # 初始态，一个两脚猫

        # state = (coherent(N, 2) + coherent(N, -2) + coherent(N, 2j) + coherent(N, -2j)).unit()  # 四脚猫

        # state = (coherent(N, 2) + coherent(N, -2) + coherent(N, 2j) + coherent(N, -2j)
        #  + coherent(N, 2 * np.exp(1j * 0.25 * np.pi)) + coherent(N, 2 * np.exp(1j * 0.75 * np.pi))
        #  + coherent(N, 2 * np.exp(1j * 1.25 * np.pi)) + coherent(N, 2 * np.exp(1j * 1.75 * np.pi))).unit()  # 八脚猫

        # state = 0.3 * thermal_dm(N, 5) + 0.7 * state * state.dag()

        _start = time.time()
        x_vec, y_vec, q_values = generate_qfunc(state, xlim, ylim, x_points, y_points)
        elapsed_time2 = time.time() - _start
        # print("generate qfunc time:", elapsed_time2)

        # plt.imshow(q_values, extent=[*xlim, *ylim])

        mesh_x, mesh_y = np.meshgrid(x_vec, y_vec)

        # combile mesh_x, and y
        mesh = np.vstack((mesh_x.reshape(-1), mesh_y.reshape(-1))).T
        q_values = q_values.reshape(-1)

        alpha0 = complex(mesh[0][0], mesh[0][1])
        alpha_cv_list = jnp.array([complex(*a) for a in mesh])

        # 矩阵A的时间
        _start = time.time()
        A_gen = Alpha2Row(N=N, N_compute=ceil(N * compute))
        A = A_gen(alpha_cv_list) * 1 / np.pi

        b = jnp.array(q_values)
        elapsed_time3 = time.time() - _start
        # print("alpha2row time:", elapsed_time3)

        _start = time.time()
        # 求解cvx问题
        # Define and solve the CVXPY problem.
        A_param = cp.Parameter((len(b), N * N), value=np.array(A), complex=True)
        X = cp.Variable((N, N), hermitian=True)
        print(X.value)
        t = cp.Variable((1,))

        constraints = [
            X >> 0,
            cp.trace(X) == 1,
            cp.norm2(A_param @ cp.vec(X, order="F") - b) <= t,
        ]

        prob = cp.Problem(cp.Minimize(t), constraints)

        prob.solve(
            solver=cp.SCS,
            # mkl=True,
            verbose=True,
        )
        elapsed_time4 = time.time() - _start
        # print("solve time:", elapsed_time4)

        print("status:", prob.status)
        print("optimal value", prob.value)
        # print("optimal var", X.value)

        # 计算保真度
        rho_reconstruct = Qobj(X.value).unit()
        rho_reconstruct = rho_reconstruct / rho_reconstruct.tr()
        z0[p] = fidelity(rho_reconstruct, state)
        print("fidelity:", z0[p])

        # 重构的state的Q函数
        # qfunc of reconstructed state
        # x_vec, y_vec, q_values = generate_qfunc(rho_reconstruct, xlim, ylim, x_points, y_points)
        # plt.imshow(q_values, extent=[*xlim, *ylim])

        # plot_fock_distribution(rho_reconstruct)
        # plot_fock_distribution(state)
        end_time1 = time.time()

        end_time1 = time.time()  # 记录结束时间
        elapsed_time1 = end_time1 - start_time1  # 计算总用时
        # elapsed_time2 = end_time2 - start_time2  # 计算生成Q函数用时
        # elapsed_time3 = end_time3 - start_time3  # 计算矩阵A用时
        # elapsed_time4 = end_time4 - start_time4  # 计算cvx求解用时
        z1[p] = elapsed_time1  # 计算总用时
        z2[p] = elapsed_time2  # 计算生成Q函数用时
        z3[p] = elapsed_time3  # 计算矩阵A用时
        z4[p] = elapsed_time4  # 计算cvx求解用时
        # elapsed_time5 = elapsed_time3 - elapsed_time4
        # z5[p] = elapsed_time5
        print("迭代的次数 =", p + 1)
        print("N =", N)
        # print("phase space limit =",limit)
        print(f"总用时 Iteration {p + 1}: {elapsed_time1:.4f} seconds")
        print(f"生成Q函数时间 Iteration {p + 1}: {elapsed_time2:.4f} seconds")
        print(f"矩阵A用时 Iteration {p + 1}: {elapsed_time3:.4f} seconds")
        print(f"cvx求解用时 Iteration {p + 1}: {elapsed_time4:.4f} seconds")
        # print(f"总迭代时间减去构建b的时间 Iteration {p + 1}: {elapsed_time5:.4f} seconds")
        print("保真度", p + 1, z0[p])
        z5[p] = limit  # 用于输出自变量

    # 数据存储
    # 创建数据
    data = [z5, z0, z1, z2, z3, z4]

    # 转换为 DataFrame
    df = pd.DataFrame(data)

    # 将数据写入 Excel 文件
    df.to_excel("time_qst_convex_conic_constraint.xlsx", index=False)

    print("数据已成功写入 time_qst_convex_conic_constraint.xlsx")
