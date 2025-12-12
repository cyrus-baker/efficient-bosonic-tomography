import jax


from qutip import coherent, basis, fidelity, plot_fock_distribution, thermal_dm, tensor,Qobj  # noqa: F401
# from qutip.qobj import Qobj
import numpy as np
import jax.numpy as jnp

import time

import wandb


from efficient_bosonic_tomography.displacer import Alpha2RowMultiModeWigner#2/pi移到Alpha2RowMultiModeWigner

# from statsmodels.nonparametric.kernel_density import KDEMultivariate
import dynamiqs as dq

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")
# enable compilation cache
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

def project_on_simplex(rho):
    from optax.projections import projection_simplex

    # 特征值分解
    w, v = jnp.linalg.eigh(rho)

    # 将特征值投影到概率单纯形上
    w_proj = projection_simplex(w)

    # 重构矩阵
    rho_proj = v @ jnp.diag(w_proj) @ v.T.conj()
    return rho_proj


def original_prox(rho, eta, updates):
    temp = rho - eta * jnp.conj(updates)
    return project_on_simplex(temp)


# 定义损失函数
# @jax.jit
def loss_func(rho: jnp.ndarray, A: jnp.ndarray, b: jnp.ndarray):
    rho_vec = rho.flatten(order="F").reshape(-1, 1)
    meas = A @ rho_vec
    loss = jnp.abs(meas - b) ** 2

    return jnp.mean(loss)

if __name__ == "__main__":
    for p in range(1):
        config = {
        "optimizer": "proximal gradient descent",
        "num_modes": 2,
        "N_single": 6   ,
        "T": 1,
        "BATCH_SIZE": 3000, 
        "eta_start": 0.5,
        "method": 2,
        }

        # optimizer = "proximal gradient descent"
        # num_modes = 2
        # N_single =  5
        # T = 200
        # BATCH_SIZE = 256
        # eta_start = 1e3
        # method = 2

        run = wandb.init(project="qst-scp-Wigner-two-mode", config=config)

        N_single = config["N_single"]
        num_modes = config["num_modes"]
        T = config["T"]
        eta_start = config["eta_start"]
        BATCH_SIZE = config["BATCH_SIZE"]

        # state

        state1 = (basis(N_single , 0) + basis(N_single, 4)).unit()
        state2 = basis(N_single , 2)
        state = tensor(state1,state2) + tensor(state2,state1) #binomial code
        state = state.unit()
        state = state * state.dag()  
        state = jnp.array(state.full())


        # randomly sample n points from the data without using histo


        alpha_values = np.random.uniform(-2, 2, (int(1e6), 4))

        alpha_cv_list = [
            alpha_values[:, 0] + 1j * alpha_values[:, 1],
            alpha_values[:, 2] + 1j * alpha_values[:, 3],

        ]
        alpha_cv_list = np.array(alpha_cv_list).T

        _start = time.time()
        A_list = []
        b_list = []
        A_gen = Alpha2RowMultiModeWigner(None, N_single, num_modes, N_compute=24)
        # split the vmap use to reduce memory usage

        N = N_single ** num_modes
        rand_matrix = np.random.normal(size=(N, N)) + 1j * np.random.normal(size=(N, N))
        rho = jax.scipy.linalg.expm(rand_matrix + rand_matrix.T.conj())
        rho = rho / jnp.real(jnp.trace(rho))


        x_km1 = rho
        x_km2 = rho

        _start = time.time()
        for k in range(400):
            # 逐步缩小步长
            eta = eta_start / (1 + k) ** 0.5
            x_tau_k_list = []
            v_0 = None

            loss_values = 0
            grads = 0

            v = x_km1 + (k - 2) / (k + 1) * (x_km1 - x_km2) if k > 3 else x_km1
            v_0 = v


            for tau in range(T):
                
                temp_list = alpha_cv_list[BATCH_SIZE * tau : BATCH_SIZE * (tau + 1)]

                A = A_gen(temp_list)
                b = (A @ jnp.reshape(state, (-1, 1), order='F'))


                loss, single_grad = jax.value_and_grad(loss_func)(v, A, b)  # 计算梯度
                # loss, single_grad = jax.value_and_grad(loss_exact)(v, A, b)

                loss_values += loss
                grads += single_grad

                # method 1
                if config["method"] == 1:
                # if method == 1:
                    x_tau_k = original_prox(v, eta, single_grad)
                    x_tau_k_list.append(x_tau_k)

                # method 2, ASSG-r method
                if config["method"] == 2:
                # if method == 2:
                    v = (1 - 2 / (tau + 1)) * v + 2 / (tau + 1) * v_0
                    v = original_prox(v, eta, single_grad)
                    x_tau_k_list.append(v)

            # 执行 proximal gradient descent
            loss_values /= T
            grads /= T


            if config["method"] == 1: #随机凸优化
            # if method == 1:
                x_tau_k = jnp.mean(jnp.array(x_tau_k_list), axis=0)

            if config["method"] == 2: #ASSG-r加速
            # if method == 2:
                x_tau_k = jnp.mean(jnp.array(x_tau_k_list), axis=0)

            # method 4, backtracking line search
            # t = backtracking_line_search(lambda x: loss_exact(x, A, b), v, grads, t0=1e-1, beta=0.7)
            # x_tau_k = original_prox(v, t, grads)

            # x_km1 = 1 / T * jnp.sum(jnp.array(x_tau_k_list), axis=0)
            x_km2 = x_km1
            x_km1 = x_tau_k

            # loss
            print(f"Epoch {k}, loss: {loss_values}")
            print("fidelity:", dq.fidelity(dq.unit(x_km1), state))

            # stat
            run.log(
                    {
                        "loss": loss_values,
                        # "fidelity": dq.fidelity(dq.unit(v), target_state),
                        "fidelity": dq.fidelity(dq.unit(x_km1), state),
                        "time_elapsed": time.time() - _start,


                    }
                )



        



    # run.finish()

    # rho_reconstruct = Qobj(
    #     np.array(v), dims=[[N_single, N_single], [N_single, N_single]]
    # ).unit()
    # rho_reconstruct = rho_reconstruct / rho_reconstruct.tr()
    # print("fidelity:", fidelity(rho_reconstruct, state))

    # # qfunc of reconstructed state
    # plot_fock_distribution(rho_reconstruct)
    # plot_fock_distribution(state)


#%% 