import jax


from qutip import coherent, basis, fidelity, plot_fock_distribution, thermal_dm, tensor,Qobj  # noqa: F401
# from qutip.qobj import Qobj
import numpy as np
import jax.numpy as jnp

import time

import wandb

import pandas as pd
from efficient_bosonic_tomography.displacer import Alpha2RowMultiModeWigner


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

    num_iterations = 1 # 循环次数
    reaserch_iterations = 1 #重复测量次数
    
    z0 = np.zeros((num_iterations,reaserch_iterations))
    z1 = np.zeros((num_iterations,reaserch_iterations))
    z2 = np.zeros((num_iterations,reaserch_iterations))
    z3 = np.zeros((num_iterations,reaserch_iterations))
    z4 = np.zeros((num_iterations,reaserch_iterations)) 
    z5 = np.zeros((num_iterations,reaserch_iterations))
    z6 = np.zeros((num_iterations,reaserch_iterations))




    
    for q in range(reaserch_iterations):
        for p in range(num_iterations):
            config = {
            "optimizer": "proximal gradient descent",
            "num_modes": 2,
            "N_single": 6    ,
            "T": 1,
            "BATCH_SIZE": 3000, 
            "eta_start": 5e-1,
            "method": 2,
            "noise_level": 0.0 + 0.00 * p,
            "steps" : 2000,
            "steps_judge" : 100,
            "fid_judge" : 0.6,
            "important_sample":0
            }


            # optimizer = "proximal gradient descent"
            # num_modes = 2
            # N_single =  6
            # T = 1
            # BATCH_SIZE = 3000
            # eta_start = 5e-1 #无噪声取大一点，有噪声取小一点
            # method = 2
            # noise_level = 0.1 
            # steps = 2000
            # steps_judge = 100
            # fid_judge = 0.6

            run = wandb.init(project="qst-scp-Wigner-two-mode", config=config)

            N_single = config["N_single"]
            num_modes = config["num_modes"]
            T = config["T"]
            eta_start = config["eta_start"]
            BATCH_SIZE = config["BATCH_SIZE"]
            noise_level = config["noise_level"]
            method      = config["method"]
            steps       = config["steps"]
            steps_judge = config["steps_judge"]
            fid_judge   = config["fid_judge"]
            important_sample = config["important_sample"]
            





            key = jax.random.PRNGKey(0)  # 随机数种子，可改成动态的

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

            # 概率分布



            _start = time.time()
            loss_best = jnp.inf
            loss_steps = 0
            fidelity_best = 0
            fid_flag = 0

            b_list = []
            for tau in range(int(len(alpha_cv_list) // BATCH_SIZE + 1)):
                    
                temp_list = alpha_cv_list[BATCH_SIZE * tau : BATCH_SIZE * (tau + 1)]
                
                A = A_gen(temp_list)
                b = (A @ jnp.reshape(state, (-1, 1), order='F'))
                b_list.append(b.flatten())

            b_list = jnp.concatenate(b_list).flatten().real
            
        






            for k in range(steps): #最好400,100是预实验
                # 逐步缩小步长
                eta = eta_start / (1 + k) ** 0.5
                x_tau_k_list = []
                v_0 = None

                loss_values = 0
                grads = 0

                v = x_km1 + (k - 2) / (k + 1) * (x_km1 - x_km2) if k > 3 else x_km1
                v_0 = v
                



                for tau in range(T):
                    
                    if important_sample == 1:
                        # 重要性采样
                        key,subkey = jax.random.split(key)
                        temp_list = jax.random.choice(
                            subkey,
                            a=alpha_cv_list,
                            #从[0，n)这些下标中选
                            shape=(BATCH_SIZE,),
                            p=b_list/b_list.sum(),
                            replace=True)
                    else:
                        temp_list = alpha_cv_list[BATCH_SIZE * tau : BATCH_SIZE * (tau + 1)]



                    A = A_gen(temp_list)
                    b = (A @ jnp.reshape(state, (-1, 1), order='F'))
                    # noise = novel_level * jax.random.normal(key, shape=b.shape)
                    noise = np.random.normal(0, noise_level,size=(BATCH_SIZE, 1))
                    b = b + noise




                    loss, single_grad = jax.value_and_grad(loss_func)(v, A, b)  # 计算梯度
                    # loss, single_grad = jax.value_and_grad(loss_exact)(v, A, b)

                    loss_values += loss
                    grads += single_grad

                    # method 1
                    # if config["method"] == 1:
                    if method == 1:
                        x_tau_k = original_prox(v, eta, single_grad)
                        x_tau_k_list.append(x_tau_k)

                    # method 2, ASSG-r method
                    # if config["method"] == 2:
                    if method == 2:
                        v = (1 - 2 / (tau + 1)) * v + 2 / (tau + 1) * v_0
                        v = original_prox(v, eta, single_grad)
                        x_tau_k_list.append(v)

                # 执行 proximal gradient descent
                loss_values /= T
                grads /= T






                # if config["method"] == 1: #随机凸优化
                if method == 1:
                    x_tau_k = jnp.mean(jnp.array(x_tau_k_list), axis=0)

                # if config["method"] == 2: #ASSG-r加速
                if method == 2:
                    x_tau_k = jnp.mean(jnp.array(x_tau_k_list), axis=0)

                # method 4, backtracking line search
                # t = backtracking_line_search(lambda x: loss_exact(x, A, b), v, grads, t0=1e-1, beta=0.7)
                # x_tau_k = original_prox(v, t, grads)

                # x_km1 = 1 / T * jnp.sum(jnp.array(x_tau_k_list), axis=0)
                x_km2 = x_km1
                x_km1 = x_tau_k



                fid= dq.fidelity(dq.unit(x_km1), state)
                if fid > fidelity_best:
                    fidelity_best = fid
                else:
                    fid_flag = 1



                # flag程序，这里我们暂时不用flag
                # if fid_flag ==1:
                #     if loss_values < loss_best:
                #         loss_best = loss_values
                #         loss_steps = 0
                #     else:
                #         loss_steps += 1

                #     if loss_steps > steps_judge and dq.fidelity(dq.unit(x_km1), state) > fid_judge:
                #         print("(loss unchange)Early stopping at epoch", k)
                #         print("time", time.time() - _start)
                #         break

                # if fid > 0.999: #
                #     print("(fid > 0.999)Early stopping at epoch", k)
                #     print("time", time.time() - _start)
                #     break






                # loss
                print(f"Epoch {k}, loss: {loss_values}")
                print("fidelity:", fid)
                time_all = time.time() - _start
                print("time:",  time_all)

                

                # stat
                run.log(
                        {
                            "loss": loss_values,
                            # "fidelity": dq.fidelity(dq.unit(v), target_state),
                            "fidelity": dq.fidelity(dq.unit(x_km1), state),
                            "time_elapsed": time.time() - _start,


                        }
                    )

                z0[p,q] = fid
                z1[p,q] = time_all  # 计算总用时
                z2[p,q] = k  # 计算采样用时
                z3[p,q] = loss_values  # 计算生成W函数用时

                # print("phase space limit =",limit)
                z4[p,q] = N_single  #用于输出自变量
                z5[p,q] = noise_level  #用于输出自变量



    z0_average = np.mean(z0, axis=1)
    z0_std = np.std(z0, axis=1)
    z1_average = np.mean(z1, axis=1)
    z1_std = np.std(z1, axis=1)
    z2_average = np.mean(z2, axis=1)
    z2_std = np.std(z2, axis=1)
    z3_average = np.mean(z3, axis=1)
    z3_std = np.std(z3, axis=1)
    z4_average = np.mean(z4, axis=1)
    z4_std = np.std(z4, axis=1)
    z5_average = np.mean(z5, axis=1)
    z5_std = np.std(z5, axis=1)



    # 数据存储
    # 创建数据
    z = np.zeros(num_iterations)
    data = [z4,z5,z0,z1,z2,z3]
    data = [z4_average,z5_average,z0_average,z1_average,z2_average,z3_average,z4_std,z5_std,z0_std,z1_std,z2_std,z3_std,]
    # data_average = [z5_average,z0_average,z1_average,z2_average,z3_average,z4_average]
    # data_std = [z5_std,z0_std,z1_std,z2_std,z3_std,z4_std]

    # 转换为 DataFrame
    df = pd.DataFrame(data)

    # 将数据写入 Excel 文件
    df.to_csv('wigner_proximal_gradient_descent_two_mode_without_optimazation.csv', index=False)

    print("数据已成功写入 wigner_proximal_gradient_descent_two_mode_without_optimazation.xlsx")

        



    # run.finish()

    # rho_reconstruct = Qobj(
    #     np.array(v), dims=[[N_single, N_single], [N_single, N_single]]
    # ).unit()
    # rho_reconstruct = rho_reconstruct / rho_reconstruct.tr()
    # print("fidelity:", fidelity(rho_reconstruct, state))

    # qfunc of reconstructed state
    # plot_fock_distribution(rho_reconstruct)
    # plot_fock_distribution(state)


#%% 