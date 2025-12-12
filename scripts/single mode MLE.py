import numpy as np
import time 
import pandas as pd
from qutip import coherent, ket2dm, Qobj, expect, fidelity, fock, fock_dm, displace, coherent_dm, qeye, squeeze  # noqa: F401

from qutip.wigner import qfunc

from tqdm.auto import tqdm # progress bar

from joblib import Parallel, delayed

# 参数定义：
num_iterations = 1 # 循环次数
z0 = np.zeros(num_iterations)
z1 = np.zeros(num_iterations)
z2 = np.zeros(num_iterations)
z3 = np.zeros(num_iterations)
z4 = np.zeros(num_iterations)
z5 = np.zeros(num_iterations)
z6 = np.zeros(num_iterations)



for p in range(num_iterations):
    start_time1 = time.time()  # 记录开始时间

    # 定义q函数的测量算符并做截断（其实就是一个相干态的密度矩阵）
    def qfunc_ops(hilbert_size, beta):
        
        # need the larger Hilbert space to not get erroneous operators for large beta
        op = coherent_dm(hilbert_size * 10, beta)/np.pi
        op = Qobj(op[:hilbert_size, :hilbert_size]) # truncate to wanted Hilbert space size
            
        return op
    # q函数的原因？应该不是，之后的betamax也不超过3 + 4j，


    # 定义state
    # Fock space dimension
    N = 15  # 待改变的那个N
    nbins = 20   # 网格数
    lim = 4   # 应该是phase space limit

    # fock空间的大小

    # different test states

    # Binomial，就两个fock态
    # state = (fock(N,0) + fock(N,4)).unit()

    # 六个fock态
    # state = (fock(N,0) + fock(N,4) + fock(N,2) + fock(N,3) + fock(N,1) + fock(N,5)).unit()

    # Squeezed vacuum 压缩真空态？
    # state = squeeze(N,1)*fock(N,0) # squeeze是压缩态的算符，即N维，压缩度为1

    #相干态
    # state = coherent(N,2) .unit()

    # 猫态
    # Coherent state amplitudes for test cat state
    alpha_range = 2    
    alphas = np.array([alpha_range, -alpha_range])                  
    # Test-state 
    psi = sum([coherent(N, a) for a in alphas])
    state = psi.unit() #归一化
    # rho = ket2dm(psi) #将态矢量转换为密度矩阵
    # 确实是猫态，没问题


    # calculate Q-function
    lim=lim # 应该是phase space limit
    # nsteps = 500
    # xvec0 = np.linspace(-lim,lim,nsteps)
    # yvec0=np.linspace(-lim,lim,nsteps)

    # Q = qfunc(state, xvec0, yvec0, g=2) # 在上述state，上述网格中计算Q函数



    # 从Q函数中取样
    # number of samples
    # nr_samples = 50000 #试着改成100000，有点用。又试着改到1000000，趋于0.67

    # # number of histogram bins 
    # nbins = nbins  #网格数

    # # Create a flat copy of the array, normalize in order to use numpy.random.choice
    # Q_flat = Q.flatten() / np.sum(Q.flatten())
    # # 一维化Q函数并做归一化

    # start_time4 = time.time()
    # def get_sample(random_state):
    #     # generate "determinisic" random state for reproducibility
    #     rng = np.random.RandomState(random_state) # 这里的random_state是种子，确保生成的随机数是恒定的
        
    #     # sample an index from the 1D array with the
    #     # probability distribution from the original array  
    #     sample_index = rng.choice(a=Q_flat.size, p=Q_flat)
    #     # a为选取数目的上限（或者说数的上限），p为选取的概率。就是以p为概率在a中选了一个

    #     adjusted_index = np.unravel_index(sample_index, Q.shape)
    #     #输出原数组元素坐标对应的转换数组坐标，sample_index是原数组，而Q.shape是数组的形状
    #     #第一个构成index（前一步已随机），第二个就是根据这些index去找原来Q中index的值
    #     #最后的输出就是之前随机数选样的坐标

    #     # x and y-coordinates
    #     idx = adjusted_index[0]
    #     idy = adjusted_index[1]
    #     # 第一列是x，第二列是y，即第一个index与第二个index

    #     # scale to phase space coordinates instead of indices
    #     X = -lim + idx*lim/(nsteps/2)
    #     Y = -lim + idy*lim/(nsteps/2)
    #     # 坐标转换，把index转换为相空间中的具体值

    #     return X,Y  # 输出的是一个两列list，第一列是X，第二列是Y

    # # 上面那个函数是一个取样函数，为啥一个简简单单的取样函数要搞得如此复杂………………

    # # generate "deterministic" seeds for reproducibility
    # np.random.seed(100) # 随机数种子
    # random_seeds = np.random.randint(1000000, size=nr_samples) #指定范围内的随机数，1000000是上限，size是形状


    # # samples = Parallel(n_jobs=-2,verbose=5,backend="multiprocessing")(delayed(get_sample)(rng) for rng in random_seeds)
    # # 采了一大堆样，但目前还不知道是怎么采的
    # # 每一个random_seeds里面的数都在get_sample函数里面跑一遍，然后
    # # 先用一个种子生成一堆随机数，这一堆随机数再每一个都get一个sample，总数量就是前面输的那个值
    # # random_seeds是种子（0-1000000），有nr_samples个种子，
    
    # end_time4 = time.time()

    # X = np.array(samples)[:,0]
    # Y = np.array(samples)[:,1]

    # # 总的来说，以上部分就是在Q的相空间采了个样，采样的数量是nr_samples，且采样的结果是一个XY坐标构成的两列数组。

    # # bin and create histogram
    # sampled_Q, yedges, xedges = np.histogram2d(X,Y,bins=nbins, density=True) # 完全不知为何但正确的一步，就非常的神奇？？


    xvec1 = np.linspace(-lim,lim,nbins + 1)
    yvec1 = np.linspace(-lim,lim,nbins + 1)

    xvec = []
    for idx, x in enumerate(xvec1[:-1]):
        xvec.append((x + xvec1[idx+1])/2)

    yvec = []
    for idx, y in enumerate(yvec1[:-1]):
        yvec.append((y + yvec1[idx+1])/2)
    #用理论值
    sampled_Q = qfunc(state, xvec, yvec, g=2)



    sampled_Q = sampled_Q/np.sum(sampled_Q)
    # sampled_Q = sampled_Q/np.sum(sampled_Q) 试了试，发现不归一，但归一后的值也没有影响
    # 二维直方图，X，Y分别对应两个坐标，bins是区块数，density表归一
    # sample_Q是这些网格中的值，后两个是x，y的边界值

    # xvec1 = xedges
    # yvec1 = yedges

    # Use midpoint of grid as phase space points
    xvec = []
    for idx, x in enumerate(xvec1[:-1]):# 抛开了最后一个元素，idx是索引，x是值
        xvec.append((x + xvec1[idx+1])/2)
    # idx是索引，x是这个索引中的值，而这里取了均值这个点与下一个点的均值做为xvec的值，即网格前后两个值的中心值。

    yvec = []
    for idx, y in enumerate(yvec1[:-1]):
        yvec.append((y + yvec1[idx+1])/2)



    # 网格构建，用前面取样的xvec，yvec（就那个划分网格后的中心值，数目和nbins一样）
    X, Y = np.meshgrid(xvec, yvec)

    # flatten the grid into a 1D array
    betas = (X + 1j*Y).ravel()
    # 变成一个一维array

    start_time3 = time.time()
    # 构建相干态的密度矩阵，就是文中的\Pi_k
    Pis = Parallel(n_jobs=16, verbose=5)(delayed(qfunc_ops)(N,beta) for beta in betas)
    end_time3 = time.time()





    # MLE操作
    # 其中提到的式子以04年那篇mle为例
    fidelities = []
    max_iterations = 20000  # 最大迭代数

    # variable names were different in this code
    b = sampled_Q.flatten()  # 采样值
    data = b   # 采样值
    hilbert_size = N   # 希尔伯特空间维度数目
    rho_true = state   # 密度矩阵
    m_ops = Pis   # 

    ops_numpy = [op.full() for op in m_ops] # 把Pis算符全部转化为numpy的数据类型，并拼起来？

    # rho = qeye(hilbert_size)/hilbert_size  # 迭代的初态是一个最大混合态
    rho = state    
    rho = ket2dm(rho)
    # 使用原态去test一下

    fidelities.append(fidelity(rho_true, rho))
    pbar = tqdm(range(max_iterations))


    start_time2 = time.time()

    for i in range(max_iterations):
        guessed_val = expect(m_ops, rho)  #即单个似然pr，式6
        ratio = data / guessed_val  #采样值（猜测是f）/pr
        rho_old = rho   #更新密度矩阵

        R = Qobj(np.einsum("aij,a->ij", ops_numpy, ratio))
        # 求和，对ops_numpy中的每个矩阵以ratio为权重进行求和，就是式2

        rho = R * rho * R
        rho = rho / rho.tr() #归一化常数
        # 这两步是迭代

        f = fidelity(rho, rho_true) #这一次的f
        fidelities.append(f) # 一大串f，可能用于看迭代的收敛速度
        

        diff_rho = rho-rho_old
        # diff_norm = np.sqrt(np.trace(diff_rho * diff_rho.dag())).real

        diff = diff_rho * diff_rho.dag()
        # diff_norm = np.sqrt(np.trace(diff.full())).real  #这一步有改动
        diff_norm = np.sqrt(np.trace((diff_rho*diff_rho.dag()).full()).real)  #原始版
        # diff_norm = np.sqrt(np.trace(diff_rho*diff_rho.dag())).real   #原始版
        z6[p] = i #用于记录步长
        if diff_norm < 1e-6: #原始是1e-6
                break
        # 计算步长，步长小于一定的值就停止进行迭代

        pbar.set_description("Fidelity iMLE {:.4f}, diff norm {:.2e}".format(f, diff_norm))
        pbar.update()

        # # 更新进度条的
    end_time2 = time.time()
    
    # 保真度
    z0[p] = f



    # 数据输入
    end_time1 = time.time()  # 记录结束时间
    elapsed_time1 = end_time1 - start_time1  # 计算总用时
    elapsed_time2 = end_time2 - start_time2  # 计算MLE迭代用时
    elapsed_time3 = end_time3 - start_time3  # 计算\Pi_k用时
    # elapsed_time4 = end_time4 - start_time4  # 计算采样用时
    z1[p] = elapsed_time1
    z2[p] = elapsed_time2
    z3[p] = elapsed_time3
    # z4[p] = elapsed_time4
    # elapsed_time5 = elapsed_time3 - elapsed_time4
    # z5[p] = elapsed_time5
    # 自变量
    # z5[p] = N
    # print("迭代的N=",N )

    # # 自变量
    z5[p] = lim
    print("相空间极限=",lim )

    print(f"总用时 Iteration {p + 1}: {elapsed_time1:.4f} seconds")
    print(f"MLE时间 Iteration {p + 1}: {elapsed_time2:.4f} seconds")
    print(f"\Pi_k用时 Iteration {p + 1}: {elapsed_time3:.4f} seconds")
    # print(f"采样用时 Iteration {p + 1}: {elapsed_time4:.4f} seconds")
    # print(f"总迭代时间减去构建b的时间 Iteration {p + 1}: {elapsed_time5:.4f} seconds")
    print("保真度",p+1,z0[p])
    print("迭代次数",p+1,z6[p])
    


#数据写出
# 创建数据
data = [z5,z0,z1,z2,z3,z4,z6]

# 转换为 DataFrame
df = pd.DataFrame(data)

# 将数据写入 Excel 文件
df.to_excel('time_MLE.xlsx', index=False)

print("数据已成功写入 time_MLE.xlsx")
