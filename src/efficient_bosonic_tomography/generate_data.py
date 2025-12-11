import numpy as np
from qutip import qfunc, Qobj, wigner
import scipy
import scipy.signal
from numpyro.distributions import constraints
import numpyro as npy
import jax
import equinox as eqx
from typing import List

import jax.numpy as jnp
import dynamiqs as dq
from functools import partial
from displacer import Displacer

def generate_qfunc(state: Qobj, xlim, y_lim, x_points, y_points, noise_level=None):
    x_vec = np.linspace(*xlim, x_points)
    y_vec = np.linspace(*y_lim, y_points)

    q_values = qfunc(state, x_vec, y_vec, g=2)

    if noise_level == 0 or noise_level is None:
        d_values = q_values
    else:
        x_mesh, y_mesh = np.meshgrid(x_vec, y_vec)

        def P_dis(x, y):
            return (
                1 / (np.pi * noise_level) * np.exp(-1 / noise_level * (x**2 + y**2))
            )  # 直接自己定义了一个

        p_values = P_dis(x_mesh, y_mesh)
        print(np.sum(p_values))
        d_values = scipy.signal.convolve2d(q_values, p_values, mode="same") * np.abs(
            (x_vec[1] - x_vec[0]) * (y_vec[1] - y_vec[0])
        )
        # 卷积很tricky————因为要从2m-1个数据点中选m个数据点，那么m就必须是奇数。若为偶数，就会偏差一个点。

    return x_vec, y_vec, d_values
    # 输出是有噪声，卷积后的q函数。noise_level是噪声光子数


def generate_wignerfunc(state: Qobj, xlim, y_lim, x_points, y_points):
    x_vec = np.linspace(*xlim, x_points)
    y_vec = np.linspace(*y_lim, y_points)

    q_values = wigner(state, x_vec, y_vec, g=2)  # 这里取2仅仅是为了简化，让alpha= x+ 1p

    return x_vec, y_vec, q_values


def _coherent(alpha, N):
    sqrtn = jnp.sqrt(jnp.arange(0, N, dtype=jnp.complex128))
    sqrtn = sqrtn.at[0].set(1)  # Get rid of divide by zero warning
    data = alpha / sqrtn

    data = data.at[0].set(jnp.exp(-(abs(alpha) ** 2) / 2.0))

    state = jnp.cumprod(data)

    return jnp.reshape(state, (N, 1))


def _coherent2(alpha, N):
    n_array = jnp.arange(0, N, dtype=jnp.float64)
    sqrtn = jnp.sqrt(jax.vmap(jax.scipy.special.factorial)(n_array))

    alpha_n_array = jnp.power(alpha, n_array) / sqrtn

    return jnp.reshape(alpha_n_array, (N, 1)) * jnp.exp(-0.5 * jnp.abs(alpha) ** 2)


def _coherent3(alpha, N):
    n_array = jnp.arange(0, N, dtype=jnp.float64)
    sqrtn = jnp.sqrt(jax.scipy.special.factorial(n_array))

    alpha_n_array = jnp.power(alpha, n_array) / sqrtn

    return jnp.reshape(alpha_n_array, (N, 1)) * jnp.exp(-0.5 * jnp.abs(alpha) ** 2)


class QRep(npy.distributions.Distribution):
    support = constraints.real

    def __init__(self, rho, N=20, nof_modes=1):
        self.rho = rho
        self.N = N
        assert N**nof_modes == rho.shape[0], "Invalid rho shape{} and N {}".format(
            rho.shape, N
        )
        self.nof_modes = nof_modes
        super().__init__(batch_shape=(), event_shape=(2 * nof_modes,))

    def sample(self, key, sample_shape=()):
        raise NotImplementedError

    @partial(jax.jit, static_argnums=0)
    def log_prob(self, value):
        value = value.reshape((self.nof_modes, 2))
        # print(value.shape)
        alpha_list = value[:, 0] + 1j * value[:, 1]

        # Calculate the alpha state list
        alpha_state_list = jax.vmap(_coherent, in_axes=(0, None))(alpha_list, self.N)

        # Use kron to reduce
        final_state = alpha_state_list[0]
        # print(final_state.shape)
        for i in range(1, self.nof_modes):
            final_state = jnp.kron(final_state, alpha_state_list[i])
        # print(final_state.shape)
        # Calculate Q function value
        q_value = jnp.real(final_state.conj().T @ self.rho @ final_state) / (
            jnp.pi**self.nof_modes
        )

        return jnp.log(q_value)



def generate_multimode_data(
    state, N_single, nof_modes, num_chains=16, num_samples=200000, key=0
):
    # state to be learned
    rho0 = dq.unit(dq.todm(state)).to_jax()

    def model():
        npy.sample("Q", QRep(rho0, N_single, nof_modes))

    mcmc = npy.infer.MCMC(
        npy.infer.NUTS(model),
        num_warmup=50000,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="vectorized",
    )

    # get samples
    mcmc.run(jax.random.PRNGKey(key))
    samples = mcmc.get_samples()

    return samples["Q"]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    jax.config.update("jax_platform_name", "cpu")
    dq.set_device('cpu')
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    N_single = 5
    nof_modes = 4
    print("?????")

    N = int(N_single**nof_modes)
    # state = dq.basis(N, 0) + dq.basis(N, 4) + dq.basis(N, 8)
    # state = dq.basis(N, 10) + dq.basis(N, 5) * 1j
    # state = dq.coherent(N, 2.0) + dq.coherent(N, -2.0) + dq.coherent(N, 2j) + dq.coherent(N, -2j)
    # state = dq.basis(N, 50)
    # state = dq.tensor(dq.basis(N_single, 1), dq.basis(N_single, 1)) + dq.tensor(
    #     dq.basis(N_single, 1), dq.basis(N_single, 0)
    # )
    # state = dq.tensor(dq.basis(N_single, 0), dq.basis(N_single, 0), dq.basis(N_single, 1), dq.basis(N_single, 1)) + dq.tensor(
    #     dq.basis(N_single, 1), dq.basis(N_single, 1), dq.basis(N_single, 0), dq.basis(N_single, 0)
    # )
    # state = dq.unit(state)
    test_alpha = 0.5
    state = (
        dq.tensor(
            dq.coherent(N_single, -test_alpha),
            dq.coherent(N_single, -test_alpha),
            dq.coherent(N_single, -test_alpha),
            dq.coherent(N_single, test_alpha),
        )
        + dq.tensor(
            dq.coherent(N_single, -test_alpha),
            dq.coherent(N_single, -test_alpha),
            dq.coherent(N_single, test_alpha),
            dq.coherent(N_single, -test_alpha),
        )
        + dq.tensor(
            dq.coherent(N_single, -test_alpha),
            dq.coherent(N_single, test_alpha),
            dq.coherent(N_single, -test_alpha),
            dq.coherent(N_single, -test_alpha),
        )
        + dq.tensor(
            dq.coherent(N_single, test_alpha),
            dq.coherent(N_single, -test_alpha),
            dq.coherent(N_single, -test_alpha),
            dq.coherent(N_single, -test_alpha),
        )
    )
    state = dq.unit(state)

    Q_samples = generate_multimode_data(
        state, N_single, nof_modes, num_chains=16, num_samples=200000, key=0
    )

    jnp.save("data/q_samples_four_modes.npy", Q_samples)

    plt.figure()
    plt.hist2d(Q_samples[:, 0], Q_samples[:, 1], bins=100)
    plt.figure()
    plt.hist2d(Q_samples[:, 2], Q_samples[:, 3], bins=100)



    # import numpy as np
    # hist = np.histogramdd(Q_samples, bins=32, density=True)
