import jax.numpy as jnp
import dynamiqs as dq
import jax.random as jr

from efficient_bosonic_tomography.displacer import (
    Displacer,
    alpha2row,
    Alpha2Row,
    Alpha2RowMultiMode,
    alpha2row_multimode,
    Alpha2RowMultiModeNoNoise,
    alpha2row_multimode_nonoise,
    Alpha2RowWigner,
    Alpha2RowMultiModeWigner,
)



def test_displacer_displace_unitary():
    n = 4
    displacer = Displacer(n)
    alpha = 0.3 + 0.2j
    Da = displacer.displace(alpha)
    eye = jnp.eye(n, dtype=Da.dtype)
    assert Da.shape == (n, n)
    assert jnp.allclose(Da.conj().T @ Da, eye, atol=1e-5)


def test_alpha2row_zero_alpha_matches_rho():
    N = 3
    nof_samples = 10
    alpha_list = jnp.zeros((nof_samples,))
    rho_h = dq.basis_dm(N, 0).to_jax()
    result = alpha2row(alpha_list, rho_h=rho_h, N=N)
    expected = jnp.array(rho_h.T.flatten(order="F"))
    assert result.shape == (nof_samples, N * N)
    assert jnp.allclose(result[0], expected, atol=1e-6)


def test_alpha2row_module_outputs_hermitian():
    N = 5
    nof_samples = 10
    key = jr.key(12)
    k1, k2 = jr.split(key)

    alpha_vec = jr.normal(k1, (nof_samples,)) + 1j * jr.normal(k2, (nof_samples,))
    module = Alpha2Row(N=N, N_compute=10)
    result = module(alpha_vec)
    mat = result[0].reshape((N, N), order="F")
    assert result.shape == (nof_samples, N * N)
    assert jnp.allclose(mat, mat.conj().T, atol=1e-6)


def test_alpha2row_multimode_shape_and_hermitian():
    N_single = 2
    num_modes = 2
    dim = N_single ** num_modes
    nof_samples = 10
    key = jr.key(12)
    k1, k2 = jr.split(key)
    alpha_vec = jr.normal(k1, (nof_samples,)) + 1j * jr.normal(k2, (nof_samples,))
    alpha_vec = jnp.vstack([alpha_vec, alpha_vec]).T
    module = Alpha2RowMultiMode(N_single=N_single, num_modes=num_modes, N_compute=10)
    result = module(alpha_vec)
    mat = result[0].reshape((dim, dim), order="F")
    assert result.shape == (nof_samples, dim * dim)
    assert jnp.allclose(mat, mat.conj().T, atol=1e-6)


def test_alpha2row_multimode_function_shape():
    N_single = 2
    num_modes = 3
    dim = N_single ** num_modes
    nof_samples = 10
    key = jr.key(12)
    k1, k2 = jr.split(key)
    alpha_vec = jr.normal(k1, (nof_samples,)) + 1j * jr.normal(k2, (nof_samples,))
    alpha_vec = jnp.vstack([alpha_vec, alpha_vec]).T
    result = alpha2row_multimode(alpha_vec, N_single=N_single, num_modes=num_modes)
    assert result.shape == (nof_samples, dim * dim)


def test_alpha2row_multimode_no_noise_trace_one():
    N_single = 3
    num_modes = 2
    dim = N_single ** num_modes
    nof_samples = 10
    key = jr.key(12)
    k1, k2 = jr.split(key)
    alpha_vec = jnp.zeros((nof_samples, 2))
    module = Alpha2RowMultiModeNoNoise(N_single=N_single, num_modes=num_modes)
    result = module(alpha_vec)
    mat = result[0].reshape((dim, dim), order="F")
    trace_val = jnp.trace(mat)
    assert result.shape == (nof_samples, dim * dim)
    assert jnp.allclose(trace_val, 1.0, atol=1e-5)


def test_alpha2row_multimode_nonoise_trace_one():
    N_single = 2
    num_modes = 2
    dim = N_single ** num_modes
    nof_samples = 10
    key = jr.key(12)
    k1, k2 = jr.split(key)
    alpha_vec = jnp.zeros((nof_samples, 2))
    result = alpha2row_multimode_nonoise(alpha_vec, N_single=N_single, num_modes=num_modes)
    mat = result[0].reshape((dim, dim), order="F")
    trace_val = jnp.trace(mat)
    assert result.shape == (nof_samples, dim * dim)
    assert jnp.allclose(trace_val, 1.0, atol=1e-5)
    


