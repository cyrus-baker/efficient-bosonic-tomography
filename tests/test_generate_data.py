
import dynamiqs as dq

from efficient_bosonic_tomography.generate_data import generate_multimode_data


def test_sample_from_qfunc():
    N_single = 5
    nof_modes = 4
    nof_samples = 10
    num_chains = 16
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
        state, N_single, nof_modes, num_chains=num_chains, num_samples=nof_samples, key=0, warmup=0
    )
    
    assert Q_samples.shape == (nof_samples * num_chains, 2 * nof_modes)
