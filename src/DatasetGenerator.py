from itertools import product

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, execute
from qiskit.circuit.library import U3Gate
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, ReadoutError, thermal_relaxation_error
from tqdm import tqdm


def get_data_point(circ, theta, phi, lam, readout_params, depol_param, thermal_params, shots):
    """Generate a dict datapoint with the given circuit and noise parameters"""
    U3_gate_length = 7.111111111111112e-08

    # extract parameters
    (p0_0, p1_0), (p0_1, p1_1) = readout_params
    depol_prob = depol_param
    t1, t2, population = thermal_params

    # Add Readout and Quantum Errors
    noise_model = NoiseModel()
    noise_model.add_all_qubit_readout_error(ReadoutError(readout_params))
    noise_model.add_all_qubit_quantum_error(depolarizing_error(depol_param, 1), 'u3', warnings=False)
    noise_model.add_all_qubit_quantum_error(
        thermal_relaxation_error(t1, t2, U3_gate_length, excited_state_population=population), 'u3',
        warnings=False)
    job = execute(circ, QasmSimulator(), shots=shots, noise_model=noise_model)
    result = job.result()

    # add data point to DataFrame
    data_point = {'theta': theta,
                  'phi': phi,
                  'lam': lam,
                  'p0_0': p0_0,
                  'p1_0': p1_0,
                  'p0_1': p0_1,
                  'p1_1': p1_1,
                  'depol_prob': depol_prob,
                  't1': t1,
                  't2': t2,
                  'population': population,
                  'E': result.get_counts(0).get('1', 0) / shots}

    return data_point


def get_noise_model_params(K=10, readout=True, thermal=True, depol=True):
    """generate a list of tuples for all noise params (readout_params, depol_param, thermal_params) """

    # Simple implementation. Use this...
    noise_model_params = []

    # Readout Error
    for p0_0 in np.linspace(0.94, 1.0, K, endpoint=True):
        for p1_1 in np.linspace(0.94, 1.0, K, endpoint=True):
            p1_0 = 1 - p0_0
            p0_1 = 1 - p1_1
            readout_params = (p0_0, p1_0), (p0_1, p1_1)

            # Thermal Error
            for t1 in np.linspace(34000, 190000, K, endpoint=True):
                for t2 in np.linspace(t1 / 5.6, t1 / 0.65, K, endpoint=True):
                    for population in np.linspace(0, 1, K, endpoint=True):
                        thermal_params = (t1, t2, population)

                        # Depolarizing Error
                        for depol_param in np.linspace(0, 0.001, K, endpoint=True):
                            noise_model_params.append((readout_params, depol_param, thermal_params))
    return noise_model_params

    # Inefficient implementation. Don't Use...
    """if readout:
        p0_0_iter = np.linspace(0.94, 1.0, K, endpoint=True)
        p1_1_iter = np.linspace(0.94, 1.0, K, endpoint=True)
    else:
        p0_0_iter = [1]
        p1_1_iter = [1]

    if depol:
        depol_iter = np.linspace(0, 0.001, K, endpoint=True)
    else:
        depol_iter = [0]

    if thermal:
        t1_iter = np.linspace(34000, 190000, K, endpoint=True)
        # thermal_iter = map(lambda t1: (t1, np.linspace(t1/5.6, t1/0.65, 10, endpoint=True)), t1_iter)
        thermal_iter = chain.from_iterable(
            map(lambda t1: product([t1], np.linspace(t1 / 5.6, t1 / 0.65, K, endpoint=True)), t1_iter))

    else:
        thermal_iter = [(np.inf, np.inf)]

    iterator = product(p0_0_iter, p1_1_iter, depol_iter, thermal_iter)
    noise_model_params = [(((p0_0, 1 - p0_0), (1 - p1_1, p1_1)), depol, (t1, t2)) for p0_0, p1_1, depol, (t1, t2) in
                          tqdm(iterator)]
    return noise_model_params"""


def U3Dataset(angle_step=10, other_steps=10, shots=4096, readout=True, thermal=True, depol=True, save_dir=None):
    df = pd.DataFrame()

    # Iterate over all U3 gates
    for theta, phi, lam in tqdm(product(np.linspace(0, 2 * np.pi, angle_step, endpoint=True), repeat=3)):
        # Generate sample data
        circ = QuantumCircuit(1, 1)
        circ.append(U3Gate(theta, phi, lam), [0])
        circ.measure(0, 0)
        print('\n', circ)
        for readout_params, depol_param, thermal_params in get_noise_model_params(other_steps, readout, thermal, depol):
            data_point = get_data_point(circ, theta, phi, lam, readout_params, depol_param, thermal_params, shots)
            df = df.append(data_point, ignore_index=True)
    # Save to CSV
    if save_dir is not None:
        df.to_csv(save_dir)
    return df


for n in [14, 12, 10]:
    angle_step = n
    other_steps = n
    U3Dataset(angle_step=angle_step, other_steps=other_steps, readout=True, thermal=True, depol=True,
              save_dir='../datasets/universal_error/AllErrors/U3_{}.csv'.format(n))
