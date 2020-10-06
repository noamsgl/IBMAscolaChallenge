import numpy as np
import pandas as pd
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.circuit.library import U3Gate, RXGate
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error


def U3Dataset(angle_step=9, probability_step=10, shots=1024, save_dir=None):
    # define constants
    basis_gates = ['u3']
    simulator = QasmSimulator()
    df = pd.DataFrame()

    # generate circuits
    for theta in np.linspace(0, np.pi, angle_step, endpoint=True):
        for phi in np.linspace(0, np.pi, angle_step, endpoint=True):
            for lam in np.linspace(0, np.pi, angle_step, endpoint=True):
                # define circuit
                circ = QuantumCircuit(1, 1)
                gate = U3Gate(theta, phi, lam)
                circ.append(gate, [0])
                circ.measure(0, 0)
                new_circ = qiskit.compiler.transpile(circ, basis_gates=basis_gates, optimization_level=0)
                print(new_circ)

                # generate noise models:
                for probability in np.linspace(0, 1, probability_step, endpoint=True):

                    # add noise
                    noise_model = NoiseModel()
                    error = depolarizing_error(probability, 1)
                    noise_model.add_all_qubit_quantum_error(error, ['x', 'u1', 'u2', 'u3'])

                    # execution - Noisy
                    job = execute(new_circ, simulator, shots=shots, noise_model=noise_model)
                    result = job.result()

                    # add to Pandas DF
                    data = {'theta': theta,
                            'phi': phi,
                            'lam': lam,
                            'p': probability,
                            'E': result.get_counts(0).get('0', 0) / shots}
                    df = df.append(data, ignore_index=True)
    df = df[['theta', 'phi', 'lam', 'E', 'p']]
    if save_dir is not None:
        df.to_csv(save_dir + "/dataframe_U3.csv")
    return df


def RXDataset(angle_step=10, probability_step=10, shots=1024, save_dir=None):
    # define constants
    basis_gates = ['u3']
    simulator = QasmSimulator()
    df = pd.DataFrame()

    # generate circuits
    for angle in np.linspace(0, np.pi, angle_step, endpoint=True):
        # define circuit
        circ = QuantumCircuit(1, 1)
        rotate = RXGate(angle)
        circ.append(rotate, [0])
        circ.measure(0, 0)
        new_circ = qiskit.compiler.transpile(circ, basis_gates=basis_gates, optimization_level=0)
        print(new_circ)

        # generate noise models:
        for probability in np.linspace(0, 1, probability_step, endpoint=True):
            # add noise
            noise_model = NoiseModel()
            error = depolarizing_error(probability, 1)
            noise_model.add_all_qubit_quantum_error(error, ['x', 'u1', 'u2', 'u3'])

            # execution - Noisy
            job = execute(new_circ, simulator, shots=shots, noise_model=noise_model)
            result = job.result()

            # add to Pandas DF
            data = {'rx_theta': angle,
                    'p': probability,
                    'E': result.get_counts(0).get('0', 0) / shots}
            df = df.append(data, ignore_index=True)
    df = df[['rx_theta', 'E', 'p']]
    df.to_csv(save_dir + "/dataframe_RX.csv")
    return df