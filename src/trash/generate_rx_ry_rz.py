import os

import matplotlib.pyplot as plt
import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.circuit.library import RXGate, RZGate, RYGate
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from qiskit.visualization import plot_histogram

# create directory
reading_file = open("../../outputs/config.txt", 'r')
number = reading_file.readline()
reading_file.close()
os.mkdir("../outputs/{}".format(number))
writing_file = open("../../outputs/config.txt", 'w')
writing_file.write(str(int(number) + 1))
writing_file.close()

working_dir = "../outputs/{}".format(number)

# define constants
shots = 4096
basis_gates = ['u3']
simulator = QasmSimulator()

# generate circuits
for gate, gate_name in [(RXGate, "rx"), (RYGate, "ry"), (RZGate, "rz")]:
    for angle in np.linspace(0, np.pi, 10, endpoint=True):
        # define circuit
        circ = QuantumCircuit(1, 1)
        rotate = gate(angle)
        circ.append(rotate, [0])
        circ.measure(0, 0)
        print(circ)
        new_circ = qiskit.compiler.transpile(circ, basis_gates=basis_gates, optimization_level=0)
        print(new_circ)

        # generate noise models:
        for probability in np.linspace(0, 1, 10, endpoint=True):
            # add noise
            noise_model = NoiseModel()
            error = depolarizing_error(probability, 1)
            noise_model.add_all_qubit_quantum_error(error, ['x', 'u1', 'u2', 'u3'])

            # execution - Noisy
            job = execute(new_circ, simulator, shots=shots, noise_model=noise_model)
            result = job.result()
            plot_histogram(result.get_counts(0))
            plt.title("Gate=RX({}), p={}".format(angle, probability))
            plt.savefig(working_dir + "/{}_{:f}_p_{:f}.jpg".format(gate_name, angle, probability))
