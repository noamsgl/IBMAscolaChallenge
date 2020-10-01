import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.circuit.library import RXGate, RZGate, RYGate, U3Gate
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from qiskit.visualization import plot_histogram
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

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
shots = 1024
basis_gates = ['u3']
simulator = QasmSimulator()
df = pd.DataFrame()

# generate circuits
gate_name = 'U3'
for theta in np.linspace(0, np.pi, 8, endpoint=True):
    for phi in np.linspace(0, np.pi, 8, endpoint=True):
        for lam in np.linspace(0, np.pi, 8, endpoint=True):
            # define circuit
            circ = QuantumCircuit(1, 1)
            gate = U3Gate(theta, phi, lam)
            circ.append(gate, [0])
            circ.measure(0, 0)
            print(circ)
            new_circ = qiskit.compiler.transpile(circ, basis_gates=basis_gates, optimization_level=0)
            print(new_circ)

            # generate noise models:
            for probability in np.linspace(0, 1, 6, endpoint=True):
                # add noise
                noise_model = NoiseModel()
                error = depolarizing_error(probability, 1)
                noise_model.add_all_qubit_quantum_error(error, ['x', 'u1', 'u2', 'u3'])

                # execution - Noisy
                job = execute(new_circ, simulator, shots=shots, noise_model=noise_model)
                result = job.result()
                # plot_histogram(result.get_counts(0))
                # plt.title("Gate=U3({}, {}, {}), p={}".format(theta, phi, lam, probability))
                # plt.savefig(working_dir + "/{}_{:f}_{:f}_{:f}_p_{:f}.jpg".format(gate_name, theta, phi, lam, probability))


                # add to Pandas DF
                data = {'theta': theta,
                        'phi': phi,
                        'lam': lam,
                        'p': probability,
                        'E': result.get_counts(0).get('0', 0) / shots}
                df = df.append(data, ignore_index=True)

# save DF to csv
df = df[['theta', 'phi', 'lam', 'E', 'p']]
df.to_csv(working_dir + "/dataframe_{}.csv".format(number))


# Regression Task
X = df[['theta', 'phi', 'lam', 'E']]
y = df[['p']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

kneighbor_regression = KNeighborsRegressor(n_neighbors=1)
kneighbor_regression.fit(X_train, y_train)
y_pred_train = kneighbor_regression.predict(X_train)

plt.plot(X_train, y_train, 'o', label="data", markersize=10)
plt.plot(X_train, y_pred_train, 's', label="prediction", markersize=4)
plt.legend(loc='best');
plt.show()

y_pred_test = kneighbor_regression.predict(X_test)

plt.plot(X_test, y_test, 'o', label="data", markersize=8)
plt.plot(X_test, y_pred_test, 's', label="prediction", markersize=4)
plt.legend(loc='best');
plt.show()

print(kneighbor_regression.score(X_test, y_test))
