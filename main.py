from pprint import pprint

import matplotlib.pyplot as plt
from qiskit import Aer
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.test.mock import FakeVigo
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

"""
# Pipeline!!

# Create Dataset
# (Xi, Yi)
# Xi = circuit + measurements
# Yi = noise model

# 1. Kinds of circuits
# Random Unitaries, Random Cliffords, Certain 1-qubit gate

# 2. Kinds of Noise Models
# Custom
# Backend noise model
# thermal_relaxation_error, readout_errors
"""


def print_noise_model(noise_model):
    pprint(vars(noise_model))


def run_circuit(qc, simulator, noise_model, coupling_map, basis_gates, shots=1000):
    result = execute(qc, simulator,
                     noise_model=noise_model,
                     coupling_map=coupling_map,
                     basis_gates=basis_gates,
                     shots=shots).result()  # we run the simulation
    counts = result.get_counts()  # we get the counts
    return counts


def main():
    # Noise

    device_backend = FakeVigo()
    coupling_map = device_backend.configuration().coupling_map
    simulator = Aer.get_backend('qasm_simulator')

    noise_model = NoiseModel()
    basis_gates = noise_model.basis_gates
    error = depolarizing_error(0.05, 1)
    noise_model.add_all_qubit_quantum_error(error, ['id', 'u1', 'u2', 'u3'])

    # Circuit
    circ = QuantumCircuit(1, 1)
    circ.id(0)
    circ.measure([0], [0])
    print("*" * 25 + " Circuit " + "*" * 25)
    print(circ.draw())

    model1 = {'name': 'noisy', 'model': noise_model}
    model2 = {'name': 'ideal', 'model': None}
    noise_models = [model1, model2]
    print()
    print("*" * 25 + " Noise Models " + "*" * 25)
    pprint(noise_models)

    # Execution
    Dataset = [(run_circuit(circ, simulator, nm['model'], coupling_map, basis_gates), nm) for nm in noise_models]

    # Data Prep
    for counts, nm in Dataset:
        counts.setdefault('0', 0)
        counts.setdefault('1', 0)

    X = [[x[1] for x in sorted(counts.items())] for counts, nm in Dataset]
    Y_raw = [nm['name'] for counts, nm in Dataset]

    le = preprocessing.LabelEncoder()
    le.fit(Y_raw)
    Y = le.transform(Y_raw)
    print()
    print("*" * 25 + " Dataset " + "*" * 25)
    print("Features: ", X)
    print("Labels: ", Y_raw)
    print("Encoded Labels: ", Y)

    # Training Classifier
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X, Y)

    print()
    print("*" * 25 + " Predictions " + "*" * 25)

    # Predict labels on original data
    print("Prediction on training set: ", list(le.inverse_transform(clf.predict(X))))

    # Predict labels on new data
    Xtest = []
    Ytest = []
    for i in range(900, 1001):
        feature = [[i, 1000 - i]]
        prediction = list(le.inverse_transform(clf.predict(feature)))
        # print("Prediction on {}: {}".format(feature, prediction))
        Xtest.append(i)
        Ytest.append(prediction[0])
    plt.scatter(Xtest, Ytest, label="test set")
    plt.scatter([x[0] for x in X], Y_raw, label="training set")
    plt.xlabel("Feature = Count of '0' measurements")
    plt.ylabel("Predicted Label")
    plt.legend()
    plt.show()

    # todo: encode circuit


if __name__ == '__main__':
    main()
