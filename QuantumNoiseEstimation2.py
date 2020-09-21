import itertools
import pandas as pd
from qiskit import Aer
from qiskit import execute
from qiskit.test.mock import FakeVigo


class DatasetGenerator:
    def __init__(self, shots=1000, L=6):
        """
        Initialize a new DatasetGenerator.
        This object provides an interface to generate synthetic quantum-noisy datasets.
        The user must prescribe the circuits and noise models.
        Some presets are available.
        :param shots: number of times to measure each circuit, default: 1000
        :param L: maximum circuit length
        """

        self.shots = shots
        self.L = L
        self.observations = []
        self.features = []
        self.labels = []
        self.dataset = self.emptyDataset()

        # Simulator variables
        self.device_backend = FakeVigo()
        self.coupling_map = self.device_backend.configuration().coupling_map
        self.simulator = Aer.get_backend('qasm_simulator')

    def add_observation(self, circuit, noise_model):
        """
        Add a new data point - (feature, label) - to the dataset.
        :param circuit: circuit, len(circuit) <= self.L
        :param noise_model:
        :return: none
        """

        assert len(circuit) <= self.L

        feature = (circuit, noise_model)
        label = self.get_counts(circuit, noise_model)
        self.dataset.add_data_point(feature, label)

    def get_dataset(self):
        """
        :return: the dataset (DataFrame)
        """
        return self.dataset

    def get_counts(self, circuit, noise_model):
        """
        Simulate a circuit with a noise_model and return measurement counts
        :param circuit:
        :param noise_model:
        :return: measurement counts (dict {'0':C0, '1':C1})
        """
        result = execute(circuit, self.simulator, noise_model=noise_model, coupling_map=self.coupling_map,
                         basis_gates=noise_model.basis_gates, shots=self.shots).result()
        counts = result.get_counts()
        return counts

    def __repr__(self):
        return "I am a dataset generator for quantum noise estimation"

    def mock_dataset(self):
        """
        Generate a mock dataset
        :return: the dataset (QuantumNoiseDataset)
        """
        raise NotImplementedError

    def emptyDataset(self):
        """
        Generate an empty Dataset.
        Column format:
        Gate_1
        NM_1
        ...
        ...
        Expected Value
        :return: df: DataFrame
        """
        gateStrings = ["Gate_{}".format(i) for i in range(self.L)]
        nmStrings = ["NM_{}".format(i) for i in range(self.L)]
        column_names = []
        for i in range(len(gateStrings)):
            column_names.append(gateStrings[i])
            column_names.append(nmStrings[i])
        column_names.append("ExpectedValue")
        df = pd.DataFrame(columns=column_names)
        return df


dgen = DatasetGenerator()