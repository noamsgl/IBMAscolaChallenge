import pandas as pd
from qiskit import Aer
from qiskit import execute
from qiskit.test.mock import FakeVigo


class DatasetGenerator:
    def __init__(self, shots=1000, L=3):
        """
        Initialize a new DatasetGenerator.
        This object provides an interface to generate synthetic quantum-noisy datasets.
        The user must prescribe the circuits and noise models.
        Some presets are available.
        :param shots: number of times to measure each circuit, default: 1000
        :param L: maximum circuit length
        """

        # Simulator Variables
        self.device_backend = FakeVigo()
        self.coupling_map = self.device_backend.configuration().coupling_map
        self.simulator = Aer.get_backend('qasm_simulator')

        self.shots = shots
        self.L = L
        self.dataset = self.emptyDataset()

        """self.observations = []
        self.features = []
        self.labels = []"""



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

        column_names = []
        for i in range(self.L):
            column_names.append("Gate_{}_theta".format(i))
            column_names.append("Gate_{}_phi".format(i))
            column_names.append("Gate_{}_lambda".format(i))
            column_names.append("NM_{}".format(i))
        column_names.append("ExpectedValue")
        df = pd.DataFrame(dtype='float64', columns=column_names)
        df.loc[0] = pd.Series(dtype='float64')
        df.loc[1] = pd.Series(dtype='float64')
        df.loc[2] = pd.Series(dtype='float64')
        return df

    def __repr__(self):
        return "I am a dataset generator for quantum noise estimation"




dgen = DatasetGenerator()
ds = dgen.dataset

print("""
ds is a DataFrame with defined columns and 3 null rows:""")
print(ds)
print()
print("ds.info()")
print(ds.info())

"""
Questions

1. how to encode noise model
2. how to generate data samples

"""


