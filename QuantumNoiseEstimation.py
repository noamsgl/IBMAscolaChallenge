from qiskit import Aer
from qiskit import execute
from qiskit.test.mock import FakeVigo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class QuantumNoiseEstimator:
    def __init__(self, dataset=None, classifier=RandomForestClassifier()):
        """
        Initialize the class
        :param dataset: the dataset (QuantumNoiseDataset)
        :param classifier: the classifier to use
        todo: generalize classifier to an estimator (regression etc.)
        """
        self.dataset = dataset
        assert self.dataset is not None, "dataset is empty"
        self.classifier = classifier
        self.encoder_decoder = QuantumDatasetEncoder(classifier)
        encoded_features = self.encoder_decoder.encode_features(self.dataset)
        encoded_labels = self.encoder_decoder.encode_labels(self.dataset)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(encoded_features, encoded_labels,
                                                                                test_size=0.33)

    def fit(self):
        """
        Fit the model to the dataset
        :return:
        """
        self.classifier.fit(self.X_train, self.Y_train)

    def predict(self, feature):
        encoded_feature = self.encoder_decoder.encode_feature(feature)
        encoded_label = self.classifier.predict(encoded_feature)
        decoded_label = self.encoder_decoder.decode_label(encoded_label)
        return decoded_label

    def __repr__(self):
        return "I am a Quantum Noise Estimator"


class DatasetGenerator:
    def __init__(self, shots=1000):
        """
        Initialize this class
        :param shots: number of times to measure each circuit
        """
        self.shots = shots
        self.observations = []
        self.features = []
        self.labels = []
        self.dataset = QuantumNoiseDataset()

        # Device Model
        self.device_backend = FakeVigo()
        self.coupling_map = self.device_backend.configuration().coupling_map
        self.simulator = Aer.get_backend('qasm simulator')

    def add_observation(self, circuit, noise_model):
        """
        Add a new data point - (feature, label) -to the dataset. Blocking method (until counts are calculated)
        :param circuit:
        :param noise_model:
        :return: none
        """
        feature = (circuit, noise_model)
        label = self.get_counts(circuit, noise_model)
        self.dataset.add_data_point(feature, label)

    def get_counts(self, circuit, noise_model):
        """
        Simulate a circuit with a noise_model and return measurement counts
        :param circuit:
        :param noise_model:
        :return: measurement counts (dict)
        """
        result = execute(circuit, self.simulator, noise_model=noise_model, coupling_map=self.coupling_map,
                         basis_gates=noise_model.basis_gates, shots=self.shots).result()
        counts = result.get_counts()
        return counts

    def get_dataset(self):
        """
        :return: the dataset (QuantumNoiseDataset)
        """
        return self.dataset

    def __repr__(self):
        return "I am a dataset generator for quantum noise estimation"

    def mock_dataset(self):
        """
        Generate a mock dataset
        :return: the dataset (QuantumNoiseDataset)
        """
        raise NotImplementedError


class QuantumNoiseDataset:
    def __init__(self):
        self.data_points = []

    def add_data_point(self, feature, label):
        dp = (feature, label)
        self.data_points.append(dp)


class QuantumDatasetEncoder:
    def __init__(self, classifier):
        """
        Initialize the class
        :param classifier: the classifier (might be useful later)
        Encodes and Decodes dataset for sklearn classifier
        """
        self.classifier = classifier
        self.features_encoder = OneHotEncoder()
        self.labels_encoder = None

    def encode_features(self, dataset):
        raise NotImplementedError

    def encode_labels(self, dataset):
        raise NotImplementedError

    def decode_labels(self, labels):
        raise NotImplementedError

    def decode_label(self, encoded_label):
        raise NotImplementedError

    def encode_feature(self, feature):
        raise NotImplementedError
