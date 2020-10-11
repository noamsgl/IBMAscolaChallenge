from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.device import thermal_relaxation_values, gate_param_values
from qiskit.providers.aer.noise.noiseerror import NoiseError

provider = IBMQ.load_account()
print("Available Backends: ", provider.backends())

for be in provider.backends():
    try:
        print()
        print("*" * 20 + be.name() + "*" * 20)
        noise_model = NoiseModel.from_backend(be)

        # readout
        print("Readout Error on q_0:", noise_model._local_readout_errors['0'])

        # thermal and depol
        properties = be.properties()
        relax_params = thermal_relaxation_values(properties)  # T1, T2 (microS) and frequency values (GHz)
        t1, t2, freq = relax_params[0]
        u3_length = properties.gate_length('u3', 0)
        print("t1: " + str(t1))
        print("t2: " + str(t2))
        print("freq: " + str(freq))
        print("universal_error gate length: " + str(u3_length))

        # depol
        device_gate_params = gate_param_values(properties)

    except NoiseError:
        pass
