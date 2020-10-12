import csv
import datetime
import os

from qiskit import IBMQ

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-education', group='technion-Weinste')
log_time = datetime.datetime.now()


def write_log_to_file(csv_writer):
    for backend in provider.backends():
        try:
            for qubit_idx, qubit_prop in enumerate(backend.properties().qubits):
                for prop in qubit_prop:
                    csv_writer.writerow(
                        [backend.name(), qubit_idx, prop.date.isoformat(), prop.name, prop.unit, prop.value,
                         log_time.isoformat()])
                id_gate = [i for i in backend.properties().gates if i.gate == 'id' and qubit_idx in i.qubits][0]
                id_gate_length = [p for p in id_gate.parameters if p.name == 'gate_length'][0]
                csv_writer.writerow([backend.name(), qubit_idx, id_gate_length.date.isoformat(), 'id_length',
                                     id_gate_length.unit, id_gate_length.value, log_time.isoformat()])
        except Exception as e:
            print("Cannot add backend", backend.name(), e)


path = '../docs/backend_properties/backend_properties_log.csv'
if not os.path.isfile(path):
    with open(path, "w", newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["backend", "qubit", "datetime", "name", "units", "value", "log_datetime"])
        write_log_to_file(csv_writer)
else:
    with open(path, "a", newline='') as f:
        write_log_to_file(csv.writer(f))
