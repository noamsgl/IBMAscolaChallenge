import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate
import numpy as np

basis_gates = ['u3']
circ = QuantumCircuit(1, 1)
RX = RXGate(0)
circ.append(RX, [0])
circ.measure(0, 0)
print("Before Transpiling:")
print(circ)
new_circ = qiskit.compiler.transpile(circ, basis_gates=basis_gates, optimization_level=0)
print("After Transpiling:")
print(new_circ)
