from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# Création du circuit (Grover 2 qubits pour l'état 11)
circuit = QuantumCircuit(2, 2)
circuit.h([0, 1])
circuit.cz(0, 1) # Oracle
circuit.h([0, 1])
circuit.z([0, 1])
circuit.cz(0, 1)
circuit.h([0, 1])
print(circuit.draw(output='text'))
circuit.measure([0, 1], [0, 1])

#Init simu
simulator = AerSimulator()

#execut
job = simulator.run(circuit, shots=1024)
result = job.result()

#Récup et affiche les résultats
counts = result.get_counts()
print("Résultats des mesures (signatures détectées) :", counts)