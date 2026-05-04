from qiskit import QuantumCircuit, transpile, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
import math
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

#données de départ 
p = np.array([0, 0, 0, 0, 0, 0.25, 0, 0, 0, 0.25, 0.25, 0.25, 0.25,0.25, 0.25, 0.25])
# Normalisation pour que la somme = 1
p_norm = p / np.sum(p)

pi = math.pi


def oracle_mappage(qc, q):
  
    # pour transformer les 0  en 1
    qc.x(q[0])  # q[0] doit être 0 → on le flip en 1
    qc.x(q[1])  # q[1] doit être 0 → on le flip en 1
    
    # Étape 2 : MCZ (Z multi-contrôlée sur 4 qubits)
    # Se déclenche quand q[0]=q[1]=q[2]=q[3]=1
    # ce qui correspond à q[0]=0, q[1]=0, q[2]=1, q[3]=1 avant le flip
    # → exactement |1100⟩
    #
    # MCZ = H sur la cible + MCX + H sur la cible
    qc.h(q[3])# transforme Z en X sur la cible
    qc.mcx([q[0], q[1], q[2]], [3] ) #qbit controle 0,1,2  qubit cible 3
    
    qc.h(q[3])          # retransformation
    
    # Étape 3 : on remet q[0] et q[1] dans leur état d'origine
    qc.x(q[0])
    qc.x(q[1])

q = QuantumRegister(4,'q')
c = ClassicalRegister(4,'c')
qc = QuantumCircuit(q,c)


initial_state = np.sqrt(p_norm)

backend = AerSimulator()

a_circuit = QuantumCircuit(4)
a_circuit.initialize(initial_state, range(4))
a_circuit_transpiled = transpile(a_circuit, backend, basis_gates=['u', 'cx'])
A_gate = a_circuit_transpiled.to_gate()
A_gate.name = "A"
qc.append(A_gate, q)

A_gate = a_circuit.to_instruction()

#Utilisation dans le circuit principal
qc.append(A_gate, q) # étape A 
qc.barrier()

oracle_mappage(qc, q)


#1100
#transforme les 2 premier qbit de 1 à 0 et les 2 suivants restent à 0
qc.x(q[0])#Pauli-x gate change |0> à |1> et |1> à |0>
qc.x(q[1])
# Mesure 
qc.barrier(q)
qc.measure(q[0], c[0])
qc.measure(q[1], c[1])
qc.measure(q[2], c[2])
qc.measure(q[3], c[3])

circ = transpile(qc, backend) 
result = backend.run(circ,shots=1024).result() #donne par chaque état la fréquence d'apparition  
counts = result.get_counts(circ)
plot_histogram(counts)
plt.show()

qc.cp(pi/4, q[0], q[3]) #cp gate (control phase) : applique une rotation de pi/4 sur le qubit cible q[3] si le qubit de contrôle q[0] est dans l'état |1>
qc.cx(q[0], q[1])# Porte CNOT (Controlled-NOT) utilisée pour la propagation du contrôle selon Barenco et al
qc.cx(q[0], q[1])
qc.cp(pi/4, q[1], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])

qc.x(q[0])
qc.x(q[1])

qc.h(q[0])
qc.h(q[1])
qc.h(q[2])
qc.h(q[3])
qc.x(q[0])
qc.x(q[1])
qc.x(q[2])
qc.x(q[3])

qc.cp(pi/4, q[0], q[3])
qc.cx(q[0], q[1])
qc.cp(-pi/4, q[1], q[3])
qc.cx(q[0], q[1])
qc.cp(pi/4, q[1], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
qc.cx(q[1], q[2])

qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])

qc.x(q[0])
qc.x(q[1])
qc.x(q[2])
qc.x(q[3])
qc.h(q[0])
qc.h(q[1])
qc.h(q[2])
qc.h(q[3])

# Mesure 
qc.barrier(q)
qc.measure(q[0], c[0])
qc.measure(q[1], c[1])
qc.measure(q[2], c[2])
qc.measure(q[3], c[3])


print('\n Simulation et diffusion de Grover\n')

backend = AerSimulator() 
circ = transpile(qc, backend) 

result = backend.run(circ,shots=1024).result() #donne par chaque état la fréquence d'apparition  
counts = result.get_counts(circ)

print('RESULT: ',counts)
plot_histogram(counts)
plt.show()
