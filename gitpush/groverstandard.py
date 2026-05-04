from qiskit import QuantumCircuit, transpile, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
import math
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

pi = math.pi

q = QuantumRegister(4,'q')
c = ClassicalRegister(4,'c')
qc = QuantumCircuit(q,c)
angle_q0 = 2.0  # Favorise légèrement le '1' sur q0
angle_q1 = 2.5  # Favorise fortement le '1' sur q1
angle_q2 = 1.0  # Favorise le '0' sur q2
angle_q3 = 1.57 # Reste à 50/50 (équivalent H)

#superposition
qc.h(q[0])
qc.h(q[1])
qc.h(q[2])
qc.h(q[3])

#1100
qc.x(q[0])#Pauli-x gate change |0> à |1> et |1> à |0>
qc.x(q[1])

qc.cp(pi/4, q[0], q[3]) #cp gate (control phase) : applique une rotation de pi/4 sur le qubit cible q[3] si le qubit de contrôle q[0] est dans l'état |1>
qc.cx(q[0], q[1])#inverse du CNOT gate : applique une porte NOT sur le qubit cible q[1] si le qubit de contrôle q[0] est dans l'état |1>
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


print('\n Executing job....\n')

backend = AerSimulator() 
circ = transpile(qc, backend) 

result = backend.run(qc,shots=1024).result() #donne par chaque état la fréquence d'apparition  
counts = result.get_counts(qc)

print('RESULT: ',counts)
print('\n Press any key to close')
plot_histogram(counts)
plt.show()
input()