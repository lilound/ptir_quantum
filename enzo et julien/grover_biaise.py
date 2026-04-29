import math
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator

# --- 1. Paramètres ---
num_qubits = 3
N = 2 ** num_qubits
raw_probs = np.array([0.05, 0.05, 0.1, 0.3, 0.1, 0.05, 0.3, 0.05])
raw_probs = raw_probs / raw_probs.sum()
amplitudes = np.sqrt(raw_probs)

# --- 3. Préparation de l'état ---
def build_state_prep_circuit(amplitudes):
    n = int(np.log2(len(amplitudes)))
    qc = QuantumCircuit(n, name='A')
    qc.initialize(amplitudes, range(n))
    return qc

A = build_state_prep_circuit(amplitudes)

# --- 4. Oracle (Corrigé pour le bit-ordering) ---
def grover_oracle(marked_states):
    if not isinstance(marked_states, list):
        marked_states = [marked_states]
    num_qubits = len(marked_states[0])
    qc = QuantumCircuit(num_qubits, name='Oracle')
    for target in marked_states:
        # Qiskit est Little Endian : l'état '011' (3) est lu [q0=1, q1=1, q2=0]
        # On inverse la chaîne pour correspondre aux index [0, 1, 2]
        rev_target = target[::-1] 
        zero_inds = [i for i in range(num_qubits) if rev_target[i] == '0']
        
        if zero_inds: qc.x(zero_inds)
        # Correction : Ajout de range(num_qubits) pour spécifier sur quels qubits appliquer la porte
        qc.compose(MCMTGate(ZGate(), num_qubits - 1, 1), range(num_qubits), inplace=True)
        if zero_inds: qc.x(zero_inds)
    return qc

marked_states = ["011"]
oracle = grover_oracle(marked_states)

# --- 5. Diffuseur Biaisé ---
def build_biased_diffuser(A):
    n = A.num_qubits
    qc = QuantumCircuit(n, name='Diffuseur_biaisé')
    qc.compose(A.inverse(), inplace=True)
    qc.x(range(n))
    # Correction : Ajout de range(n)
    qc.compose(MCMTGate(ZGate(), n - 1, 1), range(n), inplace=True)
    qc.x(range(n))
    qc.compose(A, inplace=True)
    return qc

# --- 6. Calcul des itérations ---
marked_indices = [int(s, 2) for s in marked_states]
amplitude_cibles = np.sqrt(sum(raw_probs[i] for i in marked_indices))
optimal_num_iterations_biased = math.floor(math.pi / (4 * math.asin(amplitude_cibles)))

# --- 7. Circuit Complet ---
def build_biased_grover(A, oracle, num_iterations):
    n = A.num_qubits
    diffuser = build_biased_diffuser(A)
    qc = QuantumCircuit(n)
    qc.compose(A, inplace=True)
    for _ in range(num_iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure_all()
    return qc

qc_biased = build_biased_grover(A, oracle, optimal_num_iterations_biased)

# --- 8. Simulation ---
simulator = AerSimulator()
# Décomposition propre pour éviter les erreurs avec les instructions "initialize"
qc_sim = qc_biased.decompose(reps=3) 
counts = simulator.run(qc_sim, shots=4096).result().get_counts()

print(f"Résultats (Cible {marked_states}) : {counts}")
plot_histogram(counts)
plt.show()

# --- 10. Analyse de variation (Correction de la logique de calcul) ---
max_iter = max(optimal_num_iterations_biased * 3, 6)
probs_vs_iter = []

for k in range(max_iter + 1):
    qc_k = build_biased_grover(A, oracle, k).decompose(reps=3)
    counts_k = simulator.run(qc_k, shots=2048).result().get_counts()
    
    # Correction : Utilisation directe de 's' car les clés renvoyées par le simulateur 
    # correspondent déjà à la chaîne recherchée
    prob_target = sum(counts_k.get(s, 0) for s in marked_states) / 2048
    probs_vs_iter.append(prob_target)
    print(f"k={k:2d} -> P(cible) = {prob_target:.4f}")

print(f"\nMeilleur k : {np.argmax(probs_vs_iter)} avec P = {max(probs_vs_iter):.4f}")