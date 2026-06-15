# =============================================================================
# ALGORITHME DE GROVER POUR LA DÉTECTION D'UTILISATEUR ACTIF (CAS 1 CIBLE)
# =============================================================================
# Objectif : Détecter quel utilisateur est actif dans un réseau,
#            en combinant une détection classique bayésienne (OOK) et
#            l'algorithme quantique de Grover pour amplifier la probabilité.
# =============================================================================

from qiskit import QuantumCircuit, transpile, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
import math
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import UnitaryGate


# =============================================================================
# PARTIE 1 : DÉTECTION CLASSIQUE BAYÉSIENNE (OOK = On-Off Keying)
# =============================================================================

def calcul_proba_ook():
    """
    Calcule la probabilité a posteriori P(αᵢ=1 | Y) pour chaque utilisateur i,
    c'est-à-dire : étant donné le signal reçu Y, quelle est la probabilité
    que l'utilisateur i soit bien actif ?

    Modèle :
      - K = 10 utilisateurs, chacun avec un code d'étalement de longueur M = 7
      - α_k ∈ {0,1} indique si l'user k est actif (modulation OOK)
      - Le signal reçu est Y = Σ h_k * α_k * c_k + bruit
      - Le bruit est gaussien de variance σ_n² = 1
    """

    # ── Matrice de codes d'étalement ──────────────────────────────────────────
    # Chaque ligne = code d'un utilisateur (valeurs ±1, type Hadamard/CDMA)
    # Dimension : K=10 users × M=7 symboles
    C = np.array([
        [-1,  1, -1, -1,  1, -1, -1],   # Code user 1
        [ 1, -1, -1, -1, -1,  1, -1],   # Code user 2
        [-1,  1, -1, -1, -1,  1, -1],   # Code user 3
        [ 1, -1, -1,  1, -1, -1, -1],   # Code user 4
        [-1, -1,  1, -1, -1, -1,  1],   # Code user 5
        [-1, -1,  1, -1,  1, -1,  1],   # Code user 6
        [-1, -1, -1,  1,  1, -1,  1],   # Code user 7
        [ 1,  1, -1, -1,  1, -1,  1],   # Code user 8
        [-1, -1,  1,  1,  1,  1,  1],   # Code user 9
        [-1,  1, -1,  1, -1,  1,  1]    # Code user 10
    ])

    K, M = C.shape  # K=10 users, M=7 longueur des codes

    # ── Paramètres du modèle ──────────────────────────────────────────────────
    p = 0.1           # Probabilité a priori qu'un user soit actif
    sigma_n = 1.0     # Écart-type du bruit gaussien
    # Variance de la projection du bruit sur un code (||c||² * σ²)
    sigma_sq_proj = M * (sigma_n**2)  # = 7 * 1 = 7

    # ── Simulation du canal ───────────────────────────────────────────────────
    # alpha_true : vecteur d'activité réel (vérité terrain)
    alpha_true = np.zeros(K)
    alpha_true[3] = 1  # Seul l'utilisateur 4 (index 3) est actif

    h = np.ones(K)  # Coefficients de canal (tous à 1 = pas de fading ici)

    # Bruit additif gaussien (AWGN)
    noise = np.random.normal(0, sigma_n, M)

    # Signal reçu : Y = Σ_k h_k * α_k * c_k + bruit
    Y = np.zeros(M)
    for k in range(K):
        Y += h[k] * alpha_true[k] * C[k]
    Y += noise

    print(f"--- CAS OOK ---")
    print(f"Codebook : {K} users, Longueur {M}")
    print("-" * 95)
    print(f"{'User':<6} | {'Projection':<15} | {'Terme B-A':<15} | {'P(Alpha=1|Y)':<15} | {'Vrai Alpha'}")
    print("-" * 95)

    # ── Calcul des probabilités a posteriori ──────────────────────────────────
    probas_calculees = []
    for i in range(K):
        # Projection du signal reçu sur le code de l'user i : <Y, c_i>
        y_proj = np.dot(Y, C[i])

        # Terme issu de la règle de Bayes (log-rapport de vraisemblance gaussien)
        # B - A = ||Y - c_i||² - ||Y||² (simplifié) = M² - 2*<Y,c_i>*M
        B_minus_A = (M**2) - (2 * y_proj * M)

        # δ = (B-A) / (2 * σ²_proj) : normalisation par la variance
        delta = B_minus_A / (2 * sigma_sq_proj)

        # P(α_i=1 | Y) = p / [p + (1-p) * exp(δ)]
        # Forme compacte du théorème de Bayes pour l'hypothèse binaire
        try:
            term_exp = np.exp(delta)
            prob_final = p / (p + (1 - p) * term_exp)
        except OverflowError:
            # Si δ est très grand, exp(δ) → ∞, donc P → 0
            prob_final = 0.0

        probas_calculees.append(prob_final)
        status = "✅" if (prob_final > 0.5) == alpha_true[i] else "❌"
        print(f"U{i+1:<5} | {y_proj:15.2f} | {B_minus_A:15.2f} | {prob_final:15.4f} {status} | {alpha_true[i]}")

    return np.array(probas_calculees)


# =============================================================================
# PARTIE 2 : PRÉPARATION DE L'ÉTAT QUANTIQUE INITIAL
# =============================================================================

pi = math.pi

# Calcul des probabilités classiques pour les 10 users
probas_brutes = calcul_proba_ook()

# On travaille avec 4 qubits → 2⁴ = 16 états possibles
# On mappe les 10 users sur les 10 premiers états, les 6 restants sont quasi-nuls
p_complete = np.zeros(16)
p_complete[:10] = probas_brutes
p_complete[10:] = 1e-9  # Valeur quasi-nulle (évite division par 0 à la normalisation)

# Normalisation : les probabilités doivent sommer à 1 (règle des amplitudes quantiques)
p_norm = p_complete / np.sum(p_complete)

# ── Sélection de la cible : l'utilisateur avec la probabilité maximale ────────
# C'est LA différence avec plsiteration.py : ici on ne cible QU'UN seul user
index_cible = np.argmax(p_norm)
p_target = p_norm[index_cible]

print(f"\nL'utilisateur détecté est l'INDEX : {index_cible}")
print(f"Probabilité initiale de la cible : {p_target:.4f}")


# =============================================================================
# PARTIE 3 : CALCUL DU NOMBRE D'ITÉRATIONS OPTIMAL DE GROVER
# =============================================================================

# Formule exacte de Grover pour maximiser P(mesure = cible) :
# k_opt = π / (4 * arcsin(√p)) - 1/2
# Avec p = probabilité initiale de la cible dans la superposition
k_float = (pi / (4 * np.arcsin(np.sqrt(p_target)))) - 0.5
k = max(1, round(k_float))  # Au minimum 1 itération
print(f"Nombre d'itérations optimal : {k_float:.3f} → k = {k}")


# =============================================================================
# PARTIE 4 : CONSTRUCTION DES PORTES QUANTIQUES
# =============================================================================

def oracle_mappage(qc, q, idx):
    """
    Oracle de Grover : inverse la phase (-1) de l'état quantique |idx⟩.
    
    Fonctionnement :
      - Construit la matrice identité 16×16
      - Remplace la diagonale [idx, idx] par -1
      - Applique cette transformation unitaire au circuit
    
    Effet : |idx⟩ → -|idx⟩, tous les autres états inchangés
    C'est le "marquage" de la solution cherchée.
    """
    matrix = np.eye(16)
    matrix[idx, idx] = -1  # Flip de phase sur l'état cible uniquement
    gate = UnitaryGate(matrix, label=f"Oracle_U{idx+1}")
    qc.append(gate, q)


def reflexion_zero(qc, q):
    """
    Réflexion autour de l'état |0000⟩ : opérateur (2|0⟩⟨0| - I).
    
    C'est la deuxième partie de la diffusion de Grover.
    Combiné entre  A et A†, cela réalise la réflexion autour de l'état moyen,
    ce qui amplifie l'amplitude de l'état marqué et ramène les autres états vers 0.
    
    Implémentation :
      1. X sur tous les qubits : |0⟩ → |1⟩ (pour viser |1111⟩ avec le CNOT)
      2. H sur q[3] : prépare la porte de phase
      3. Toffoli multi-contrôlé (MCX) : flip conditionnel sur |1111⟩
      4. H et X inverses pour défaire la transformation
    """
    qc.x(q)              # NOT sur chaque qubit
    qc.h(q[3])           # Hadamard sur le qubit cible
    qc.mcx([q[0], q[1], q[2]], q[3])  # Toffoli 3-contrôles → flip si tous à |1⟩
    qc.h(q[3])           # Hadamard inverse
    qc.x(q)              # NOT inverse


def build_unitary_from_state(psi):
    """
    Construit une matrice unitaire U telle que U|0⟩ = |ψ⟩.
    
    Méthode : décomposition QR d'une matrice aléatoire dont la première
    colonne est forcée à |ψ⟩. La décomposition QR garantit l'orthonormalité
    (donc l'unitarité) tout en préservant la première colonne.
    
    Args:
        psi : vecteur d'amplitudes normalisé (√probabilités)
    Returns:
        Q : matrice unitaire 16×16
    """
    n = len(psi)
    # Matrice aléatoire complexe
    M = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    # On force la première colonne = état cible
    M[:, 0] = psi
    # Décomposition QR pour orthonormaliser
    Q, R = np.linalg.qr(M)
    # Correction de phase pour que Q soit bien une matrice unitaire canonique
    d = np.diagonal(R)
    phases = d / np.abs(d)
    Q = Q @ np.diag(phases).conj().T
    return Q


# ── Construction de l'opérateur A (encodage de l'état initial) ───────────────
# Les amplitudes quantiques sont les racines carrées des probabilités
# (car la mesure donne P = |amplitude|²)
initial_state_vector = np.sqrt(p_norm).astype(complex)

# A : prépare l'état initial depuis |0000⟩
U_matrix = build_unitary_from_state(initial_state_vector)
A_gate = UnitaryGate(U_matrix, label="A")

# A† (A-dagger) : inverse de A, utilisé dans la diffusion de Grover
A_gate_inv = UnitaryGate(U_matrix.conj().T, label="A†")


# =============================================================================
# PARTIE 5 : CONSTRUCTION ET EXÉCUTION DU CIRCUIT QUANTIQUE
# =============================================================================

backend = AerSimulator()  # Simulateur quantique classique d'Aer

# Registres quantique (4 qubits) et classique (4 bits de mesure)
q = QuantumRegister(4, 'q')
c = ClassicalRegister(4, 'c')
qc = QuantumCircuit(q, c)

# ── Étape 1 : Encodage de l'état initial ─────────────────────────────────────
# Applique A pour transformer |0000⟩ en la superposition pondérée par les probas
qc.append(A_gate, q)
qc.barrier()  # Séparateur visuel dans le circuit

# ── Étape 2 : Itérations de Grover ───────────────────────────────────────────
# Chaque itération = Oracle + Diffusion
# L'amplitude de l'état cible augmente à chaque itération
for i in range(k + 1):
    # 2a. Oracle : marque l'état cible par un flip de phase
    oracle_mappage(qc, q, index_cible)
    qc.barrier()# Séparateur visuel dans le circuit (pas )visible dans l'exécution, mais utile pour la lecture du circuit)
    
    # 2b. Diffusion de Grover = A† · Réflexion(|0⟩) · A
    # Réflexion autour de l'état initial (pas de |0⟩ uniforme, mais de |ψ₀⟩)
    qc.append(A_gate_inv, q)   # Ramène vers la base |0⟩
    reflexion_zero(qc, q)      # Réflexion autour de |0000⟩
    qc.append(A_gate, q)       # Revient dans la base des probas
    qc.barrier()

# ── Étape 3 : Mesure ─────────────────────────────────────────────────────────
qc.measure(q, c)


# =============================================================================
# PARTIE 6 : RÉSULTATS
# =============================================================================

print(f"Simulation avec k={k} itération(s)...")

# Transpilation : adaptation du circuit à l'architecture du simulateur car on utilise une matrice16x16 
transpiled_qc = transpile(qc, backend)

# Exécution sur 1024 shots (répétitions de la mesure quantique)
result = backend.run(transpiled_qc, shots=1024).result()
counts = result.get_counts()  # Dictionnaire {état_binaire: nombre_de_fois_mesuré}

# ── Lecture du résultat ───────────────────────────────────────────────────────
# littlenedian q3 q2 q1 q0 (ordre inversé)
# Donc l'index N correspond à la chaîne format(N, '04b') en lecture standard
cible_str = format(index_cible, '04b')  # Ex : index 3 → "0011"
val_cible = counts.get(cible_str, 0)    # Nombre de fois où la cible a été mesurée

print("\nRESULTATS FINAUX :")
print(f"Chaîne binaire cherchée pour l'index {index_cible} : {cible_str}")
print(f"user{index_cible+1} détecté")
print(f"Counts pour la cible : {val_cible}/1024 ({(val_cible/10.24):.1f}%)")

# ── Histogramme des mesures ───────────────────────────────────────────────────
# Idéalement, la barre de la cible devrait être dominante (~100% avec k optimal) 
# or plus on a de cnadidats plus on a de chances d'avoir des erreurs de mesure et 
# amplitudes non idéales, donc la barre des cible peuvt être moins dominante.
plot_histogram(counts)
plt.title(f"Résultat Grover - Cible User {index_cible+1}")
plt.show()