# =============================================================================
# ALGORITHME DE GROVER POUR LA DÉTECTION MULTI-UTILISATEURS (CAS MULTI-CIBLES)
# =============================================================================
# Objectif : Même principe que algofinalquantum.py, mais on itère sur tous
#            les candidats ayant une probabilité > 0.5, pas seulement le max car ya plusieurs ca==user co en meme temps.
#
# Différence clé : au lieu d'un seul circuit Grover pour le meilleur candidat,
# on lance UN circuit Grover indépendant par candidat détecté.
# =============================================================================

from qiskit import QuantumCircuit, transpile, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
import math
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import UnitaryGate


# =============================================================================
# PARTIE 1 : DÉTECTION CLASSIQUE BAYÉSIENNE (identique à algofinalquantum.py)
# =============================================================================

def calcul_proba_ook():
    """
    Calcule P(αᵢ=1 | Y) pour chaque utilisateur i via le théorème de Bayes.
    
    Modèle OOK (On-Off Keying) en CDMA :
      - Chaque user a un code d'étalement unique c_k ∈ {-1,+1}^M
      - Le signal reçu est Y = Σ_k h_k * α_k * c_k + bruit
      - On calcule la probabilité que chaque user soit l'émetteur
    """

    # Matrice de codes : 10 users × 7 symboles (valeurs ±1)
    C = np.array([
        [-1,  1, -1, -1,  1, -1, -1],   # User 1
        [ 1, -1, -1, -1, -1,  1, -1],   # User 2
        [-1,  1, -1, -1, -1,  1, -1],   # User 3
        [ 1, -1, -1,  1, -1, -1, -1],   # User 4
        [-1, -1,  1, -1, -1, -1,  1],   # User 5
        [-1, -1,  1, -1,  1, -1,  1],   # User 6
        [-1, -1, -1,  1,  1, -1,  1],   # User 7
        [ 1,  1, -1, -1,  1, -1,  1],   # User 8
        [-1, -1,  1,  1,  1,  1,  1],   # User 9
        [-1,  1, -1,  1, -1,  1,  1]    # User 10
    ])

    K, M = C.shape   # K=10, M=7

    p = 0.1           # Probabilité a priori d'activité d'un user
    sigma_n = 1.0     # Écart-type du bruit AWGN
    sigma_sq_proj = M * (sigma_n**2)   # Variance de la projection : M * σ²

    # Vérité terrain : seul l'user 4 (index 3) est actif
    alpha_true = np.zeros(K)
    alpha_true[3] = 1

    h = np.ones(K)  # Canal idéal (gain = 1 pour tous)
    noise = np.random.normal(0, sigma_n, M)  # Bruit gaussien

    # Signal reçu Y = somme des contributions actives + bruit
    Y = np.zeros(M)
    for k in range(K):
        Y += h[k] * alpha_true[k] * C[k]
    Y += noise

    print(f"--- CAS OOK ---")
    print(f"Codebook : {K} users, Longueur {M}")
    print("-" * 95)
    print(f"{'User':<6} | {'Projection':<15} | {'Terme B-A':<15} | {'P(Alpha=1|Y)':<15} | {'Vrai Alpha'}")
    print("-" * 95)

    probas_calculees = []
    for i in range(K):
        # Projection du signal reçu sur le code de l'user i
        y_proj = np.dot(Y, C[i])

        # Terme du log-rapport de vraisemblance (issu de la densité gaussienne)
        B_minus_A = (M**2) - (2 * y_proj * M)
        delta = B_minus_A / (2 * sigma_sq_proj)

        # Probabilité a posteriori via Bayes : P(α=1|Y) = p / [p + (1-p)*exp(δ)]
        try:
            term_exp = np.exp(delta)
            prob_final = p / (p + (1 - p) * term_exp)
        except OverflowError:
            prob_final = 0.0  # exp(δ) → ∞ ⟹ proba → 0

        probas_calculees.append(prob_final)
        status = "✅" if (prob_final > 0.5) == alpha_true[i] else "❌"
        print(f"U{i+1:<5} | {y_proj:15.2f} | {B_minus_A:15.2f} | {prob_final:15.4f} {status} | {alpha_true[i]}")

    return np.array(probas_calculees)


# =============================================================================
# PARTIE 2 : PRÉPARATION DE L'ÉTAT QUANTIQUE INITIAL
# =============================================================================

pi = math.pi
probas_brutes = calcul_proba_ook()

# Remplissage sur 16 états (4 qubits → 2⁴ = 16)
p_complete = np.zeros(16)
p_complete[:10] = probas_brutes
p_complete[10:] = 1e-9  # États sans signification physique → quasi-nuls mias pour éviter les divisions par zéro

# Normalisation pour que Σ p_norm = 1 (exigence des amplitudes quantiques)
p_norm = p_complete / np.sum(p_complete)

# ── DIFFÉRENCE CLÉ avec uen itération.py ──────────────────────────────────
# Ici on identifie TOUS les candidats dont la probabilité dépasse 0.5,
# au lieu de prendre uniquement le maximum.
# permet de traiter le cas où plusieurs users sont actifs simultanément.
candidates = [i for i in range(10) if probas_brutes[i] > 0.5]
print(f"\nCandidats détectés : {[i+1 for i in candidates]} (users 1-based)")

if not candidates:
    print("Aucun candidat trouvé avec prob > 0.5.")
else:

    # =============================================================================
    # PARTIE 3 : DÉFINITION DES PORTES QUANTIQUES (identiques à algofinalquantum.py)
    # =============================================================================

    def oracle_mappage(qc, q, idx):
        """
        Oracle de Grover pour l'index idx.
        Applique une inversion de phase (-1) sur l'état |idx⟩ uniquement.
        
        Construit la matrice diagonale : diag(1,...,1,-1,1,...,1)
        avec le -1 à la position idx.
        """
        matrix = np.eye(16)
        matrix[idx, idx] = -1
        gate = UnitaryGate(matrix, label=f"Oracle_U{idx+1}")
        qc.append(gate, q)

    def reflexion_zero(qc, q):
        """
        Réflexion autour de |0000⟩ : (2|0⟩⟨0| - I).
        
        Partie centrale de la diffusion de Grover.
        Séquence : X → H → MCX → H → X
        Réalise un flip de phase conditionnel sur l'état |0000⟩.
        """
        qc.x(q)
        qc.h(q[3])
        qc.mcx([q[0], q[1], q[2]], q[3])
        qc.h(q[3])
        qc.x(q)

    def build_unitary_from_state(psi):
        """
        Construit une matrice unitaire U telle que U|0⟩ = |ψ⟩.
        
        Utilise la décomposition QR d'une matrice dont la première colonne
        est le vecteur cible psi. La correction de phase assure l'unicité.
        """
        n = len(psi)
        M = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        M[:, 0] = psi
        Q, R = np.linalg.qr(M)
        d = np.diagonal(R)
        phases = d / np.abs(d)
        Q = Q @ np.diag(phases).conj().T
        return Q

    # ── Construction des portes A et A† (communes à tous les candidats) ───────
    # A est construit UNE SEULE FOIS et réutilisé pour chaque candidat.
    # C'est cohérent car l'état initial (distribution des probas) est le même.
    initial_state_vector = np.sqrt(p_norm).astype(complex)  # Amplitudes = √probas
    U_matrix = build_unitary_from_state(initial_state_vector)
    A_gate = UnitaryGate(U_matrix, label="A")
    A_gate_inv = UnitaryGate(U_matrix.conj().T, label="A†")  # Unitaire inverse = transposée conjuguée

    backend = AerSimulator()  # Simulateur quantique partagé

    # =============================================================================
    # PARTIE 4 : BOUCLE SUR CHAQUE CANDIDAT — UN CIRCUIT GROVER PAR USER
    # =============================================================================
    # C'est LA différence structurelle avec une itération.py :
    # on répète le circuit Grover pour chaque candidat détecté séparément.

    for idx_cible in candidates:
        p_target = p_norm[idx_cible]

        # ── Calcul du k optimal pour ce candidat ──────────────────────────────
        # k varie selon la probabilité initiale de chaque candidat
        k_float = (pi / (4 * np.arcsin(np.sqrt(p_target)))) - 0.5
        k = max(1, round(k_float))

        print(f"\n--- Simulation pour User {idx_cible+1} ---")
        print(f"Probabilité initiale de la cible : {p_target:.4f}")
        print(f"Nombre d'itérations optimal : {k_float:.3f} → k = {k}")

        # ── Construction du circuit pour ce candidat ──────────────────────────
        # Un nouveau circuit est créé à chaque itération de la boucle
        q = QuantumRegister(4, 'q')
        c = ClassicalRegister(4, 'c')
        qc = QuantumCircuit(q, c)

        # Étape 1 : Encodage de l'état initial commun
        qc.append(A_gate, q)
        qc.barrier()

        # Étape 2 : k+1 itérations de Grover ciblant idx_cible
        for i in range(k + 1):
            # Oracle : flip de phase sur l'état de l'user courant
            oracle_mappage(qc, q, idx_cible)
            qc.barrier()

            # Diffusion : A† → Réflexion(|0⟩) → A
            # Amplifie l'amplitude de l'état marqué
            qc.append(A_gate_inv, q)
            reflexion_zero(qc, q)
            qc.append(A_gate, q)
            qc.barrier()

        # Mesure finale
        qc.measure(q, c)

        # ── Exécution du circuit ──────────────────────────────────────────────
        print(f"Simulation avec k={k} itération(s)...")
        transpiled_qc = transpile(qc, backend)
        result = backend.run(transpiled_qc, shots=1024).result()
        counts = result.get_counts()

        # ── Lecture du résultat ───────────────────────────────────────────────
        # La chaîne binaire Qiskit est en ordre q3 q2 q1 q0
        # format(idx, '04b') donne la représentation correcte
        cible_str = format(idx_cible, '04b')
        val_cible = counts.get(cible_str, 0)

        print(f"Résultats pour User {idx_cible+1} :")
        print(f"Chaîne binaire cherchée : {cible_str}")
        print(f"Counts pour la cible : {val_cible}/1024 ({(val_cible/10.24):.1f}%)")

        # ── Histogramme par candidat ──────────────────────────────────────────
        # Un graphique est généré pour chaque user candidat
        plot_histogram(counts)
        plt.title(f"Résultat Grover - Cible User {idx_cible+1}")
        plt.show()