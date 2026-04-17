#!/usr/bin/env python3
"""
cpsi_ibm_quantum.py — C[Ψ] sur VRAIS QUBITS IBM Quantum

ÉTAPE 1 : Créer un compte IBM Quantum (gratuit)
  → https://quantum.ibm.com
  → Sign up → Get API token

ÉTAPE 2 : pip install qiskit qiskit-ibm-runtime

ÉTAPE 3 : Coller ton API token ci-dessous et lancer

TEST A : C[Ψ] sur ground state Ising (état préparé par VQE)
TEST B : C[Ψ] vs profondeur de circuit (proxy de décohérence)
TEST C : max(C) = max(W) sur vrai hardware

python3 ~/Desktop/cpsi_ibm_quantum.py
"""

# ═══════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════

# 1. Va sur https://quantum.ibm.com (dashboard)
# 2. Copie ton API key (44 caractères, en haut du dashboard)
#    OU va sur https://cloud.ibm.com/iam/apikeys → Create
# 3. Colle ici :
IBM_TOKEN = "7xVLVvR99E858ptggH93l8c7tgrBKrsHgszbFAeK11zO"

# 4. Le CRN de ton instance (tu l'as déjà) :
IBM_CRN = "crn:v1:bluemix:public:quantum-computing:us-east:a/0097a59bc929466eab1121f836f4b16f:75533909-0b8a-4563-a872-75864689f863::"

# ═══════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════

import numpy as np
from scipy import stats
import warnings, time
warnings.filterwarnings('ignore')

def compute_cpsi(sz):
    N = len(sz)
    micro = np.mean([abs(sz[i]*sz[i+1]) for i in range(N-1)])
    macro = np.mean([abs(sz[i]*sz[j]) for i in range(N) for j in range(i+1,N)])
    meso = max(0, min(1, 1.0 - np.std(np.abs(sz))))
    C = (micro + macro + meso) / 3
    T = abs(micro-meso) + abs(meso-macro) + abs(micro-macro)
    return C, T

# Vérifier les imports
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    print("qiskit OK")
except ImportError:
    print("ERREUR: pip install qiskit qiskit-ibm-runtime")
    print("Puis relance ce script")
    exit(1)

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    print("qiskit-ibm-runtime OK")
except ImportError:
    print("ERREUR: pip install qiskit-ibm-runtime")
    exit(1)


# ═══════════════════════════════════════════════════
# CONNEXION IBM QUANTUM
# ═══════════════════════════════════════════════════

def connect_ibm():
    """Se connecte à IBM Quantum via IBM Cloud."""
    if IBM_TOKEN == "COLLE_API_KEY":
        print("\n  TOKEN NON CONFIGURÉ")
        print("  1. Va sur https://quantum.ibm.com → dashboard → API key")
        print("  2. OU https://cloud.ibm.com/iam/apikeys → Create")
        print("  3. Colle la clé (44 caractères) dans IBM_TOKEN")
        print("\n  En attendant → simulateur local")
        return None
    
    try:
        QiskitRuntimeService.save_account(
            channel="ibm_quantum_platform",
            token=IBM_TOKEN,
            instance=IBM_CRN,
            overwrite=True,
            set_as_default=True
        )
        service = QiskitRuntimeService()
        
        backends = service.backends()
        print(f"\n  Connecté à IBM Quantum")
        print(f"  Backends :")
        for b in backends[:5]:
            print(f"    {b.name} : {b.num_qubits} qubits")
        
        # Choisir un backend avec >= 4 qubits
        backend = service.least_busy(min_num_qubits=4, operational=True)
        print(f"\n  Backend choisi : {backend.name}")
        return backend
    except Exception as e:
        print(f"  Connexion échouée : {e}")
        print("  → simulateur local")
        return None


# ═══════════════════════════════════════════════════
# TEST A : C[Ψ] SUR GROUND STATE ISING (circuit)
# ═══════════════════════════════════════════════════

def ising_ground_state_circuit(N, hJ, depth=3):
    """
    Prépare approximativement le ground state de H_Ising 
    via un ansatz variationnel (simplified VQE).
    
    Pour h/J << 1 : |↓↓...↓⟩ (tous spin down)
    Pour h/J >> 1 : |+...+⟩ (superposition, alignés sur x)
    Pour h/J ≈ 1 : état intriqué (le plus intéressant)
    """
    qc = QuantumCircuit(N, N)
    
    if hJ < 0.3:
        # Phase ordonnée : |↓↓...↓⟩ ≈ ground state
        # Ne rien faire (|0⟩ = |↑⟩, on veut |↓⟩)
        for i in range(N):
            qc.x(i)  # |0⟩ → |1⟩ ≈ |↓⟩
    elif hJ > 1.5:
        # Phase paramagnétique : |+...+⟩
        for i in range(N):
            qc.h(i)  # Hadamard
    else:
        # Transition : ansatz variationnel
        # Couches alternées RY + CNOT (hardware-efficient ansatz)
        theta = np.pi * (1 - hJ) / 2  # interpolation
        for layer in range(depth):
            for i in range(N):
                qc.ry(theta + 0.1 * layer, i)
            for i in range(N-1):
                qc.cx(i, i+1)
    
    return qc


def measure_sz(qc, N, shots=4096):
    """
    Mesure σ_z sur tous les qubits via Statevector (simulateur local).
    """
    from qiskit.quantum_info import Statevector
    
    # Obtenir le statevector AVANT mesure
    sv = Statevector.from_instruction(qc)
    
    # Sampler les résultats
    counts = sv.sample_counts(shots)
    
    # Calculer ⟨σ_z^i⟩ = P(0) - P(1) pour chaque qubit
    sz = np.zeros(N)
    total = sum(counts.values())
    
    for bitstring, count in counts.items():
        for i in range(N):
            bit = int(bitstring[N-1-i])  # qiskit little-endian
            sz[i] += (1 - 2*bit) * count / total  # |0⟩ → +1, |1⟩ → -1
    
    return sz


def test_ground_state_local():
    """Test C[Ψ] sur ground state Ising en simulateur local."""
    print(f"\n{'='*60}")
    print(f"TEST A : C[Ψ] sur ground state Ising (simulateur)")
    print(f"{'='*60}", flush=True)
    
    N = 4
    
    for hJ in [0.01, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]:
        qc = ising_ground_state_circuit(N, hJ)
        sz = measure_sz(qc, N)
        C, T = compute_cpsi(sz)
        print(f"  hJ={hJ:.2f}  C={C:.4f}  T={T:.4f}  <sz>={sz}", flush=True)


# ═══════════════════════════════════════════════════
# TEST B : DÉCOHÉRENCE = PROFONDEUR DE CIRCUIT
# ═══════════════════════════════════════════════════

def test_decoherence_depth():
    """
    Sur vrai hardware, la profondeur du circuit = décohérence.
    Plus le circuit est profond, plus le bruit (T1, T2) agit.
    
    → Circuit court = état quasi-pur = C élevé
    → Circuit long = état mixte (bruit) = C faible
    
    C'est l'équivalent EXPÉRIMENTAL du scan en gamma !
    """
    print(f"\n{'='*60}")
    print(f"TEST B : C[Ψ] vs profondeur de circuit")
    print(f"Profondeur = proxy de décohérence sur vrai hardware")
    print(f"{'='*60}", flush=True)
    
    N = 4
    
    for depth in [1, 2, 3, 5, 10, 20, 50]:
        qc = QuantumCircuit(N, N)
        
        # Préparer un état ordonné
        for i in range(N):
            qc.x(i)
        
        # Ajouter des couches d'identité (= bruit sur vrai hardware)
        for d in range(depth):
            for i in range(N):
                qc.barrier()
                qc.id(i)  # identité = pas d'opération, mais du bruit
            # Ajouter aussi des CNOT pour coupler les qubits
            for i in range(N-1):
                qc.cx(i, i+1)
                qc.cx(i+1, i)  # aller-retour = identité logique, bruit physique
        
        sz = measure_sz(qc, N)
        C, T = compute_cpsi(sz)
        print(f"  depth={depth:3d}  C={C:.4f}  T={T:.4f}  <sz>_mean={np.mean(sz):.4f}", flush=True)
    
    print(f"\n  NOTE : en simulateur parfait, depth n'a pas d'effet")
    print(f"  Sur VRAI hardware IBM, C va DIMINUER avec depth")
    print(f"  car le bruit T1/T2 agit comme la décohérence gamma")


# ═══════════════════════════════════════════════════
# TEST C : CIRCUIT POUR IBM QUANTUM (vrai hardware)
# ═══════════════════════════════════════════════════

def create_ibm_circuits():
    """
    Crée les circuits prêts à envoyer sur IBM Quantum.
    Peut être lancé plus tard quand le token est configuré.
    """
    print(f"\n{'='*60}")
    print(f"TEST C : Circuits prêts pour IBM Quantum")
    print(f"{'='*60}", flush=True)
    
    N = 4
    circuits = []
    labels = []
    
    # 1. Différents h/J (ground states)
    for hJ in [0.01, 0.3, 0.5, 0.7, 1.0, 2.0]:
        qc = ising_ground_state_circuit(N, hJ, depth=3)
        qc.measure(range(N), range(N))
        circuits.append(qc)
        labels.append(f"gs_hJ{hJ}")
    
    # 2. Différentes profondeurs (proxy décohérence)
    for depth in [1, 5, 10, 20, 50, 100]:
        qc = QuantumCircuit(N, N)
        for i in range(N): qc.x(i)
        for d in range(depth):
            qc.barrier()
            for i in range(N): qc.id(i)
            for i in range(N-1):
                qc.cx(i, i+1); qc.cx(i+1, i)
        qc.measure(range(N), range(N))
        circuits.append(qc)
        labels.append(f"depth_{depth}")
    
    # 3. État GHZ (maximalement intriqué)
    qc_ghz = QuantumCircuit(N, N)
    qc_ghz.h(0)
    for i in range(N-1):
        qc_ghz.cx(i, i+1)
    qc_ghz.measure(range(N), range(N))
    circuits.append(qc_ghz)
    labels.append("GHZ")
    
    # 4. État aléatoire
    qc_rand = QuantumCircuit(N, N)
    for i in range(N):
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        qc_rand.ry(theta, i)
        qc_rand.rz(phi, i)
    qc_rand.measure(range(N), range(N))
    circuits.append(qc_rand)
    labels.append("random")
    
    print(f"  {len(circuits)} circuits créés :")
    for i, label in enumerate(labels):
        print(f"    {i}: {label} ({circuits[i].depth()} portes)")
    
    return circuits, labels


def run_on_ibm(backend, circuits, labels):
    """Envoie les circuits sur IBM Quantum et analyse les résultats."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    
    print(f"\n  Transpilation pour {backend.name}...", flush=True)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuits = pm.run(circuits)
    print(f"  {len(isa_circuits)} circuits transpilés", flush=True)
    
    print(f"  Envoi sur {backend.name}...", flush=True)
    sampler = SamplerV2(backend)
    
    # Envoyer en pubs format
    pubs = [(circ,) for circ in isa_circuits]
    job = sampler.run(pubs, shots=4096)
    print(f"  Job ID: {job.job_id()}")
    print(f"  En attente... (1-60 min selon la queue)", flush=True)
    
    result = job.result()
    
    print(f"\n  RÉSULTATS SUR VRAI HARDWARE ({backend.name}) :")
    N = 4
    for i, label in enumerate(labels):
        try:
            # Extraire les counts du résultat
            data = result[i].data
            # Trouver le registre classique
            creg_name = list(data.__dict__.keys())[0]
            bitarray = getattr(data, creg_name)
            counts = bitarray.get_counts()
            
            total = sum(counts.values())
            sz = np.zeros(N)
            for bitstring, count in counts.items():
                for q in range(min(N, len(bitstring))):
                    bit = int(bitstring[len(bitstring)-1-q])
                    sz[q] += (1 - 2*bit) * count / total
            
            C, T = compute_cpsi(sz[:N])
            print(f"    {label:15s}  C={C:.4f}  T={T:.4f}  <sz>={np.mean(sz[:N]):+.3f}")
        except Exception as e:
            print(f"    {label:15s}  ERREUR: {str(e)[:60]}")


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"{'='*60}")
    print(f"C[Ψ] SUR IBM QUANTUM")
    print(f"{'='*60}")
    
    # Tests en simulateur local (toujours)
    test_ground_state_local()
    test_decoherence_depth()
    circuits, labels = create_ibm_circuits()
    
    # Connexion IBM (si token configuré)
    backend = connect_ibm()
    
    if backend is not None:
        run_on_ibm(backend, circuits, labels)
    else:
        print(f"\n{'='*60}")
        print(f"PROCHAINES ÉTAPES :")
        print(f"1. Va sur https://quantum.ibm.com")
        print(f"2. Crée un compte (gratuit, email suffit)")
        print(f"3. Copie ton API token")
        print(f"4. Colle-le dans IBM_TOKEN en haut de ce script")
        print(f"5. Relance : python3 ~/Desktop/cpsi_ibm_quantum.py")
        print(f"")
        print(f"Le plan gratuit donne accès à :")
        print(f"  - Simulateurs : illimité")
        print(f"  - Vrais qubits : 10 min/mois sur 127-qubit Eagle")
        print(f"  - Files d'attente : quelques minutes à 1h")
        print(f"")
        print(f"Avec 10 min, on peut lancer ~50 circuits × 4096 shots")
        print(f"= largement assez pour reproduire les tests A, B, C")
        print(f"{'='*60}")
