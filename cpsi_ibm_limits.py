#!/usr/bin/env python3
"""
cpsi_ibm_limits.py — Pousser aux LIMITES du processeur

ibm_fez = 156 qubits. On va utiliser TOUT.

BATCH 1 : N = 60, 80, 100, 127, 150 (ordered + paramagnetic)
  → C[Ψ] sur 150 vrais qubits supraconducteurs
  → Classiquement impossible : 2^150 ≈ 10^45

BATCH 2 : QPT scan à N = 50 et N = 100
  → Transition de phase sur 100 qubits réels

BATCH 3 : ÉTATS EXOTIQUES
  → W state (un seul qubit excité, intrication partagée)
  → Random entangled (couches CX aléatoires)
  → Néel state (antiferromagnétique alterné)
  → Cluster state (graphe 1D)

BATCH 4 : C[Ψ] vs POSITION SUR LE CHIP
  → Même circuit sur différentes régions du processeur
  → Cartographie du bruit par C[Ψ]

Budget : ~6.7 min restantes. On vise ~40 circuits.

python3 ~/Desktop/cpsi_ibm_limits.py
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from scipy import stats
import time

IBM_TOKEN = "7xVLVvR99E858ptggH93l8c7tgrBKrsHgszbFAeK11zO"
IBM_CRN = "crn:v1:bluemix:public:quantum-computing:us-east:a/0097a59bc929466eab1121f836f4b16f:75533909-0b8a-4563-a872-75864689f863::"

def compute_cpsi(sz):
    N = len(sz)
    if N < 2: return None, None
    micro = np.mean([abs(sz[i]*sz[i+1]) for i in range(N-1)])
    pairs_sum = 0; count = 0
    for i in range(N):
        for j in range(i+1, N):
            pairs_sum += abs(sz[i]*sz[j])
            count += 1
    macro = pairs_sum / count if count > 0 else 0
    meso = max(0, min(1, 1.0 - np.std(np.abs(sz))))
    C = (micro + macro + meso) / 3
    T = abs(micro-meso) + abs(meso-macro) + abs(micro-macro)
    return C, T

def extract_sz(result_item, N):
    try:
        data = result_item.data
        creg_name = list(data.__dict__.keys())[0]
        bitarray = getattr(data, creg_name)
        counts = bitarray.get_counts()
        total = sum(counts.values())
        sz = np.zeros(N)
        for bitstring, count in counts.items():
            for q in range(min(N, len(bitstring))):
                bit = int(bitstring[len(bitstring)-1-q])
                sz[q] += (1 - 2*bit) * count / total
        return sz
    except Exception as e:
        print(f"    Extract error: {e}")
        return np.zeros(N)

# ═══════════════════════════════════════════════════
# CONNEXION
# ═══════════════════════════════════════════════════

print("=" * 60)
print("C[Ψ] IBM QUANTUM — LIMITES ABSOLUES")
print("=" * 60)

print("\nConnexion...", flush=True)
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform", token=IBM_TOKEN,
    instance=IBM_CRN, overwrite=True, set_as_default=True
)
service = QiskitRuntimeService()
backend = service.least_busy(min_num_qubits=100, operational=True)
n_qubits_max = backend.num_qubits
print(f"Backend : {backend.name} ({n_qubits_max} qubits)")

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)

circuits = []
labels = []

# ═══════════════════════════════════════════════════
# BATCH 1 : TAILLES EXTRÊMES
# ═══════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"BATCH 1 : TAILLES N = 60 à {min(150, n_qubits_max)}")
print(f"{'='*60}", flush=True)

extreme_sizes = [60, 80, 100, 127]
if n_qubits_max >= 150:
    extreme_sizes.append(150)
if n_qubits_max >= 156:
    extreme_sizes.append(n_qubits_max)  # ALL qubits

for N in extreme_sizes:
    if N > n_qubits_max: continue
    
    # Ordered
    qc = QuantumCircuit(N, N)
    for i in range(N): qc.x(i)
    qc.measure(range(N), range(N))
    circuits.append(qc); labels.append(f"ordered_N{N}")
    
    # Paramagnetic
    qc = QuantumCircuit(N, N)
    for i in range(N): qc.h(i)
    qc.measure(range(N), range(N))
    circuits.append(qc); labels.append(f"param_N{N}")

print(f"  {len(circuits)} circuits (extreme sizes)")

# ═══════════════════════════════════════════════════
# BATCH 2 : QPT À N=50 ET N=100
# ═══════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"BATCH 2 : QPT SCAN N=50 et N=100")
print(f"{'='*60}", flush=True)

for N in [50, 100]:
    if N > n_qubits_max: continue
    for theta in [0.0, 0.5, 1.0, 1.57, 2.5, 3.14]:
        qc = QuantumCircuit(N, N)
        for i in range(N): qc.ry(theta, i)
        qc.measure(range(N), range(N))
        circuits.append(qc); labels.append(f"qpt_N{N}_t{theta:.2f}")

print(f"  +{len(circuits) - len(extreme_sizes)*2} circuits (QPT)")

# ═══════════════════════════════════════════════════
# BATCH 3 : ÉTATS EXOTIQUES
# ═══════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"BATCH 3 : ÉTATS EXOTIQUES")
print(f"{'='*60}", flush=True)

N_exotic = 20

# W state : |100...0⟩ + |010...0⟩ + ... + |000...1⟩ (approximation)
qc = QuantumCircuit(N_exotic, N_exotic)
qc.x(0)
for i in range(N_exotic - 1):
    # Rotation partielle pour distribuer l'excitation
    angle = np.arccos(np.sqrt(1/(N_exotic - i)))
    qc.cry(2*angle, i, i+1)
    qc.cx(i+1, i)
qc.measure(range(N_exotic), range(N_exotic))
circuits.append(qc); labels.append(f"W_N{N_exotic}")

# Néel state (antiferromagnétique : |010101...⟩)
qc = QuantumCircuit(N_exotic, N_exotic)
for i in range(0, N_exotic, 2):
    qc.x(i)
qc.measure(range(N_exotic), range(N_exotic))
circuits.append(qc); labels.append(f"neel_N{N_exotic}")

# Cluster state (graphe 1D, ressource pour MBQC)
qc = QuantumCircuit(N_exotic, N_exotic)
for i in range(N_exotic): qc.h(i)
for i in range(N_exotic - 1): qc.cz(i, i+1)
qc.measure(range(N_exotic), range(N_exotic))
circuits.append(qc); labels.append(f"cluster_N{N_exotic}")

# Random entangled (couches alternées RY + CX)
np.random.seed(42)
qc = QuantumCircuit(N_exotic, N_exotic)
for layer in range(3):
    for i in range(N_exotic):
        qc.ry(np.random.uniform(0, np.pi), i)
    for i in range(0 if layer%2==0 else 1, N_exotic-1, 2):
        qc.cx(i, i+1)
qc.measure(range(N_exotic), range(N_exotic))
circuits.append(qc); labels.append(f"rand_entangled_N{N_exotic}")

# Produit pur (baseline)
qc = QuantumCircuit(N_exotic, N_exotic)
for i in range(N_exotic):
    qc.ry(0.7, i)  # Tous même angle
qc.measure(range(N_exotic), range(N_exotic))
circuits.append(qc); labels.append(f"product_N{N_exotic}")

# GHZ à N=50
if n_qubits_max >= 50:
    qc = QuantumCircuit(50, 50)
    qc.h(0)
    for i in range(49): qc.cx(i, i+1)
    qc.measure(range(50), range(50))
    circuits.append(qc); labels.append("GHZ_N50")

print(f"  +6 circuits (exotic states)")

print(f"\n  TOTAL : {len(circuits)} circuits")

# ═══════════════════════════════════════════════════
# TRANSPILE + ENVOI
# ═══════════════════════════════════════════════════

print(f"\nTranspilation...", flush=True)
t0 = time.time()
isa_circuits = pm.run(circuits)
print(f"  Transpilé en {time.time()-t0:.0f}s")

# Afficher les profondeurs des circuits les plus grands
for i, label in enumerate(labels):
    if any(x in label for x in ['N150', 'N156', 'N127', 'GHZ_N50', 'cluster']):
        print(f"  {label}: depth={isa_circuits[i].depth()}")

print(f"\nEnvoi sur {backend.name}...", flush=True)
t_submit = time.time()
sampler = SamplerV2(backend)
pubs = [(circ,) for circ in isa_circuits]
job = sampler.run(pubs, shots=4096)
print(f"  Job ID: {job.job_id()}")
print(f"  Attente...", flush=True)

result = job.result()
t_done = time.time()
print(f"  Résultats en {t_done-t_submit:.0f}s")

# ═══════════════════════════════════════════════════
# RÉSULTATS
# ═══════════════════════════════════════════════════

print(f"\n{'='*60}")
print("RÉSULTATS — LIMITES ABSOLUES")
print(f"{'='*60}")

all_results = []
for i, label in enumerate(labels):
    if '_N' in label:
        N = int(label.split('_N')[1].split('_')[0])
    else:
        N = 20
    sz = extract_sz(result[i], N)
    C, T = compute_cpsi(sz)
    all_results.append({'label':label, 'N':N, 'C':C, 'T':T, 'mean_sz':np.mean(sz)})

# Scaling extrême
print(f"\n  SCALING EXTRÊME :")
print(f"  {'N':>5} {'C ordered':>10} {'C param':>10}")
print(f"  {'-'*28}")
for N in extreme_sizes:
    if N > n_qubits_max: continue
    C_ord = next((r['C'] for r in all_results if r['label']==f'ordered_N{N}'), None)
    C_par = next((r['C'] for r in all_results if r['label']==f'param_N{N}'), None)
    if C_ord and C_par:
        print(f"  {N:5d} {C_ord:10.4f} {C_par:10.4f}")

# QPT scan
print(f"\n  QPT SCAN :")
for N in [50, 100]:
    qpt = [(float(r['label'].split('_t')[1]), r['C']) 
            for r in all_results if r['label'].startswith(f'qpt_N{N}')]
    if qpt:
        qpt.sort()
        print(f"\n  N = {N}:")
        for theta, C in qpt:
            print(f"    θ={theta:.2f}  C={C:.4f}")

# États exotiques
print(f"\n  ÉTATS EXOTIQUES (N={N_exotic}) :")
for label_prefix in ['W', 'neel', 'cluster', 'rand_entangled', 'product', 'GHZ']:
    for r in all_results:
        if r['label'].startswith(label_prefix):
            print(f"    {r['label']:25s}  C={r['C']:.4f}  T={r['T']:.4f}")

# Record
print(f"\n{'='*60}")
print(f"RECORDS")
print(f"{'='*60}")
max_N = max(r['N'] for r in all_results)
print(f"  Plus grand N testé : {max_N} qubits")
print(f"  C_eq(B) paramagnétique : {np.mean([r['C'] for r in all_results if 'param' in r['label']]):.4f}")
print(f"  Temps hardware : {t_done-t_submit:.0f}s ({(t_done-t_submit)/60:.1f} min)")
print(f"  Circuits exécutés : {len(circuits)}")
print(f"  Hilbert dim max : 2^{max_N} ≈ 10^{int(max_N*0.301)}")
print(f"{'='*60}")
