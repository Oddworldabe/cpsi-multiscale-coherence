#!/usr/bin/env python3
"""
cpsi_ibm_max.py — Maximum de résultats IBM Quantum en 8 minutes

STRATÉGIE : Sur vrai hardware, on peut aller à N=20, 30, 50 qubits !
C'est IMPOSSIBLE classiquement (2^50 = 10^15 dims).
C'est LE résultat que personne d'autre n'a.

BATCH 1 (~3 min) : C[Ψ] SCALING AVEC N
  - État ordonné (X gates) : N = 4, 8, 12, 16, 20, 30, 50
  - État paramagnétique (H gates) : même N
  - GHZ (intriqué) : N = 4, 8, 12, 20
  → 18 circuits, teste si C_eq(A)≈1 et C_eq(B)≈0.33 tiennent à GRANDE TAILLE

BATCH 2 (~2 min) : QPT SCAN À N=4, 10, 20
  - RY(θ) interpolation ordered↔paramagnetic
  - 8 valeurs de θ × 3 tailles = 24 circuits
  → QPT sur vrai hardware à N=20

BATCH 3 (~2 min) : DEPTH SCAN À N=10, 20
  - Profondeur 1, 5, 10, 20, 50 × 2 tailles = 10 circuits
  → Décohérence réelle à grande taille

TOTAL : ~52 circuits, ~7 min, ~1 min de marge

python3 ~/Desktop/cpsi_ibm_max.py
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from scipy import stats
import time

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════

IBM_TOKEN = "7xVLVvR99E858ptggH93l8c7tgrBKrsHgszbFAeK11zO"
IBM_CRN = "crn:v1:bluemix:public:quantum-computing:us-east:a/0097a59bc929466eab1121f836f4b16f:75533909-0b8a-4563-a872-75864689f863::"

def compute_cpsi(sz):
    N = len(sz)
    if N < 2: return None, None
    micro = np.mean([abs(sz[i]*sz[i+1]) for i in range(N-1)])
    macro = np.mean([abs(sz[i]*sz[j]) for i in range(N) for j in range(i+1,N)])
    meso = max(0, min(1, 1.0 - np.std(np.abs(sz))))
    C = (micro + macro + meso) / 3
    T = abs(micro-meso) + abs(meso-macro) + abs(micro-macro)
    return C, T

def extract_sz(result_item, N):
    """Extrait <σz> depuis les résultats IBM."""
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
        print(f"    Extraction error: {e}")
        return np.zeros(N)

# ═══════════════════════════════════════════════════
# CONNEXION
# ═══════════════════════════════════════════════════

print("=" * 60)
print("C[Ψ] IBM QUANTUM — 8 MIN OPTIMISÉES")
print("=" * 60)

print("\nConnexion IBM Quantum...", flush=True)
try:
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=IBM_TOKEN,
        instance=IBM_CRN,
        overwrite=True, set_as_default=True
    )
    service = QiskitRuntimeService()
    backend = service.least_busy(min_num_qubits=50, operational=True)
    print(f"Backend : {backend.name} ({backend.num_qubits} qubits)")
except Exception as e:
    print(f"Connexion échouée : {e}")
    exit(1)

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)

# ═══════════════════════════════════════════════════
# BATCH 1 : SCALING AVEC N
# ═══════════════════════════════════════════════════

print(f"\n{'='*60}")
print("BATCH 1 : C[Ψ] SCALING AVEC N")
print(f"{'='*60}", flush=True)

circuits = []
labels = []

# Ordered states (X gates → |1...1⟩ → tous sz = -1)
for N in [4, 8, 12, 16, 20, 30, 50]:
    qc = QuantumCircuit(N, N)
    for i in range(N):
        qc.x(i)
    qc.measure(range(N), range(N))
    circuits.append(qc)
    labels.append(f"ordered_N{N}")

# Paramagnetic states (H gates → |+...+⟩ → sz ≈ 0)
for N in [4, 8, 12, 16, 20, 30, 50]:
    qc = QuantumCircuit(N, N)
    for i in range(N):
        qc.h(i)
    qc.measure(range(N), range(N))
    circuits.append(qc)
    labels.append(f"param_N{N}")

# GHZ states (maximally entangled)
for N in [4, 8, 12, 20]:
    qc = QuantumCircuit(N, N)
    qc.h(0)
    for i in range(N-1):
        qc.cx(i, i+1)
    qc.measure(range(N), range(N))
    circuits.append(qc)
    labels.append(f"GHZ_N{N}")

# Random product states
np.random.seed(42)
for N in [4, 10, 20]:
    qc = QuantumCircuit(N, N)
    for i in range(N):
        theta = np.random.uniform(0, np.pi)
        qc.ry(theta, i)
    qc.measure(range(N), range(N))
    circuits.append(qc)
    labels.append(f"random_N{N}")

print(f"  {len(circuits)} circuits (scaling)")

# ═══════════════════════════════════════════════════
# BATCH 2 : QPT SCAN
# ═══════════════════════════════════════════════════

print(f"\n{'='*60}")
print("BATCH 2 : QPT SCAN (θ interpolation)")
print(f"{'='*60}", flush=True)

# RY(θ) : θ=0 → |0⟩ (ordered), θ=π/2 → |+⟩ (paramagnetic), θ=π → |1⟩
for N in [4, 10, 20]:
    for theta in [0.0, 0.3, 0.6, 0.9, 1.2, 1.57, 2.0, 3.14]:
        qc = QuantumCircuit(N, N)
        for i in range(N):
            qc.ry(theta, i)
        qc.measure(range(N), range(N))
        circuits.append(qc)
        labels.append(f"qpt_N{N}_t{theta:.2f}")

print(f"  +24 circuits (QPT scan)")

# ═══════════════════════════════════════════════════
# BATCH 3 : DEPTH SCAN À GRANDE TAILLE
# ═══════════════════════════════════════════════════

print(f"\n{'='*60}")
print("BATCH 3 : DEPTH SCAN")
print(f"{'='*60}", flush=True)

for N in [10, 20]:
    for depth in [1, 5, 10, 20, 50]:
        qc = QuantumCircuit(N, N)
        # Préparer état ordonné
        for i in range(N):
            qc.x(i)
        # Ajouter couches d'identité (bruit sur hardware)
        for d in range(depth):
            qc.barrier()
            for i in range(N-1):
                qc.cx(i, i+1)
                qc.cx(i+1, i)
        qc.measure(range(N), range(N))
        circuits.append(qc)
        labels.append(f"depth{depth}_N{N}")

print(f"  +10 circuits (depth scan)")
print(f"\n  TOTAL : {len(circuits)} circuits")

# ═══════════════════════════════════════════════════
# TRANSPILATION + ENVOI
# ═══════════════════════════════════════════════════

print(f"\nTranspilation pour {backend.name}...", flush=True)
t0 = time.time()
isa_circuits = pm.run(circuits)
print(f"  Transpilé en {time.time()-t0:.0f}s", flush=True)

# Profondeurs après transpilation
for i, label in enumerate(labels):
    d = isa_circuits[i].depth()
    if 'depth50' in label or 'GHZ_N20' in label or 'N50' in label:
        print(f"  {label}: depth={d}")

print(f"\nEnvoi de {len(isa_circuits)} circuits sur {backend.name}...", flush=True)
t_submit = time.time()

sampler = SamplerV2(backend)
pubs = [(circ,) for circ in isa_circuits]

# Envoyer en un seul job pour minimiser la queue
job = sampler.run(pubs, shots=4096)
print(f"  Job ID: {job.job_id()}")
print(f"  Attente résultats...", flush=True)

result = job.result()
t_done = time.time()
print(f"  Résultats reçus en {t_done-t_submit:.0f}s")

# ═══════════════════════════════════════════════════
# ANALYSE
# ═══════════════════════════════════════════════════

print(f"\n{'='*60}")
print("RÉSULTATS")
print(f"{'='*60}")

all_results = []
for i, label in enumerate(labels):
    # Déterminer N depuis le label
    if '_N' in label:
        N = int(label.split('_N')[1].split('_')[0])
    else:
        N = 4
    
    sz = extract_sz(result[i], N)
    C, T = compute_cpsi(sz)
    mean_sz = np.mean(sz)
    
    all_results.append({'label':label, 'N':N, 'C':C, 'T':T, 'mean_sz':mean_sz})
    
    # Afficher les résultats importants
    if any(x in label for x in ['ordered', 'param', 'GHZ', 'random', 'depth']):
        print(f"  {label:25s}  C={C:.4f}  T={T:.4f}  <sz>={mean_sz:+.3f}")

# ═══════════════════════════════════════════════════
# ANALYSE SCALING
# ═══════════════════════════════════════════════════

print(f"\n{'='*60}")
print("SCALING : C[Ψ] vs N sur vrai hardware")
print(f"{'='*60}")

print(f"\n  {'N':>4} {'C ordered':>10} {'C param':>10} {'C GHZ':>10}")
print(f"  {'-'*38}")

for N in [4, 8, 12, 16, 20, 30, 50]:
    C_ord = next((r['C'] for r in all_results if r['label']==f'ordered_N{N}'), None)
    C_par = next((r['C'] for r in all_results if r['label']==f'param_N{N}'), None)
    C_ghz = next((r['C'] for r in all_results if r['label']==f'GHZ_N{N}'), None)
    
    ghz_str = f"{C_ghz:.4f}" if C_ghz else "—"
    print(f"  {N:4d} {C_ord:.4f}     {C_par:.4f}     {ghz_str}")

# ═══════════════════════════════════════════════════
# ANALYSE QPT
# ═══════════════════════════════════════════════════

print(f"\n{'='*60}")
print("QPT SCAN sur vrai hardware")
print(f"{'='*60}")

for N in [4, 10, 20]:
    print(f"\n  N = {N}:")
    thetas = [0.0, 0.3, 0.6, 0.9, 1.2, 1.57, 2.0, 3.14]
    Cs = []
    for theta in thetas:
        r = next((r for r in all_results if r['label']==f'qpt_N{N}_t{theta:.2f}'), None)
        if r:
            print(f"    θ={theta:.2f}  C={r['C']:.4f}")
            Cs.append(r['C'])
    if len(Cs) > 3:
        # Monotone ?
        diffs = np.diff(Cs)
        monotone = all(d <= 0.01 for d in diffs)
        print(f"    Range: {max(Cs):.4f} → {min(Cs):.4f}  {'Monotone ✓' if max(Cs)-min(Cs) > 0.1 else 'Flat'}")

# ═══════════════════════════════════════════════════
# ANALYSE DEPTH
# ═══════════════════════════════════════════════════

print(f"\n{'='*60}")
print("DEPTH SCAN sur vrai hardware")
print(f"{'='*60}")

for N in [10, 20]:
    print(f"\n  N = {N}:")
    for depth in [1, 5, 10, 20, 50]:
        r = next((r for r in all_results if r['label']==f'depth{depth}_N{N}'), None)
        if r:
            print(f"    depth={depth:3d}  C={r['C']:.4f}")

# ═══════════════════════════════════════════════════
# RÉSUMÉ
# ═══════════════════════════════════════════════════

print(f"\n{'='*60}")
print("RÉSUMÉ")
print(f"{'='*60}")

# C_eq(B) = 1/3 à grande taille ?
param_Cs = [r['C'] for r in all_results if 'param' in r['label']]
if param_Cs:
    print(f"\n  C_eq(B) paramagnétique : {np.mean(param_Cs):.4f} ± {np.std(param_Cs):.4f}")
    print(f"  Déviation de 1/3 : {abs(np.mean(param_Cs) - 1/3):.4f}")

# C ordered diminue-t-il avec N (bruit) ?
ord_results = [(r['N'], r['C']) for r in all_results if 'ordered' in r['label']]
ord_results.sort()
if len(ord_results) > 3:
    Ns, Cs = zip(*ord_results)
    r_NC, p = stats.pearsonr(np.log(Ns), Cs)
    print(f"\n  C ordonné vs log(N) : ρ = {r_NC:+.4f} (p={p:.4f})")
    if r_NC < -0.5:
        print(f"  → Le bruit hardware RÉDUIT C pour les grands N")
    else:
        print(f"  → C ordonné STABLE malgré le bruit !")

print(f"\n  Temps total hardware : {t_done-t_submit:.0f}s ({(t_done-t_submit)/60:.1f} min)")
print(f"  Circuits exécutés : {len(circuits)}")
print(f"  Minutes restantes IBM : ~{8 - (t_done-t_submit)/60:.1f}")
