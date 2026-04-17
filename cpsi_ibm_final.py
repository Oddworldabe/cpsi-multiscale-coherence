#!/usr/bin/env python3
"""
cpsi_ibm_final.py — Dernier batch, 5 min restantes

PRIORITÉ 1 : BARRES D'ERREUR sur N=156 (5 répétitions)
  → Sans ça, un reviewer dit "one shot, not reproducible"
  → C_eq(B) = 0.323 ± ? et C_ordered = 0.887 ± ?

PRIORITÉ 2 : QPT FINE à N=50 (20 points)
  → Courbe publiable avec résolution suffisante

PRIORITÉ 3 : ÉVOLUTION TEMPORELLE TROTTER (N=10)
  → Trotterized Ising dynamics = vraie simulation quantique
  → Mesurer C(t) pendant l'évolution sous H_Ising

python3 ~/Desktop/cpsi_ibm_final.py
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
            pairs_sum += abs(sz[i]*sz[j]); count += 1
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
    except:
        return np.zeros(N)

# ═══════════════════════════════════════════════════
print("=" * 60)
print("C[Ψ] IBM — BATCH FINAL (5 min restantes)")
print("=" * 60)

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform", token=IBM_TOKEN,
    instance=IBM_CRN, overwrite=True, set_as_default=True
)
service = QiskitRuntimeService()
backend = service.least_busy(min_num_qubits=100, operational=True)
print(f"Backend : {backend.name} ({backend.num_qubits} qubits)")
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)

circuits = []
labels = []

# ═══════════════════════════════════════════════════
# P1 : BARRES D'ERREUR (5 répétitions × 4 circuits = 20)
# ═══════════════════════════════════════════════════

print(f"\nP1 : BARRES D'ERREUR (5 répétitions)")

for rep in range(5):
    # N=156 ordered
    N = backend.num_qubits
    qc = QuantumCircuit(N, N)
    for i in range(N): qc.x(i)
    qc.measure(range(N), range(N))
    circuits.append(qc); labels.append(f"ord156_r{rep}")
    
    # N=156 paramagnetic
    qc = QuantumCircuit(N, N)
    for i in range(N): qc.h(i)
    qc.measure(range(N), range(N))
    circuits.append(qc); labels.append(f"par156_r{rep}")
    
    # N=50 ordered
    qc = QuantumCircuit(50, 50)
    for i in range(50): qc.x(i)
    qc.measure(range(50), range(50))
    circuits.append(qc); labels.append(f"ord50_r{rep}")
    
    # N=50 paramagnetic
    qc = QuantumCircuit(50, 50)
    for i in range(50): qc.h(i)
    qc.measure(range(50), range(50))
    circuits.append(qc); labels.append(f"par50_r{rep}")

print(f"  20 circuits (error bars)")

# ═══════════════════════════════════════════════════
# P2 : QPT FINE à N=50 (20 points)
# ═══════════════════════════════════════════════════

print(f"\nP2 : QPT FINE N=50 (20 points)")

thetas = np.linspace(0, np.pi, 20)
for theta in thetas:
    qc = QuantumCircuit(50, 50)
    for i in range(50): qc.ry(theta, i)
    qc.measure(range(50), range(50))
    circuits.append(qc); labels.append(f"qptfine_t{theta:.3f}")

print(f"  +20 circuits (fine QPT)")

# ═══════════════════════════════════════════════════
# P3 : TROTTER DYNAMICS (N=10, 8 pas de temps)
# ═══════════════════════════════════════════════════

print(f"\nP3 : TROTTER DYNAMICS N=10")

N_trotter = 10
dt_trotter = 0.3  # pas de temps Trotter

for n_steps in [0, 1, 2, 3, 4, 6, 8, 12]:
    qc = QuantumCircuit(N_trotter, N_trotter)
    
    # État initial : tous spin up (ordonné)
    for i in range(N_trotter): qc.x(i)
    
    # Évolution Trotterisée : exp(-iHt) ≈ Π exp(-iH_k dt)
    # H = -J Σ σz_i σz_{i+1} - h Σ σx_i
    # Trotter : exp(-iJdt σz_i σz_{i+1}) × exp(ihdt σx_i)
    hJ = 0.5  # au milieu de la transition
    J = 1.0
    h = hJ * J
    
    for step in range(n_steps):
        # ZZ interaction : exp(-iJ dt σz_i σz_{i+1})
        for i in range(N_trotter - 1):
            qc.cx(i, i+1)
            qc.rz(2 * J * dt_trotter, i+1)
            qc.cx(i, i+1)
        # X field : exp(-ih dt σx_i) = Rx(2h dt)
        for i in range(N_trotter):
            qc.rx(2 * h * dt_trotter, i)
    
    qc.measure(range(N_trotter), range(N_trotter))
    circuits.append(qc); labels.append(f"trotter_s{n_steps}")

print(f"  +8 circuits (Trotter)")

print(f"\n  TOTAL : {len(circuits)} circuits")

# ═══════════════════════════════════════════════════
# ENVOI
# ═══════════════════════════════════════════════════

print(f"\nTranspilation...", flush=True)
isa_circuits = pm.run(circuits)

# Profondeurs Trotter
for i, label in enumerate(labels):
    if 'trotter' in label:
        print(f"  {label}: depth={isa_circuits[i].depth()}")

print(f"\nEnvoi sur {backend.name}...", flush=True)
t_submit = time.time()
sampler = SamplerV2(backend)
job = sampler.run([(c,) for c in isa_circuits], shots=4096)
print(f"  Job ID: {job.job_id()}")
print(f"  Attente...", flush=True)
result = job.result()
t_done = time.time()
print(f"  Résultats en {t_done-t_submit:.0f}s")

# ═══════════════════════════════════════════════════
# ANALYSE P1 : BARRES D'ERREUR
# ═══════════════════════════════════════════════════

print(f"\n{'='*60}")
print("P1 : BARRES D'ERREUR")
print(f"{'='*60}")

for prefix, N, desc in [("ord156", backend.num_qubits, "Ordered N=156"), 
                          ("par156", backend.num_qubits, "Param N=156"),
                          ("ord50", 50, "Ordered N=50"),
                          ("par50", 50, "Param N=50")]:
    Cs = []
    for i, label in enumerate(labels):
        if label.startswith(prefix):
            sz = extract_sz(result[i], N)
            C, T = compute_cpsi(sz)
            Cs.append(C)
    Cs = np.array(Cs)
    print(f"  {desc:20s}: C = {np.mean(Cs):.4f} ± {np.std(Cs):.4f}  (n={len(Cs)})")

# ═══════════════════════════════════════════════════
# ANALYSE P2 : QPT FINE
# ═══════════════════════════════════════════════════

print(f"\n{'='*60}")
print("P2 : QPT FINE N=50 (20 points)")
print(f"{'='*60}")

qpt_data = []
for i, label in enumerate(labels):
    if label.startswith("qptfine"):
        theta = float(label.split('_t')[1])
        sz = extract_sz(result[i], 50)
        C, T = compute_cpsi(sz)
        qpt_data.append((theta, C))
        print(f"  θ={theta:.3f}  C={C:.4f}")

if len(qpt_data) > 5:
    thetas_arr = np.array([x[0] for x in qpt_data])
    Cs_arr = np.array([x[1] for x in qpt_data])
    C_max = Cs_arr.max()
    C_min = Cs_arr.min()
    theta_mid = thetas_arr[np.argmin(np.abs(Cs_arr - (C_max+C_min)/2))]
    print(f"\n  C range: {C_max:.4f} → {C_min:.4f}")
    print(f"  θ midpoint (C = {(C_max+C_min)/2:.3f}): θ ≈ {theta_mid:.3f}")
    print(f"  En degrés: {np.degrees(theta_mid):.1f}°")

# ═══════════════════════════════════════════════════
# ANALYSE P3 : TROTTER DYNAMICS
# ═══════════════════════════════════════════════════

print(f"\n{'='*60}")
print("P3 : TROTTER DYNAMICS (N=10, h/J=0.5)")
print(f"{'='*60}")

print(f"\n  {'Steps':>6} {'t':>6} {'C':>8} {'<sz>':>8}")
for i, label in enumerate(labels):
    if label.startswith("trotter"):
        n_steps = int(label.split('_s')[1])
        t = n_steps * dt_trotter
        sz = extract_sz(result[i], N_trotter)
        C, T = compute_cpsi(sz)
        print(f"  {n_steps:6d} {t:6.2f} {C:8.4f} {np.mean(sz):+8.4f}")

print(f"\n  Si C diminue avec le temps → le champ transverse")
print(f"  désorganise l'état ordonné initial")
print(f"  = VRAIE dynamique quantique sur vrai hardware")

# ═══════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Temps hardware : {t_done-t_submit:.0f}s ({(t_done-t_submit)/60:.1f} min)")
print(f"Circuits : {len(circuits)}")
print(f"{'='*60}")
