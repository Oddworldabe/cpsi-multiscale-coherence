#!/usr/bin/env python3
"""
cpsi_trivial_test.py — Trivial ou profond ?

À h/J FIXE, varier SEULEMENT gamma (0.01 à 1.0).
Si les corrélations C↔surprise/σ_prod/conductance tiennent → PHYSIQUE
Si elles s'effondrent → juste "tout monotone en h/J"

python3 ~/Desktop/cpsi_trivial_test.py
"""
import numpy as np
import qutip as qt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print(f"qutip {qt.__version__}")

def compute_cpsi(sz):
    N = len(sz)
    micro = np.mean([abs(sz[i]*sz[i+1]) for i in range(N-1)])
    macro = np.mean([abs(sz[i]*sz[j]) for i in range(N) for j in range(i+1,N)])
    meso = max(0, min(1, 1.0 - np.std(np.abs(sz))))
    C = (micro + macro + meso) / 3
    T = abs(micro-meso) + abs(meso-macro) + abs(micro-macro)
    return C, T

N = 4
gamma_values = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]

for hJ in [0.3, 0.5, 0.6, 0.8, 1.0]:
    print(f"\n{'='*60}")
    print(f"h/J = {hJ} FIXE — scan gamma")
    print(f"{'='*60}")
    
    results = []
    J = 1.0; h = hJ * J; eps = 0.01
    
    for gamma in gamma_values:
        H = 0
        for i in range(N-1):
            o = [qt.qeye(2)]*N; o[i] = qt.sigmaz(); o[i+1] = qt.sigmaz()
            H += -J * qt.tensor(o)
        for i in range(N):
            o = [qt.qeye(2)]*N; o[i] = qt.sigmax()
            H += -h * qt.tensor(o)
            o2 = [qt.qeye(2)]*N; o2[i] = qt.sigmaz()
            H += -eps * qt.tensor(o2)
        
        c_ops = []
        for i in range(N):
            o = [qt.qeye(2)]*N; o[i] = qt.destroy(2)
            c_ops.append(np.sqrt(gamma) * qt.tensor(o))
        
        rho = qt.steadystate(H, c_ops, method='direct')
        
        sz_ops = []
        for i in range(N):
            o = [qt.qeye(2)]*N; o[i] = qt.sigmaz()
            sz_ops.append(qt.tensor(o))
        sz = np.array([qt.expect(op, rho) for op in sz_ops])
        C, T = compute_cpsi(sz)
        
        # Production d'entropie
        sigma = 0
        for i in range(N):
            o = [qt.qeye(2)]*N; o[i] = qt.destroy(2)
            Lk = qt.tensor(o)
            sigma += gamma * qt.expect(Lk.dag() * Lk, rho)
        
        # Surprise
        probs = [(1 + s)/2 for s in sz]
        surprise = 0
        for p in probs:
            p = np.clip(p, 1e-10, 1-1e-10)
            surprise -= p * np.log(p) + (1-p) * np.log(1-p)
        surprise /= N
        
        # Conductance
        conductance = np.mean([abs(sz[i]*sz[i+1]) for i in range(N-1)])
        
        # Énergie
        E = qt.expect(H, rho)
        S = qt.entropy_vn(rho, 2)
        P = (rho*rho).tr().real
        F = E - gamma * S
        
        print(f"  g={gamma:.3f}  C={C:.4f}  surp={surprise:.4f}  sig={sigma:.4f}  cond={conductance:.4f}  P={P:.4f}")
        results.append({'gamma':gamma,'C':C,'surprise':surprise,'sigma':sigma,
                        'conductance':conductance,'E':E,'S':S,'P':P,'F':F})
    
    if len(results) < 5: continue
    
    C_a = np.array([r['C'] for r in results])
    surp_a = np.array([r['surprise'] for r in results])
    sig_a = np.array([r['sigma'] for r in results])
    cond_a = np.array([r['conductance'] for r in results])
    P_a = np.array([r['P'] for r in results])
    F_a = np.array([r['F'] for r in results])
    
    print(f"\n  CORRÉLATIONS à h/J={hJ} FIXE (gamma varie) :")
    
    tests = [
        ("Surprise", surp_a),
        ("sigma_prod", sig_a),
        ("Conductance", cond_a),
        ("Purete", P_a),
        ("-F", -F_a),
    ]
    
    for name, arr in tests:
        r, p = stats.pearsonr(C_a, arr)
        trivial = "TRIVIAL" if abs(r) < 0.5 else "PHYSIQUE" if abs(r) > 0.8 else "modere"
        print(f"    rho(C, {name:12s}) = {r:+.4f}  p={p:.4f}  -> {trivial}")

print(f"\n{'='*60}")
print(f"VERDICT FINAL")
print(f"Si les correlations tiennent a h/J fixe -> le lien est PHYSIQUE")
print(f"Si elles s'effondrent -> c'etait juste 'tout monotone en h/J'")
print(f"{'='*60}")
