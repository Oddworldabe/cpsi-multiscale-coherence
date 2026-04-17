#!/usr/bin/env python3
"""
cpsi_variational.py — C n'est pas monotone en t, mais est-il monotone en E ?

Hypothèse : le système relaxe en minimisant E (garanti par dissipation).
Si C = f(E) monotone, alors C est "variationnel via l'énergie".

Aussi : existe-t-il une combinaison f(C, T) qui EST monotone en t ?

python3 ~/Desktop/cpsi_variational.py
"""
import numpy as np
import qutip as qt
from scipy import stats
from pathlib import Path
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
    return C, T, micro, macro, meso

N = 4
gamma = 0.1

for hJ in [0.3, 0.6, 1.0]:
    print(f"\n{'='*60}")
    print(f"h/J = {hJ}")
    print(f"{'='*60}")
    
    J = 1.0; h = hJ * J; epsilon = 0.01
    H = 0
    for i in range(N-1):
        o = [qt.qeye(2)]*N; o[i] = qt.sigmaz(); o[i+1] = qt.sigmaz()
        H += -J * qt.tensor(o)
    for i in range(N):
        o = [qt.qeye(2)]*N; o[i] = qt.sigmax()
        H += -h * qt.tensor(o)
        o2 = [qt.qeye(2)]*N; o2[i] = qt.sigmaz()
        H += -epsilon * qt.tensor(o2)
    
    c_ops = []
    for i in range(N):
        o = [qt.qeye(2)]*N; o[i] = qt.destroy(2)
        c_ops.append(np.sqrt(gamma) * qt.tensor(o))
    
    sz_ops = []
    for i in range(N):
        o = [qt.qeye(2)]*N; o[i] = qt.sigmaz()
        sz_ops.append(qt.tensor(o))
    
    # 5 trajectoires random
    tlist = np.linspace(0, 80, 400)
    dim = 2**N
    
    for trial in range(5):
        psi0 = qt.Qobj(qt.rand_ket(dim).full(), dims=[[2]*N, [1]*N])
        rho0 = qt.ket2dm(psi0)
        
        # mesolve avec H comme e_op aussi pour avoir E(t)
        result = qt.mesolve(H, rho0, tlist, c_ops, sz_ops + [H])
        
        C_t = []; T_t = []; E_t = []; S_t = []
        mi_t = []; ma_t = []; me_t = []
        
        for t_idx in range(len(tlist)):
            sz = np.array([result.expect[i][t_idx] for i in range(N)])
            C, T, mi, ma, me = compute_cpsi(sz)
            E = result.expect[N][t_idx]  # <H>
            C_t.append(C); T_t.append(T); E_t.append(E)
            mi_t.append(mi); ma_t.append(ma); me_t.append(me)
        
        C_t = np.array(C_t); T_t = np.array(T_t); E_t = np.array(E_t)
        mi_t = np.array(mi_t); ma_t = np.array(ma_t); me_t = np.array(me_t)
        
        # 1. E(t) est-il monotone ?
        dE = np.diff(E_t)
        E_mono = np.sum(dE < 1e-8)  # combien de pas E diminue
        
        # 2. C est-il monotone en E ? (pas en t)
        # Trier par E et vérifier si C est monotone
        sort_idx = np.argsort(E_t)
        C_sorted_by_E = C_t[sort_idx]
        dC_by_E = np.diff(C_sorted_by_E)
        C_mono_E = np.sum(dC_by_E < -1e-6)  # violations monotonie C(E)
        
        # 3. Corrélation C(t) vs E(t)
        rCE, _ = stats.pearsonr(C_t, E_t)
        
        # 4. Tester des combinaisons f(C,T)
        dC = np.diff(C_t)
        dT = np.diff(T_t)
        
        # Candidats variationnels
        candidates = {
            'C': C_t,
            '-T': -T_t,
            'C-T': C_t - T_t,
            'C+T': C_t + T_t,
            'C*T': C_t * T_t,
            'C/(1+T)': C_t / (1 + T_t),
            'C^2': C_t**2,
            'log(C)': np.log(C_t + 1e-10),
            'C-T/3': C_t - T_t/3,
            'micro': mi_t,
            'macro': ma_t,
            'meso': me_t,
            'micro+macro': mi_t + ma_t,
            'E': E_t,
        }
        
        if trial == 0:
            print(f"\n  {'Candidat':<16} {'violations':<12} {'direction':<10} {'rho(·,t)':<10}")
            print(f"  {'-'*48}")
        
        for name, arr in candidates.items():
            d = np.diff(arr)
            # Compter violations dans la direction dominante
            n_pos = np.sum(d > 1e-8)
            n_neg = np.sum(d < -1e-8)
            if n_pos > n_neg:
                violations = n_neg
                direction = "croissant"
            else:
                violations = n_pos
                direction = "decroissant"
            
            rho_t, _ = stats.pearsonr(arr, tlist)
            
            if trial == 0:
                print(f"  {name:<16} {violations:<12} {direction:<10} {rho_t:+.4f}")
        
        # Résumé trajectoire
        if trial == 0:
            print(f"\n  E monotone decroissant: {E_mono}/{len(dE)} pas")
            print(f"  rho(C, E) = {rCE:+.4f}")
            print(f"  C monotone en E: violations = {C_mono_E}/{len(dC_by_E)}")

print(f"\n{'='*60}")
print("INTERPRETATION:")
print("Si un candidat a 0 violations -> principe variationnel")
print("Si E a 0 violations -> relaxation dissipative OK")
print("Si C monotone en E -> C variationnel VIA energie")
print(f"{'='*60}")
