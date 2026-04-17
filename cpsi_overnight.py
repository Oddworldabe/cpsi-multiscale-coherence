#!/usr/bin/env python3
"""
cpsi_overnight.py — Tests pour la nuit

PRIORITÉ 1 : Test trivial/physique à N=5 (steadystate faisable)
  → Même test que N=4 mais à N=5 où C commence à se différencier de Pureté
  → Si corrélations tiennent → pas trivial même à plus grande taille

PRIORITÉ 2 : max(C) = max(W) à N=5
  → Le résultat killer confirmé à N>4

PRIORITÉ 3 : Finite-size scaling N=15, N=16 (ground state, rapide)
  → Étendre le dataset de 121 à 143 points

PRIORITÉ 4 : Dérivation λ(γ) à N=5
  → λ/γ converge-t-il vers une constante ?

Tout séquentiel, un seul process, résultats incrémentaux.

python3 -u ~/Desktop/cpsi_overnight.py 2>&1 | tee ~/Desktop/overnight_log.txt
"""
import numpy as np
import qutip as qt
from scipy import stats, optimize
import warnings, time, csv
warnings.filterwarnings('ignore')
from pathlib import Path

print(f"qutip {qt.__version__}", flush=True)
OUT = Path.home() / "Desktop" / "cpsi_overnight"
OUT.mkdir(exist_ok=True)

def compute_cpsi(sz):
    N = len(sz)
    micro = np.mean([abs(sz[i]*sz[i+1]) for i in range(N-1)])
    macro = np.mean([abs(sz[i]*sz[j]) for i in range(N) for j in range(i+1,N)])
    meso = max(0, min(1, 1.0 - np.std(np.abs(sz))))
    C = (micro + macro + meso) / 3
    T = abs(micro-meso) + abs(meso-macro) + abs(micro-macro)
    return C, T

def ising_H(N, hJ, J=1.0, eps=0.01):
    h = hJ * J; H = 0
    for i in range(N-1):
        o = [qt.qeye(2)]*N; o[i] = qt.sigmaz(); o[i+1] = qt.sigmaz()
        H += -J * qt.tensor(o)
    for i in range(N):
        o = [qt.qeye(2)]*N; o[i] = qt.sigmax()
        H += -h * qt.tensor(o)
        o2 = [qt.qeye(2)]*N; o2[i] = qt.sigmaz()
        H += -eps * qt.tensor(o2)
    return H

def make_cops(N, gamma):
    c_ops = []
    for i in range(N):
        o = [qt.qeye(2)]*N; o[i] = qt.destroy(2)
        c_ops.append(np.sqrt(gamma) * qt.tensor(o))
    return c_ops

def make_sz_ops(N):
    ops = []
    for i in range(N):
        o = [qt.qeye(2)]*N; o[i] = qt.sigmaz()
        ops.append(qt.tensor(o))
    return ops


# ═══════════════════════════════════════════════════
# PRIORITÉ 1 : TRIVIAL/PHYSIQUE à N=5
# ═══════════════════════════════════════════════════

def test_trivial_n5():
    print(f"\n{'='*60}")
    print(f"P1 : TEST TRIVIAL/PHYSIQUE — N=5")
    print(f"h/J FIXE, gamma varie")
    print(f"{'='*60}", flush=True)
    
    N = 5
    gamma_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    
    all_results = []
    for hJ in [0.3, 0.5, 0.6, 0.8, 1.0]:
        print(f"\n  h/J = {hJ}:", flush=True)
        results = []
        
        for gamma in gamma_values:
            t0 = time.time()
            H = ising_H(N, hJ)
            c_ops = make_cops(N, gamma)
            sz_ops = make_sz_ops(N)
            
            try:
                rho = qt.steadystate(H, c_ops, method='direct')
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
                surprise = sum(-np.clip(p,1e-10,1-1e-10)*np.log(np.clip(p,1e-10,1-1e-10))
                              -(1-np.clip(p,1e-10,1-1e-10))*np.log(1-np.clip(p,1e-10,1-1e-10))
                              for p in probs) / N
                
                # Conductance
                conductance = np.mean([abs(sz[i]*sz[i+1]) for i in range(N-1)])
                
                E = qt.expect(H, rho)
                S = qt.entropy_vn(rho, 2)
                P = (rho*rho).tr().real
                F = E - gamma * S
                
                dt = time.time() - t0
                print(f"    g={gamma:.3f}  C={C:.4f}  surp={surprise:.4f}  P={P:.4f}  ({dt:.0f}s)", flush=True)
                results.append({'gamma':gamma,'C':C,'surprise':surprise,'sigma':sigma,
                                'conductance':conductance,'E':E,'P':P,'F':F})
            except Exception as e:
                print(f"    g={gamma:.3f}  X {str(e)[:50]}", flush=True)
        
        if len(results) < 5: continue
        C_a = np.array([r['C'] for r in results])
        P_a = np.array([r['P'] for r in results])
        
        print(f"\n    CORRÉLATIONS à h/J={hJ} (N=5) :")
        for name, arr in [("Surprise", [r['surprise'] for r in results]),
                          ("sigma_prod", [r['sigma'] for r in results]),
                          ("Conductance", [r['conductance'] for r in results]),
                          ("Purete", [r['P'] for r in results]),
                          ("-F", [-r['F'] for r in results])]:
            arr = np.array(arr)
            r, p = stats.pearsonr(C_a, arr)
            # Aussi tester si C corrèle MIEUX que Pureté
            rP, _ = stats.pearsonr(P_a, arr) if name != "Purete" else (1.0, 0)
            better = "C>P" if abs(r) > abs(rP) else "P>C"
            trivial = "PHYSIQUE" if abs(r) > 0.8 else "TRIVIAL" if abs(r) < 0.5 else "modere"
            print(f"      rho(C,{name:12s})={r:+.4f}  rho(P,·)={rP:+.4f}  {better}  {trivial}", flush=True)
        
        all_results.extend(results)
    
    with open(OUT / 'trivial_n5.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_results[0].keys())
        w.writeheader(); w.writerows(all_results)


# ═══════════════════════════════════════════════════
# PRIORITÉ 2 : max(C) = max(W) à N=5
# ═══════════════════════════════════════════════════

def test_control_n5():
    print(f"\n{'='*60}")
    print(f"P2 : max(C) = max(W) ? — N=5")
    print(f"{'='*60}", flush=True)
    
    N = 5; hJ = 0.3
    H = ising_H(N, hJ, eps=0)
    sz_ops = make_sz_ops(N)
    dim = 2**N
    E_gs = H.eigenenergies(eigvals=1, sparse=True)[0]
    
    n_steps = 200; dt = 0.5
    
    protocols = {
        'constant_0.05': lambda t: 0.05,
        'constant_0.1': lambda t: 0.1,
        'constant_0.3': lambda t: 0.3,
        'constant_0.5': lambda t: 0.5,
        'rampe_up': lambda t: 0.01 + 0.49 * t / (n_steps * dt),
        'rampe_down': lambda t: 0.50 - 0.49 * t / (n_steps * dt),
        'pulse': lambda t: 0.5 if int(t / 10) % 2 == 0 else 0.01,
        'fort_puis_faible': lambda t: 0.5 if t < n_steps*dt/2 else 0.05,
    }
    
    # Même état initial pour tous
    np.random.seed(42)
    psi0 = qt.Qobj(qt.rand_ket(dim).full(), dims=[[2]*N, [1]*N])
    
    results = {}
    for name, gamma_func in protocols.items():
        rho = qt.ket2dm(psi0)
        E_init = qt.expect(H, rho)
        
        for step in range(n_steps):
            t = step * dt
            gamma = gamma_func(t)
            c_ops = make_cops(N, gamma)
            res = qt.mesolve(H, rho, [0, dt], c_ops, [], options={"store_states": True})
            rho = res.states[-1]
        
        sz = np.array([qt.expect(op, rho) for op in sz_ops])
        C_final, _ = compute_cpsi(sz)
        E_final = qt.expect(H, rho)
        W = E_init - E_final
        
        results[name] = {'C_final': C_final, 'W': W, 'E_final': E_final}
        print(f"  {name:20s}  C={C_final:.4f}  W={W:.4f}", flush=True)
    
    best_C = max(results, key=lambda k: results[k]['C_final'])
    best_W = max(results, key=lambda k: results[k]['W'])
    
    print(f"\n  Best C: {best_C} (C={results[best_C]['C_final']:.4f})")
    print(f"  Best W: {best_W} (W={results[best_W]['W']:.4f})")
    
    if best_C == best_W:
        print(f"  --> MEME PROTOCOLE maximise C et W a N=5 !")
    else:
        print(f"  --> Protocoles differents : {best_C} vs {best_W}")
        # Corrélation C_final vs W across protocols
        Cs = [results[k]['C_final'] for k in results]
        Ws = [results[k]['W'] for k in results]
        r, p = stats.pearsonr(Cs, Ws)
        print(f"  rho(C_final, W) = {r:+.4f}  p={p:.4f}")


# ═══════════════════════════════════════════════════
# PRIORITÉ 3 : FINITE-SIZE N=15, N=16
# ═══════════════════════════════════════════════════

def test_finitesize_extend():
    print(f"\n{'='*60}")
    print(f"P3 : FINITE-SIZE N=15, N=16 (ground state)")
    print(f"{'='*60}", flush=True)
    
    hJ_values = [0.01, 0.2, 0.4, 0.5, 0.57, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]
    
    for N in [15, 16]:
        dim = 2**N
        print(f"\n  N={N} (dim={dim}):", flush=True)
        t_total = time.time()
        
        if dim > 65536:
            print(f"    dim={dim} trop grand pour eigenstates complet", flush=True)
            continue
        
        for hJ in hJ_values:
            t0 = time.time()
            try:
                H = ising_H(N, hJ)
                evals, estates = H.eigenstates(eigvals=2, sparse=True)
                psi = estates[0]; gap = evals[1] - evals[0]
                sz = []
                for i in range(N):
                    o = [qt.qeye(2)]*N; o[i] = qt.sigmaz()
                    sz.append(qt.expect(qt.tensor(o), psi))
                C, T = compute_cpsi(np.array(sz))
                dt = time.time() - t0
                print(f"    hJ={hJ:.2f}  C={C:.4f}  T={T:.4f}  gap={gap:.4f}  ({dt:.0f}s)", flush=True)
            except Exception as e:
                print(f"    hJ={hJ:.2f}  X {str(e)[:50]}", flush=True)
        
        print(f"  N={N} termine en {time.time()-t_total:.0f}s", flush=True)


# ═══════════════════════════════════════════════════
# PRIORITÉ 4 : DÉRIVATION λ(γ) à N=5
# ═══════════════════════════════════════════════════

def test_lambda_n5():
    print(f"\n{'='*60}")
    print(f"P4 : lambda(gamma) a N=5")
    print(f"{'='*60}", flush=True)
    
    N = 5; hJ = 0.3
    H = ising_H(N, hJ, eps=0)
    sz_ops = make_sz_ops(N)
    dim = 2**N
    tlist = np.linspace(0, 150, 500)
    
    def exp_relax(t, C_eq, C_0, lam):
        return C_eq + (C_0 - C_eq) * np.exp(-lam * t)
    
    print(f"  {'gamma':>8} {'lambda':>8} {'lam/gam':>8} {'R2':>8} {'C_eq':>8}", flush=True)
    
    for gamma in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        c_ops = make_cops(N, gamma)
        psi0 = qt.Qobj(qt.rand_ket(dim).full(), dims=[[2]*N, [1]*N])
        rho0 = qt.ket2dm(psi0)
        
        try:
            result = qt.mesolve(H, rho0, tlist, c_ops, sz_ops)
            C_t = []
            for t_idx in range(len(tlist)):
                sz = np.array([result.expect[i][t_idx] for i in range(N)])
                C, _ = compute_cpsi(sz)
                C_t.append(C)
            C_t = np.array(C_t)
            
            popt, _ = optimize.curve_fit(exp_relax, tlist, C_t, 
                                         p0=[C_t[-1], C_t[0], gamma], maxfev=5000)
            C_eq, C_0, lam = popt
            C_fit = exp_relax(tlist, *popt)
            r2 = 1 - np.sum((C_t - C_fit)**2) / np.sum((C_t - C_t.mean())**2)
            
            print(f"  {gamma:8.3f} {lam:8.4f} {lam/gamma:8.2f} {r2:8.4f} {C_eq:8.4f}", flush=True)
        except Exception as e:
            print(f"  {gamma:8.3f} FAIL: {str(e)[:40]}", flush=True)


# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"{'='*60}")
    print(f"C[Psi] OVERNIGHT — tests pour la nuit")
    print(f"{'='*60}", flush=True)
    
    t0 = time.time()
    
    test_finitesize_extend()  # ~30 min (N=15,16)
    test_lambda_n5()          # ~10 min
    test_trivial_n5()         # ~2-4h (55 steadystate N=5)
    test_control_n5()         # ~2h (8 protocoles × 200 steps)
    
    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"TOTAL: {total/3600:.1f}h")
    print(f"Resultats: {OUT}")
    print(f"{'='*60}")
