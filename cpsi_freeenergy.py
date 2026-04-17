#!/usr/bin/env python3
"""
cpsi_freeenergy.py — C[Ψ] = énergie libre organisationnelle ?

TEST 1 : C/(1+T) quasi-variationnel (N=5, 7, multi-gamma, 20 trajectoires)
TEST 2 : C vs F = E - T_bath * S pendant dynamique
TEST 3 : Efficacité de travail extractible vs C[Ψ]
TEST 4 : Contrôle : moduler gamma(t) pour MAXIMISER C → maximise-t-on le travail ?

python3 ~/Desktop/cpsi_freeenergy.py
"""
import numpy as np
import qutip as qt
from scipy import stats
import warnings, time
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

def ising_H(N, hJ, J=1.0, eps=0.01):
    h = hJ * J
    H = 0
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

def run_trajectory(H, c_ops, sz_ops, N, tlist):
    dim = 2**N
    psi0 = qt.Qobj(qt.rand_ket(dim).full(), dims=[[2]*N, [1]*N])
    rho0 = qt.ket2dm(psi0)
    result = qt.mesolve(H, rho0, tlist, c_ops, sz_ops + [H])
    
    C_t = []; T_t = []; E_t = []
    for t_idx in range(len(tlist)):
        sz = np.array([result.expect[i][t_idx] for i in range(N)])
        C, T = compute_cpsi(sz)
        E = result.expect[N][t_idx]
        C_t.append(C); T_t.append(T); E_t.append(E)
    return np.array(C_t), np.array(T_t), np.array(E_t)


# ═══════════════════════════════════════════════════
# TEST 1 : C/(1+T) — test robuste
# ═══════════════════════════════════════════════════

def test_quasi_variational():
    print(f"\n{'='*60}")
    print(f"TEST 1 : C/(1+T) QUASI-VARIATIONNEL ?")
    print(f"20 trajectoires, N=4 et N=5, 3 gammas")
    print(f"{'='*60}", flush=True)
    
    tlist = np.linspace(0, 80, 400)
    
    for N in [4, 5]:
        for gamma in [0.05, 0.1, 0.3]:
            for hJ in [0.3, 0.6]:
                H = ising_H(N, hJ)
                c_ops = make_cops(N, gamma)
                sz_ops = make_sz_ops(N)
                
                violations_C = []
                violations_CT = []
                violations_E = []
                
                for trial in range(20):
                    C_t, T_t, E_t = run_trajectory(H, c_ops, sz_ops, N, tlist)
                    CT = C_t / (1 + T_t)
                    
                    dC = np.diff(C_t)
                    dCT = np.diff(CT)
                    dE = np.diff(E_t)
                    
                    violations_C.append(np.sum(dC < -1e-6))
                    violations_CT.append(np.sum(dCT < -1e-6))
                    violations_E.append(np.sum(dE > 1e-6))
                
                vC = np.mean(violations_C)
                vCT = np.mean(violations_CT)
                vE = np.mean(violations_E)
                
                # Combien de trajectoires sont "quasi-monotones" (< 5% violations)
                threshold = len(tlist) * 0.05
                mono_C = sum(1 for v in violations_C if v < threshold)
                mono_CT = sum(1 for v in violations_CT if v < threshold)
                mono_E = sum(1 for v in violations_E if v < threshold)
                
                print(f"  N={N} g={gamma:.2f} hJ={hJ}:  E:{mono_E}/20  C:{mono_C}/20  C/(1+T):{mono_CT}/20  (violations moy: E={vE:.0f} C={vC:.0f} CT={vCT:.0f})", flush=True)


# ═══════════════════════════════════════════════════
# TEST 2 : C vs ÉNERGIE LIBRE pendant dynamique
# F = E - gamma * S_vN (gamma comme proxy de température)
# ═══════════════════════════════════════════════════

def test_free_energy():
    print(f"\n{'='*60}")
    print(f"TEST 2 : C vs ÉNERGIE LIBRE F = E - gamma*S")
    print(f"Si C ~ -F alors C = énergie libre organisationnelle")
    print(f"{'='*60}", flush=True)
    
    N = 4
    gamma = 0.1
    tlist = np.linspace(0, 80, 200)
    
    for hJ in [0.3, 0.6, 1.0]:
        H = ising_H(N, hJ)
        c_ops = make_cops(N, gamma)
        sz_ops = make_sz_ops(N)
        dim = 2**N
        
        psi0 = qt.Qobj(qt.rand_ket(dim).full(), dims=[[2]*N, [1]*N])
        rho0 = qt.ket2dm(psi0)
        
        # mesolve avec states pour calculer S_vN
        result = qt.mesolve(H, rho0, tlist, c_ops, [], options={"store_states": True})
        
        C_t = []; F_t = []; E_t = []; S_t = []; W_t = []
        E_gs = H.eigenenergies(eigvals=1, sparse=True)[0]
        
        for t_idx in range(len(tlist)):
            rho = result.states[t_idx]
            sz = np.array([qt.expect(op, rho) for op in sz_ops])
            C, T = compute_cpsi(sz)
            E = qt.expect(H, rho)
            S = qt.entropy_vn(rho, 2)
            F = E - gamma * S
            W = E - E_gs  # travail extractible
            
            C_t.append(C); F_t.append(F); E_t.append(E)
            S_t.append(S); W_t.append(W)
        
        C_t = np.array(C_t); F_t = np.array(F_t)
        E_t = np.array(E_t); S_t = np.array(S_t); W_t = np.array(W_t)
        
        rCE, _ = stats.pearsonr(C_t, E_t)
        rCF, _ = stats.pearsonr(C_t, F_t)
        rCS, _ = stats.pearsonr(C_t, S_t)
        rCW, _ = stats.pearsonr(C_t, W_t)
        
        print(f"\n  h/J={hJ}:")
        print(f"    rho(C, E)     = {rCE:+.4f}  {'C ~ -E' if rCE < -0.9 else ''}")
        print(f"    rho(C, F)     = {rCF:+.4f}  {'C ~ -F !' if rCF < -0.9 else ''}")
        print(f"    rho(C, S_vN)  = {rCS:+.4f}")
        print(f"    rho(C, W_ext) = {rCW:+.4f}  {'C ~ -W_extractible' if rCW < -0.9 else ''}")
        
        # C(0) vs C(inf) vs F(0) vs F(inf)
        print(f"    C: {C_t[0]:.4f} -> {C_t[-1]:.4f}")
        print(f"    F: {F_t[0]:.4f} -> {F_t[-1]:.4f}")
        print(f"    E: {E_t[0]:.4f} -> {E_t[-1]:.4f}")
        print(f"    S: {S_t[0]:.4f} -> {S_t[-1]:.4f}")


# ═══════════════════════════════════════════════════
# TEST 3 : CONTRÔLE — moduler gamma pour maximiser C
# Si C = énergie libre, alors max(C) = max rendement
# ═══════════════════════════════════════════════════

def test_control():
    """
    Protocole :
    A) gamma constant = 0.1
    B) gamma rampe croissante 0.01 → 0.3
    C) gamma rampe décroissante 0.3 → 0.01
    D) gamma pulsé (on/off)
    
    Question : quel protocole donne le C final le plus élevé ?
    Et est-ce aussi celui qui extrait le plus de travail ?
    """
    print(f"\n{'='*60}")
    print(f"TEST 3 : CONTRÔLE gamma(t) → maximiser C")
    print(f"Quel protocole maximise la cohérence finale ?")
    print(f"{'='*60}", flush=True)
    
    N = 4
    hJ = 0.3
    H = ising_H(N, hJ)
    sz_ops = make_sz_ops(N)
    dim = 2**N
    E_gs = H.eigenenergies(eigvals=1, sparse=True)[0]
    
    n_steps = 200
    dt = 0.5
    
    protocols = {
        'constant_0.1': lambda t: 0.1,
        'constant_0.3': lambda t: 0.3,
        'rampe_up': lambda t: 0.01 + 0.29 * t / (n_steps * dt),
        'rampe_down': lambda t: 0.30 - 0.29 * t / (n_steps * dt),
        'pulse': lambda t: 0.3 if int(t / 10) % 2 == 0 else 0.01,
        'optimal_guess': lambda t: 0.3 if t < n_steps*dt/2 else 0.01,
    }
    
    results = {}
    
    for name, gamma_func in protocols.items():
        # Simuler pas à pas
        psi0 = qt.Qobj(qt.rand_ket(dim).full(), dims=[[2]*N, [1]*N])
        rho = qt.ket2dm(psi0)
        
        C_history = []
        E_history = []
        
        for step in range(n_steps):
            t = step * dt
            gamma = gamma_func(t)
            c_ops = make_cops(N, gamma)
            
            # Un petit pas mesolve
            result = qt.mesolve(H, rho, [0, dt], c_ops, [], options={"store_states": True})
            rho = result.states[-1]
            
            sz = np.array([qt.expect(op, rho) for op in sz_ops])
            C, T = compute_cpsi(sz)
            E = qt.expect(H, rho)
            
            C_history.append(C)
            E_history.append(E)
        
        C_final = C_history[-1]
        E_final = E_history[-1]
        W_extracted = E_history[0] - E_final  # travail extrait = perte d'énergie
        efficiency = C_final / (abs(E_history[0] - E_gs) + 1e-10)
        
        results[name] = {
            'C_final': C_final,
            'E_final': E_final,
            'W_extracted': W_extracted,
            'efficiency': efficiency,
        }
        print(f"  {name:20s}  C_final={C_final:.4f}  W={W_extracted:.4f}  eff={efficiency:.4f}", flush=True)
    
    # Quel protocole gagne ?
    best_C = max(results, key=lambda k: results[k]['C_final'])
    best_W = max(results, key=lambda k: results[k]['W_extracted'])
    
    print(f"\n  Meilleur C final : {best_C}")
    print(f"  Meilleur W extrait : {best_W}")
    
    if best_C == best_W:
        print(f"  --> MEME PROTOCOLE maximise C et W !")
        print(f"  --> Maximiser C[Psi] = maximiser le travail extractible")
        print(f"  --> C[Psi] EST la clé vers la maitrise de l'énergie libre")
    else:
        print(f"  --> Protocoles différents pour C et W")
        print(f"  --> C[Psi] ne controle pas directement le travail")


# ═══════════════════════════════════════════════════
# TEST 4 : RÉSUMÉ — l'équation organisationnelle
# ═══════════════════════════════════════════════════

def test_equation():
    """Synthèse : quelle est l'équation ?"""
    print(f"\n{'='*60}")
    print(f"TEST 4 : L'ÉQUATION ORGANISATIONNELLE")
    print(f"{'='*60}", flush=True)
    
    N = 4
    gamma = 0.1
    hJ_values = np.linspace(0.01, 2.0, 20)
    
    H_list = [ising_H(N, hJ) for hJ in hJ_values]
    
    results = []
    for idx, hJ in enumerate(hJ_values):
        H = H_list[idx]
        c_ops = make_cops(N, gamma)
        
        rho = qt.steadystate(H, c_ops, method='direct')
        sz_ops = make_sz_ops(N)
        sz = np.array([qt.expect(op, rho) for op in sz_ops])
        C, T = compute_cpsi(sz)
        E = qt.expect(H, rho)
        S = qt.entropy_vn(rho, 2)
        P = (rho*rho).tr().real
        F = E - gamma * S
        E_gs = H.eigenenergies(eigvals=1, sparse=True)[0]
        
        results.append({
            'hJ':hJ, 'C':C, 'T':T, 'E':E, 'S':S, 'P':P, 'F':F,
            'E_gs':E_gs, 'CT':C/(1+T), 'neg_F':-F
        })
    
    C_a = np.array([r['C'] for r in results])
    neg_F = np.array([r['neg_F'] for r in results])
    E_a = np.array([r['E'] for r in results])
    P_a = np.array([r['P'] for r in results])
    CT_a = np.array([r['CT'] for r in results])
    
    # L'équation : C ~ -F ?
    rCF, _ = stats.pearsonr(C_a, neg_F)
    rCE, _ = stats.pearsonr(C_a, -E_a)
    rCP, _ = stats.pearsonr(C_a, P_a)
    rCTF, _ = stats.pearsonr(CT_a, neg_F)
    
    print(f"\n  Sur 20 points steady-state (N={N}, gamma={gamma}) :")
    print(f"    rho(C, -F)       = {rCF:+.4f}")
    print(f"    rho(C, -E)       = {rCE:+.4f}")
    print(f"    rho(C, Purete)   = {rCP:+.4f}")
    print(f"    rho(C/(1+T), -F) = {rCTF:+.4f}")
    
    # Fit linéaire C = a*(-F) + b
    from numpy.linalg import lstsq
    X = np.column_stack([neg_F, np.ones(len(neg_F))])
    b, _, _, _ = lstsq(X, C_a, rcond=None)
    r2 = 1 - np.sum((C_a - X@b)**2) / np.sum((C_a - C_a.mean())**2)
    
    print(f"\n  FIT : C = {b[0]:.4f} * (-F) + {b[1]:.4f}")
    print(f"  R2 = {r2:.4f}")
    
    if r2 > 0.95:
        print(f"\n  --> C[Psi] ~ -F (energie libre)")
        print(f"  --> L'EQUATION : C[Psi] = a * F_org + b")
        print(f"  --> ou F_org = energie libre organisationnelle")
        print(f"  --> Maximiser C = minimiser F = maximiser travail extractible")


# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"{'='*60}")
    print(f"C[Psi] = ENERGIE LIBRE ORGANISATIONNELLE ?")
    print(f"{'='*60}")
    
    t0 = time.time()
    
    test_free_energy()        # ~3 min
    test_equation()           # ~1 min
    test_control()            # ~5 min
    test_quasi_variational()  # ~10 min
    
    print(f"\nTOTAL: {(time.time()-t0)/60:.1f} min")
