#!/usr/bin/env python3
"""
cpsi_freeenergy2.py — Vers la maîtrise de l'énergie libre

TEST 1 : RENDEMENT DE CARNOT — C prédit-il l'efficacité thermodynamique ?
TEST 2 : PROTOCOLE OPTIMAL — quel gamma(t) maximise W pour un budget donné ?
TEST 3 : FEEDBACK CONTROL — utiliser C en temps réel pour optimiser gamma
TEST 4 : COUPLAGE MULTI-SYSTÈME — 2 systèmes échangent de l'énergie via C[Ψ]
TEST 5 : RÉSONANCE — le système extrait-il plus de travail à la résonance ?

python3 -u ~/Desktop/cpsi_freeenergy2.py 2>&1 | tee ~/Desktop/freeenergy2_log.txt
"""
import numpy as np
import qutip as qt
from scipy import stats, optimize
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
# TEST 1 : C vs EFFICACITÉ THERMODYNAMIQUE
# ═══════════════════════════════════════════════════

def test_efficiency():
    """
    Question : C[Ψ] prédit-il l'efficacité de conversion énergie → travail ?
    
    Efficacité = W_extrait / E_input
    W_extrait = E(t=0) - E(t=inf) 
    E_input = énergie du Hamiltonien + énergie de dissipation
    
    Si ρ(C_final, efficacité) > 0.8 → C est un indicateur de rendement
    """
    print(f"\n{'='*60}")
    print(f"TEST 1 : C vs EFFICACITÉ THERMODYNAMIQUE")
    print(f"{'='*60}", flush=True)
    
    N = 4; hJ = 0.3
    H = ising_H(N, hJ, eps=0)
    sz_ops = make_sz_ops(N)
    dim = 2**N
    E_gs = H.eigenenergies(eigvals=1, sparse=True)[0]
    
    n_steps = 200; dt = 0.5
    
    results = []
    # Tester beaucoup de gamma constants
    for gamma in [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        np.random.seed(42)
        psi0 = qt.Qobj(qt.rand_ket(dim).full(), dims=[[2]*N, [1]*N])
        rho = qt.ket2dm(psi0)
        E_init = qt.expect(H, rho)
        
        c_ops = make_cops(N, gamma)
        
        # Énergie totale fournie par la dissipation
        E_dissipated = 0
        for step in range(n_steps):
            E_before = qt.expect(H, rho)
            res = qt.mesolve(H, rho, [0, dt], c_ops, [], options={"store_states": True})
            rho = res.states[-1]
            E_after = qt.expect(H, rho)
            # La dissipation fournit de l'énergie au système (amplitude damping pousse vers |0⟩)
            E_dissipated += gamma * dt  # proxy du coût énergétique
        
        sz = np.array([qt.expect(op, rho) for op in sz_ops])
        C_final, T_final = compute_cpsi(sz)
        E_final = qt.expect(H, rho)
        W = E_init - E_final  # travail extrait
        efficiency = W / (E_dissipated + 1e-10)
        
        print(f"  gamma={gamma:.3f}  C={C_final:.4f}  W={W:.4f}  E_diss={E_dissipated:.2f}  eff={efficiency:.4f}", flush=True)
        results.append({'gamma':gamma, 'C':C_final, 'W':W, 'eff':efficiency, 'E_diss':E_dissipated})
    
    C_a = np.array([r['C'] for r in results])
    W_a = np.array([r['W'] for r in results])
    eff_a = np.array([r['eff'] for r in results])
    
    rCW, _ = stats.pearsonr(C_a, W_a)
    rCE, _ = stats.pearsonr(C_a, eff_a)
    
    print(f"\n  rho(C, W_extrait)  = {rCW:+.4f}")
    print(f"  rho(C, efficacite) = {rCE:+.4f}")
    
    # Quel gamma donne le meilleur rendement ?
    best_eff = max(results, key=lambda r: r['eff'])
    best_W = max(results, key=lambda r: r['W'])
    best_C = max(results, key=lambda r: r['C'])
    
    print(f"\n  Meilleur rendement : gamma={best_eff['gamma']} (eff={best_eff['eff']:.4f})")
    print(f"  Meilleur W         : gamma={best_W['gamma']} (W={best_W['W']:.4f})")
    print(f"  Meilleur C         : gamma={best_C['gamma']} (C={best_C['C']:.4f})")


# ═══════════════════════════════════════════════════
# TEST 2 : FEEDBACK — utiliser C pour contrôler gamma en temps réel
# ═══════════════════════════════════════════════════

def test_feedback():
    """
    Protocole feedback :
    - Mesurer C à chaque pas de temps
    - Ajuster gamma en fonction de C :
      - Si C < C_target : augmenter gamma (plus de dissipation pour ordonner)
      - Si C > C_target : diminuer gamma (économiser l'énergie)
    
    Comparer avec gamma constant optimal.
    """
    print(f"\n{'='*60}")
    print(f"TEST 2 : FEEDBACK CONTROL — gamma(C) en temps réel")
    print(f"{'='*60}", flush=True)
    
    N = 4; hJ = 0.3
    H = ising_H(N, hJ, eps=0)
    sz_ops = make_sz_ops(N)
    dim = 2**N
    
    n_steps = 200; dt = 0.5
    np.random.seed(42)
    psi0 = qt.Qobj(qt.rand_ket(dim).full(), dims=[[2]*N, [1]*N])
    
    protocols = {}
    
    # A) Gamma constant (baseline)
    for g_const in [0.1, 0.3]:
        rho = qt.ket2dm(psi0)
        E_init = qt.expect(H, rho)
        total_gamma = 0
        
        for step in range(n_steps):
            c_ops = make_cops(N, g_const)
            res = qt.mesolve(H, rho, [0, dt], c_ops, [], options={"store_states": True})
            rho = res.states[-1]
            total_gamma += g_const * dt
        
        sz = np.array([qt.expect(op, rho) for op in sz_ops])
        C_f, _ = compute_cpsi(sz)
        W = E_init - qt.expect(H, rho)
        eff = W / (total_gamma + 1e-10)
        protocols[f'constant_{g_const}'] = {'C':C_f, 'W':W, 'eff':eff, 'cost':total_gamma}
    
    # B) Feedback proportionnel : gamma = g_max * (1 - C)
    for g_max in [0.3, 0.5, 1.0]:
        rho = qt.ket2dm(psi0)
        E_init = qt.expect(H, rho)
        total_gamma = 0
        
        for step in range(n_steps):
            sz = np.array([qt.expect(op, rho) for op in sz_ops])
            C_now, _ = compute_cpsi(sz)
            gamma = g_max * max(0, 1 - C_now)  # Fort quand C bas, nul quand C ≈ 1
            gamma = max(gamma, 0.001)  # plancher
            
            c_ops = make_cops(N, gamma)
            res = qt.mesolve(H, rho, [0, dt], c_ops, [], options={"store_states": True})
            rho = res.states[-1]
            total_gamma += gamma * dt
        
        sz = np.array([qt.expect(op, rho) for op in sz_ops])
        C_f, _ = compute_cpsi(sz)
        W = E_init - qt.expect(H, rho)
        eff = W / (total_gamma + 1e-10)
        protocols[f'feedback_{g_max}'] = {'C':C_f, 'W':W, 'eff':eff, 'cost':total_gamma}
    
    # C) Rampe optimale : fort au début, faible à la fin
    rho = qt.ket2dm(psi0)
    E_init = qt.expect(H, rho)
    total_gamma = 0
    for step in range(n_steps):
        t = step * dt
        gamma = 0.5 * np.exp(-t / 20)  # décroissance exponentielle
        gamma = max(gamma, 0.001)
        c_ops = make_cops(N, gamma)
        res = qt.mesolve(H, rho, [0, dt], c_ops, [], options={"store_states": True})
        rho = res.states[-1]
        total_gamma += gamma * dt
    sz = np.array([qt.expect(op, rho) for op in sz_ops])
    C_f, _ = compute_cpsi(sz)
    W = E_init - qt.expect(H, rho)
    eff = W / (total_gamma + 1e-10)
    protocols['exp_decay'] = {'C':C_f, 'W':W, 'eff':eff, 'cost':total_gamma}
    
    # Résultats
    print(f"\n  {'Protocole':<20} {'C_final':<10} {'W':<10} {'Cout_g':<10} {'Efficacite':<10}")
    print(f"  {'-'*60}")
    for name, r in sorted(protocols.items(), key=lambda x: -x[1]['eff']):
        print(f"  {name:<20} {r['C']:.4f}     {r['W']:.4f}     {r['cost']:.2f}       {r['eff']:.4f}")
    
    best = max(protocols, key=lambda k: protocols[k]['eff'])
    print(f"\n  MEILLEUR RENDEMENT : {best}")
    
    if 'feedback' in best:
        print(f"  --> Le FEEDBACK basé sur C[Psi] bat le gamma constant !")
        print(f"  --> C[Psi] comme signal de controle = optimisation energetique")


# ═══════════════════════════════════════════════════
# TEST 3 : RÉSONANCE — max travail au point critique ?
# ═══════════════════════════════════════════════════

def test_resonance():
    """
    Hypothèse : le point critique (h/J ≈ 0.57) est le point
    où le système peut absorber/émettre le plus d'énergie.
    
    Test : balayer h/J et mesurer le travail extractible
    pour chaque point.
    """
    print(f"\n{'='*60}")
    print(f"TEST 3 : RÉSONANCE — max W au point critique ?")
    print(f"{'='*60}", flush=True)
    
    N = 4; gamma = 0.1
    dim = 2**N
    n_steps = 200; dt = 0.5
    
    results = []
    for hJ in np.linspace(0.01, 2.0, 20):
        H = ising_H(N, hJ, eps=0)
        c_ops = make_cops(N, gamma)
        sz_ops = make_sz_ops(N)
        
        np.random.seed(42)
        psi0 = qt.Qobj(qt.rand_ket(dim).full(), dims=[[2]*N, [1]*N])
        rho = qt.ket2dm(psi0)
        E_init = qt.expect(H, rho)
        
        for step in range(n_steps):
            res = qt.mesolve(H, rho, [0, dt], c_ops, [], options={"store_states": True})
            rho = res.states[-1]
        
        sz = np.array([qt.expect(op, rho) for op in sz_ops])
        C_f, T_f = compute_cpsi(sz)
        E_final = qt.expect(H, rho)
        W = E_init - E_final
        S = qt.entropy_vn(rho, 2)
        
        # Susceptibilité = dC/d(hJ) ≈ proxy de response
        results.append({'hJ':hJ, 'C':C_f, 'W':W, 'S':S, 'E_final':E_final})
    
    # Trouver le max de W
    W_arr = np.array([r['W'] for r in results])
    hJ_arr = np.array([r['hJ'] for r in results])
    C_arr = np.array([r['C'] for r in results])
    
    max_W_idx = np.argmax(W_arr)
    
    print(f"  Max W = {W_arr[max_W_idx]:.4f} a h/J = {hJ_arr[max_W_idx]:.2f}")
    print(f"  C a ce point = {C_arr[max_W_idx]:.4f}")
    
    # Le max de W est-il proche du point critique ?
    if abs(hJ_arr[max_W_idx] - 0.57) < 0.2:
        print(f"  --> Max W PRES du point critique (0.57) !")
        print(f"  --> La transition de phase = point de max extraction d'energie")
    else:
        print(f"  --> Max W a h/J={hJ_arr[max_W_idx]:.2f}, loin du critique")
    
    rCW, _ = stats.pearsonr(C_arr, W_arr)
    print(f"  rho(C, W) = {rCW:+.4f}")


# ═══════════════════════════════════════════════════
# TEST 4 : TAILLE DU SYSTÈME vs EFFICACITÉ
# ═══════════════════════════════════════════════════

def test_size_efficiency():
    """
    L'efficacité augmente-t-elle avec la taille du système ?
    Si oui → les gros systèmes organisés extraient plus de travail
    → la cohérence est SCALABLE
    """
    print(f"\n{'='*60}")
    print(f"TEST 4 : TAILLE vs EFFICACITÉ")
    print(f"{'='*60}", flush=True)
    
    gamma = 0.1; hJ = 0.3
    n_steps = 100; dt = 0.5
    
    for N in [3, 4, 5, 6]:
        dim = 2**N
        H = ising_H(N, hJ, eps=0)
        c_ops = make_cops(N, gamma)
        sz_ops = make_sz_ops(N)
        E_gs = H.eigenenergies(eigvals=1, sparse=True)[0]
        
        np.random.seed(42)
        psi0 = qt.Qobj(qt.rand_ket(dim).full(), dims=[[2]*N, [1]*N])
        rho = qt.ket2dm(psi0)
        E_init = qt.expect(H, rho)
        
        t0 = time.time()
        total_gamma = 0
        for step in range(n_steps):
            res = qt.mesolve(H, rho, [0, dt], c_ops, [], options={"store_states": True})
            rho = res.states[-1]
            total_gamma += gamma * dt
        
        sz = np.array([qt.expect(op, rho) for op in sz_ops])
        C_f, _ = compute_cpsi(sz)
        W = E_init - qt.expect(H, rho)
        eff = W / (total_gamma + 1e-10)
        eff_per_qubit = eff / N
        dt_calc = time.time() - t0
        
        print(f"  N={N}  C={C_f:.4f}  W={W:.4f}  eff={eff:.4f}  eff/qubit={eff_per_qubit:.4f}  ({dt_calc:.1f}s)", flush=True)


# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"{'='*60}")
    print(f"C[Psi] — VERS LA MAITRISE DE L'ENERGIE LIBRE")
    print(f"{'='*60}")
    
    t0 = time.time()
    
    test_efficiency()       # ~3 min
    test_feedback()         # ~5 min
    test_resonance()        # ~3 min
    test_size_efficiency()  # ~5 min
    
    print(f"\nTOTAL: {(time.time()-t0)/60:.1f} min")
