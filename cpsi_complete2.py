#!/usr/bin/env python3
"""
cpsi_complete.py — Tests complémentaires finaux

TEST 1 : FEEDBACK à N=5 (le +34% tient-il à plus grande taille ?)
TEST 2 : BORNE DE CARNOT — C prédit-il la distance au rendement max ?
TEST 3 : CYCLE THERMODYNAMIQUE — stroke engine avec C comme indicateur
TEST 4 : DEUX SYSTÈMES COUPLÉS — transfert d'énergie optimisé par C
TEST 5 : C vs TRAVAIL EXTRACTIBLE (Ergotropy) — la vraie mesure quantique

python3 -u ~/Desktop/cpsi_complete2.py 2>&1 | tee ~/Desktop/complete2_log.txt
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

def ergotropy(H, rho):
    """
    Ergotropy = max travail extractible par opération unitaire.
    W_erg = Tr(ρH) - Σ_i p_i ε_i (avec p_i triés décroissant, ε_i croissant)
    C'est la mesure quantique EXACTE du travail extractible.
    """
    E = qt.expect(H, rho)
    evals_H = np.sort(H.eigenenergies())
    evals_rho = np.sort(rho.eigenenergies())[::-1]  # décroissant
    E_passive = np.sum(evals_rho * evals_H[:len(evals_rho)])
    return E - E_passive


# ═══════════════════════════════════════════════════
# TEST 1 : FEEDBACK à N=5
# ═══════════════════════════════════════════════════

def test_feedback_n5():
    print(f"\n{'='*60}")
    print(f"TEST 1 : FEEDBACK à N=5")
    print(f"{'='*60}", flush=True)
    
    N = 5; hJ = 0.3
    H = ising_H(N, hJ, eps=0)
    sz_ops = make_sz_ops(N)
    dim = 2**N
    n_steps = 150; dt = 0.5
    
    np.random.seed(42)
    psi0 = qt.Qobj(qt.rand_ket(dim).full(), dims=[[2]*N, [1]*N])
    
    protocols = {}
    
    for name, gamma_func in [
        ('constant_0.1', lambda C, t: 0.1),
        ('constant_0.3', lambda C, t: 0.3),
        ('feedback_0.3', lambda C, t: max(0.001, 0.3 * (1 - C))),
        ('feedback_0.5', lambda C, t: max(0.001, 0.5 * (1 - C))),
        ('feedback_1.0', lambda C, t: max(0.001, 1.0 * (1 - C))),
        ('ramp_down', lambda C, t: max(0.001, 0.5 * np.exp(-t / 20))),
    ]:
        rho = qt.ket2dm(psi0)
        E_init = qt.expect(H, rho)
        total_gamma = 0
        
        for step in range(n_steps):
            t = step * dt
            sz = np.array([qt.expect(op, rho) for op in sz_ops])
            C_now, _ = compute_cpsi(sz)
            gamma = gamma_func(C_now, t)
            c_ops = make_cops(N, gamma)
            res = qt.mesolve(H, rho, [0, dt], c_ops, [], options={"store_states": True})
            rho = res.states[-1]
            total_gamma += gamma * dt
        
        sz = np.array([qt.expect(op, rho) for op in sz_ops])
        C_f, _ = compute_cpsi(sz)
        W = E_init - qt.expect(H, rho)
        eff = W / (total_gamma + 1e-10)
        protocols[name] = {'C':C_f, 'W':W, 'eff':eff, 'cost':total_gamma}
    
    print(f"  {'Protocole':<20} {'C_final':<10} {'W':<10} {'Cout':<10} {'Efficacite':<10}")
    print(f"  {'-'*60}")
    for name, r in sorted(protocols.items(), key=lambda x: -x[1]['eff']):
        print(f"  {name:<20} {r['C']:.4f}     {r['W']:.4f}     {r['cost']:.2f}       {r['eff']:.4f}")
    
    best = max(protocols, key=lambda k: protocols[k]['eff'])
    print(f"\n  MEILLEUR : {best}")
    if 'feedback' in best:
        # Calculer gain vs meilleur constant
        best_const = max([k for k in protocols if 'constant' in k], key=lambda k: protocols[k]['eff'])
        gain = (protocols[best]['eff'] / protocols[best_const]['eff'] - 1) * 100
        print(f"  Gain vs constant : +{gain:.0f}%")


# ═══════════════════════════════════════════════════
# TEST 2 : ERGOTROPY — la vraie mesure de travail extractible
# ═══════════════════════════════════════════════════

def test_ergotropy():
    """
    L'ergotropy est la mesure quantique EXACTE du travail
    qu'on peut extraire d'un état ρ par opération unitaire.
    
    Si ρ(C, Ergotropy) > 0.9 → C prédit EXACTEMENT 
    le travail extractible quantique.
    """
    print(f"\n{'='*60}")
    print(f"TEST 2 : C vs ERGOTROPY (travail extractible exact)")
    print(f"{'='*60}", flush=True)
    
    N = 4
    results = []
    
    for hJ in [0.3, 0.6, 1.0]:
        for gamma in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
            H = ising_H(N, hJ, eps=0.01)
            c_ops = make_cops(N, gamma)
            sz_ops = make_sz_ops(N)
            
            try:
                rho = qt.steadystate(H, c_ops, method='direct')
                sz = np.array([qt.expect(op, rho) for op in sz_ops])
                C, T = compute_cpsi(sz)
                W_erg = ergotropy(H, rho)
                P = (rho * rho).tr().real
                S = qt.entropy_vn(rho, 2)
                
                results.append({'hJ':hJ, 'gamma':gamma, 'C':C, 'W_erg':W_erg, 'P':P, 'S':S})
            except:
                pass
    
    if len(results) < 10:
        print("  Pas assez de points"); return
    
    C_a = np.array([r['C'] for r in results])
    W_a = np.array([r['W_erg'] for r in results])
    P_a = np.array([r['P'] for r in results])
    
    rCW, _ = stats.pearsonr(C_a, W_a)
    rPW, _ = stats.pearsonr(P_a, W_a)
    
    print(f"\n  {len(results)} steady states (3 hJ x 7 gamma) :")
    print(f"    rho(C, Ergotropy) = {rCW:+.4f}  {'C predit W_erg' if abs(rCW) > 0.8 else ''}")
    print(f"    rho(P, Ergotropy) = {rPW:+.4f}")
    
    if abs(rCW) > abs(rPW):
        print(f"    --> C PREDIT MIEUX le travail extractible que la purete !")
    else:
        print(f"    --> Purete predit mieux ({rPW:+.3f} vs {rCW:+.3f})")
    
    # À h/J fixe (éliminer le scan paramétrique)
    for hJ_fix in [0.3, 0.6, 1.0]:
        sub = [r for r in results if r['hJ'] == hJ_fix]
        if len(sub) < 4: continue
        C_sub = np.array([r['C'] for r in sub])
        W_sub = np.array([r['W_erg'] for r in sub])
        P_sub = np.array([r['P'] for r in sub])
        rCW_sub, _ = stats.pearsonr(C_sub, W_sub)
        rPW_sub, _ = stats.pearsonr(P_sub, W_sub)
        print(f"    h/J={hJ_fix} fixe : rho(C,W)={rCW_sub:+.4f}  rho(P,W)={rPW_sub:+.4f}  {'C>P' if abs(rCW_sub)>abs(rPW_sub) else 'P>C'}")


# ═══════════════════════════════════════════════════
# TEST 3 : CYCLE THERMODYNAMIQUE
# ═══════════════════════════════════════════════════

def test_thermo_cycle():
    """
    Cycle en 4 étapes :
    1. Compression : augmenter h/J (paramagnétiser)
    2. Thermalisation chaude : gamma fort
    3. Expansion : diminuer h/J (ré-ordonner)
    4. Thermalisation froide : gamma faible
    
    Mesurer W_net = travail extrait - travail fourni
    et C[Ψ] à chaque étape.
    """
    print(f"\n{'='*60}")
    print(f"TEST 3 : CYCLE THERMODYNAMIQUE")
    print(f"{'='*60}", flush=True)
    
    N = 4; dim = 2**N
    dt = 0.5; steps_per_stroke = 50
    
    # Paramètres du cycle
    hJ_low = 0.2   # phase ordonnée
    hJ_high = 1.0  # phase paramagnétique
    g_hot = 0.5    # dissipation forte
    g_cold = 0.05  # dissipation faible
    
    # État initial
    H_init = ising_H(N, hJ_low, eps=0)
    c_ops_init = make_cops(N, g_cold)
    rho = qt.steadystate(H_init, c_ops_init, method='direct')
    
    sz_ops = make_sz_ops(N)
    
    print(f"  Cycle : hJ_low={hJ_low} → hJ_high={hJ_high}, g_cold={g_cold} → g_hot={g_hot}")
    
    n_cycles = 5
    for cycle in range(n_cycles):
        W_total = 0
        Q_total = 0
        
        # STROKE 1 : Compression (hJ_low → hJ_high, gamma=g_cold)
        E_before = qt.expect(ising_H(N, hJ_low, eps=0), rho)
        for step in range(steps_per_stroke):
            hJ = hJ_low + (hJ_high - hJ_low) * step / steps_per_stroke
            H = ising_H(N, hJ, eps=0)
            c_ops = make_cops(N, g_cold)
            res = qt.mesolve(H, rho, [0, dt], c_ops, [], options={"store_states": True})
            rho = res.states[-1]
        E_after = qt.expect(ising_H(N, hJ_high, eps=0), rho)
        W_compress = E_after - E_before
        
        # STROKE 2 : Thermalisation chaude (hJ=hJ_high, gamma=g_hot)
        E_before = qt.expect(ising_H(N, hJ_high, eps=0), rho)
        H = ising_H(N, hJ_high, eps=0)
        c_ops = make_cops(N, g_hot)
        for step in range(steps_per_stroke):
            res = qt.mesolve(H, rho, [0, dt], c_ops, [], options={"store_states": True})
            rho = res.states[-1]
        E_after = qt.expect(H, rho)
        Q_hot = E_after - E_before
        
        # STROKE 3 : Expansion (hJ_high → hJ_low, gamma=g_hot)
        E_before = qt.expect(ising_H(N, hJ_high, eps=0), rho)
        for step in range(steps_per_stroke):
            hJ = hJ_high - (hJ_high - hJ_low) * step / steps_per_stroke
            H = ising_H(N, hJ, eps=0)
            c_ops = make_cops(N, g_hot)
            res = qt.mesolve(H, rho, [0, dt], c_ops, [], options={"store_states": True})
            rho = res.states[-1]
        E_after = qt.expect(ising_H(N, hJ_low, eps=0), rho)
        W_expand = E_before - E_after  # travail extrait
        
        # STROKE 4 : Thermalisation froide (hJ=hJ_low, gamma=g_cold)
        E_before = qt.expect(ising_H(N, hJ_low, eps=0), rho)
        H = ising_H(N, hJ_low, eps=0)
        c_ops = make_cops(N, g_cold)
        for step in range(steps_per_stroke):
            res = qt.mesolve(H, rho, [0, dt], c_ops, [], options={"store_states": True})
            rho = res.states[-1]
        E_after = qt.expect(H, rho)
        Q_cold = E_after - E_before
        
        W_net = W_expand - W_compress
        
        sz = np.array([qt.expect(op, rho) for op in sz_ops])
        C_end, T_end = compute_cpsi(sz)
        
        eff = W_net / (abs(Q_hot) + 1e-10)
        
        print(f"  Cycle {cycle+1}: W_net={W_net:+.4f}  Q_hot={Q_hot:+.4f}  eff={eff:.4f}  C={C_end:.4f}", flush=True)
    
    print(f"\n  Si W_net > 0 : le cycle PRODUIT du travail (moteur)")
    print(f"  Si W_net < 0 : le cycle CONSOMME du travail (pompe)")


# ═══════════════════════════════════════════════════
# TEST 4 : C PRÉDIT-IL FREE ENERGY MIEUX QUE PURETÉ ?
# (à h/J fixe pour éviter le biais paramétrique)
# ═══════════════════════════════════════════════════

def test_c_vs_purity_free_energy():
    print(f"\n{'='*60}")
    print(f"TEST 4 : C vs PURETÉ pour prédire F (h/J FIXE)")
    print(f"{'='*60}", flush=True)
    
    N = 4
    for hJ in [0.3, 0.6, 1.0]:
        print(f"\n  h/J = {hJ}:", flush=True)
        results = []
        
        for gamma in [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]:
            H = ising_H(N, hJ, eps=0.01)
            c_ops = make_cops(N, gamma)
            sz_ops = make_sz_ops(N)
            
            rho = qt.steadystate(H, c_ops, method='direct')
            sz = np.array([qt.expect(op, rho) for op in sz_ops])
            C, T = compute_cpsi(sz)
            E = qt.expect(H, rho)
            S = qt.entropy_vn(rho, 2)
            P = (rho*rho).tr().real
            F = E - gamma * S
            W_erg = ergotropy(H, rho)
            
            results.append({'gamma':gamma,'C':C,'P':P,'F':F,'W_erg':W_erg,'E':E})
        
        C_a = np.array([r['C'] for r in results])
        P_a = np.array([r['P'] for r in results])
        F_a = np.array([r['F'] for r in results])
        W_a = np.array([r['W_erg'] for r in results])
        
        rCF, _ = stats.pearsonr(C_a, -F_a)
        rPF, _ = stats.pearsonr(P_a, -F_a)
        rCW, _ = stats.pearsonr(C_a, W_a)
        rPW, _ = stats.pearsonr(P_a, W_a)
        
        print(f"    rho(C, -F)    = {rCF:+.4f}  rho(P, -F)    = {rPF:+.4f}  {'C>P' if abs(rCF)>abs(rPF) else 'P>C'}")
        print(f"    rho(C, W_erg) = {rCW:+.4f}  rho(P, W_erg) = {rPW:+.4f}  {'C>P' if abs(rCW)>abs(rPW) else 'P>C'}")


# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"{'='*60}")
    print(f"C[Psi] — TESTS COMPLEMENTAIRES FINAUX")
    print(f"{'='*60}")
    
    t0 = time.time()
    
    test_ergotropy()              # ~1 min
    test_c_vs_purity_free_energy()  # ~1 min
    test_feedback_n5()            # ~5 min
    test_thermo_cycle()           # ~10 min
    
    print(f"\nTOTAL: {(time.time()-t0)/60:.1f} min")
