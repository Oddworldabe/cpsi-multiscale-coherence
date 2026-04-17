#!/usr/bin/env python3
"""
cpsi_deep.py — Tests fondamentaux

PARTIE 1 : Renforcer P3q
  A) Ising ground state avec brisure de symétrie (fix du test raté)
  B) Thermo N=7 (10 points, C vs E/S_vN/Pureté)

PARTIE 2 : L'équation observateur ↔ univers
  C) C[Ψ] vs information mutuelle I(sys:env)
  D) C[Ψ] vs force de mesure (décohérence progressive)
  E) Quantum Darwinism : C[Ψ] vs redondance environnementale

python3 -u ~/Desktop/cpsi_deep.py 2>&1 | tee ~/Desktop/deep_log.txt
"""
import numpy as np
import time, warnings, sys, csv
warnings.filterwarnings('ignore')

try:
    import qutip as qt
    print(f"qutip {qt.__version__}", flush=True)
except ImportError:
    print("pip3 install qutip"); sys.exit(1)

from scipy import stats
from pathlib import Path

OUT = Path.home() / "Desktop" / "cpsi_deep"
OUT.mkdir(exist_ok=True)


def compute_cpsi(sz):
    N = len(sz)
    micro = np.mean([abs(sz[i]*sz[i+1]) for i in range(N-1)])
    macro = np.mean([abs(sz[i]*sz[j]) for i in range(N) for j in range(i+1,N)])
    meso = max(0, min(1, 1.0 - np.std(np.abs(sz))))
    C = (micro + macro + meso) / 3
    T = abs(micro-meso) + abs(meso-macro) + abs(micro-macro)
    return C, T


def ising_H(N, hJ, J=1.0, epsilon=0.0):
    """Ising transverse + champ longitudinal epsilon pour briser la symétrie Z2."""
    h = hJ * J
    H = 0
    for i in range(N-1):
        o = [qt.qeye(2)]*N; o[i] = qt.sigmaz(); o[i+1] = qt.sigmaz()
        H += -J * qt.tensor(o)
    for i in range(N):
        o = [qt.qeye(2)]*N; o[i] = qt.sigmax()
        H += -h * qt.tensor(o)
    if epsilon != 0:
        for i in range(N):
            o = [qt.qeye(2)]*N; o[i] = qt.sigmaz()
            H += -epsilon * qt.tensor(o)
    return H


# ═══════════════════════════════════════════════════
# PARTIE 1A : ISING GROUND STATE AVEC BRISURE DE SYMÉTRIE
# ═══════════════════════════════════════════════════

def test_closed_symmetry_broken():
    """
    Le test ground state fermé ratait parce que le GS est dégénéré 
    pour h/J < 1 (symétrie Z2 : |↑↑...↑⟩ et |↓↓...↓⟩).
    On ajoute un petit champ longitudinal ε qui brise la symétrie.
    """
    print(f"\n{'='*60}")
    print(f"TEST A : ISING GROUND STATE FERMÉ + ε (N=7)")
    print(f"Brisure Z2 avec ε = 0.01")
    print(f"{'='*60}", flush=True)
    
    N = 7
    epsilon = 0.01  # Petit champ longitudinal
    hJ_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.57, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]
    
    results = []
    for hJ in hJ_values:
        t0 = time.time()
        H = ising_H(N, hJ, epsilon=epsilon)
        _, eigstates = H.eigenstates(eigvals=1, sparse=True)
        psi = eigstates[0]
        
        sz = []
        for i in range(N):
            o = [qt.qeye(2)]*N; o[i] = qt.sigmaz()
            sz.append(qt.expect(qt.tensor(o), psi))
        C, T = compute_cpsi(np.array(sz))
        
        # Gap pour info
        evals = H.eigenenergies(eigvals=2, sparse=True)
        gap = evals[1] - evals[0]
        
        dt = time.time() - t0
        print(f"  hJ={hJ:.2f}  C={C:.4f}  T={T:.4f}  gap={gap:.4f}  ⟨σz⟩₀={sz[0]:.4f}  ({dt:.1f}s)", flush=True)
        results.append({'hJ':hJ, 'C':C, 'T':T, 'gap':gap, 'sz0':sz[0]})
    
    # Analyse
    C_arr = np.array([r['C'] for r in results])
    hJ_arr = np.array([r['hJ'] for r in results])
    
    # QPT visible ?
    C_ordered = np.mean([r['C'] for r in results if r['hJ'] < 0.4])
    C_para = np.mean([r['C'] for r in results if r['hJ'] > 0.8])
    
    print(f"\n  ANALYSE :")
    print(f"    C moyen (h/J < 0.4) = {C_ordered:.4f}")
    print(f"    C moyen (h/J > 0.8) = {C_para:.4f}")
    print(f"    Ratio = {C_ordered/C_para:.2f}x")
    
    if C_ordered > C_para * 1.5:
        print(f"    ✅ QPT VISIBLE sur ground state fermé !")
        print(f"    → 2ème protocole positif pour P3q")
    else:
        print(f"    ❌ Pas de QPT claire")
    
    with open(OUT / 'ising_closed.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)


# ═══════════════════════════════════════════════════
# PARTIE 1B : THERMO N=7
# ═══════════════════════════════════════════════════

def test_thermo_n7():
    print(f"\n{'='*60}")
    print(f"TEST B : THERMO N=7 (C vs E, S_vN, Pureté)")
    print(f"{'='*60}", flush=True)
    
    N = 7; gamma = 0.1
    hJ_values = [0.01, 0.2, 0.4, 0.5, 0.57, 0.65, 0.8, 1.0, 1.5, 2.0]
    
    results = []
    for hJ in hJ_values:
        t0 = time.time()
        H = ising_H(N, hJ)
        c_ops = []
        for i in range(N):
            o = [qt.qeye(2)]*N; o[i] = qt.destroy(2)
            c_ops.append(np.sqrt(gamma) * qt.tensor(o))
        
        try:
            rho = qt.steadystate(H, c_ops, method='direct')
            sz = []
            for i in range(N):
                o = [qt.qeye(2)]*N; o[i] = qt.sigmaz()
                sz.append(qt.expect(qt.tensor(o), rho))
            C, T = compute_cpsi(np.array(sz))
            E = qt.expect(H, rho)
            S_vN = qt.entropy_vn(rho, 2)
            purity = (rho * rho).tr().real
            E_gs = H.eigenenergies(eigvals=1, sparse=True)[0]
            F = E - gamma * S_vN
            
            dt = time.time() - t0
            print(f"  hJ={hJ:.2f}  C={C:.4f}  E={E:.2f}  S={S_vN:.3f}  Pur={purity:.4f}  ({dt:.0f}s)", flush=True)
            results.append({'hJ':hJ,'C':C,'T':T,'E':E,'S_vN':S_vN,'purity':purity,'F':F})
        except Exception as e:
            print(f"  hJ={hJ:.2f}  X {str(e)[:60]}  ({time.time()-t0:.0f}s)", flush=True)
    
    if len(results) < 5: return
    
    C_a = np.array([r['C'] for r in results])
    P_a = np.array([r['purity'] for r in results])
    print(f"\n  CORRÉLATIONS ({len(results)} points) :")
    for name in ['E','S_vN','F','purity']:
        arr = np.array([r[name] for r in results])
        rC, _ = stats.pearsonr(C_a, arr)
        rP, _ = stats.pearsonr(P_a, arr)
        better = "C>P" if abs(rC) > abs(rP) else "P>C"
        print(f"    ρ(C,{name:6s})={rC:+.4f}  ρ(Pur,·)={rP:+.4f}  {better}", flush=True)
    
    with open(OUT / 'thermo_n7.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)


# ═══════════════════════════════════════════════════
# PARTIE 2C : C[Ψ] vs INFORMATION MUTUELLE
# "L'échange observateur ↔ univers"
# ═══════════════════════════════════════════════════

def test_mutual_information():
    """
    Question fondamentale : C[Ψ] mesure-t-il combien d'information
    le système a échangé avec l'environnement ?
    
    Protocole :
    - Système = N qubits dans un état pur |ψ⟩
    - On fait une trace partielle sur k qubits (= "environnement")
    - On mesure C[Ψ] sur les N-k qubits restants (= "système observé")
    - On mesure I(sys:env) = S(sys) + S(env) - S(total)
    
    Si ρ(C, I) > 0.8 → C[Ψ] MESURE l'échange d'information
    """
    print(f"\n{'='*60}")
    print(f"TEST C : C[Ψ] vs INFORMATION MUTUELLE I(sys:env)")
    print(f"L'équation observateur ↔ univers")
    print(f"{'='*60}", flush=True)
    
    N = 6  # 6 qubits total
    gamma_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
    hJ_values = [0.3, 0.6, 1.0]
    
    results = []
    
    for hJ in hJ_values:
        for gamma in gamma_values:
            t0 = time.time()
            H = ising_H(N, hJ)
            c_ops = []
            for i in range(N):
                o = [qt.qeye(2)]*N; o[i] = qt.destroy(2)
                c_ops.append(np.sqrt(gamma) * qt.tensor(o))
            
            try:
                rho = qt.steadystate(H, c_ops, method='direct')
                
                # C[Ψ] du système complet
                sz = []
                for i in range(N):
                    o = [qt.qeye(2)]*N; o[i] = qt.sigmaz()
                    sz.append(qt.expect(qt.tensor(o), rho))
                C_full, T_full = compute_cpsi(np.array(sz))
                
                # Entropie du système complet
                S_total = qt.entropy_vn(rho, 2)
                
                # Trace partielle : garder les 4 premiers qubits ("système observé")
                # Tracer sur les 2 derniers ("environnement")
                rho_sys = rho.ptrace([0, 1, 2, 3])
                rho_env = rho.ptrace([4, 5])
                
                S_sys = qt.entropy_vn(rho_sys, 2)
                S_env = qt.entropy_vn(rho_env, 2)
                
                # Information mutuelle I(sys:env) = S(sys) + S(env) - S(total)
                I_mutual = S_sys + S_env - S_total
                
                # Pureté du sous-système
                purity_sys = (rho_sys * rho_sys).tr().real
                
                # C[Ψ] du sous-système (4 qubits)
                sz_sub = []
                for i in range(4):
                    o = [qt.qeye(2)]*4; o[i] = qt.sigmaz()
                    sz_sub.append(qt.expect(qt.tensor(o), rho_sys))
                C_sub, T_sub = compute_cpsi(np.array(sz_sub))
                
                # Intrication (concurrence pour 2 qubits voisins)
                rho_pair = rho.ptrace([0, 1])
                try:
                    concurrence = qt.concurrence(rho_pair)
                except:
                    concurrence = np.nan
                
                dt = time.time() - t0
                print(f"  hJ={hJ:.1f} γ={gamma:.3f}  C={C_full:.4f}  I={I_mutual:.4f}  S_sys={S_sys:.3f}  conc={concurrence:.3f}  ({dt:.1f}s)", flush=True)
                
                results.append({
                    'hJ': hJ, 'gamma': gamma,
                    'C_full': C_full, 'T_full': T_full,
                    'C_sub': C_sub, 'T_sub': T_sub,
                    'I_mutual': I_mutual,
                    'S_total': S_total, 'S_sys': S_sys, 'S_env': S_env,
                    'purity_sys': purity_sys,
                    'concurrence': concurrence,
                })
            except Exception as e:
                print(f"  hJ={hJ:.1f} γ={gamma:.3f}  X {str(e)[:50]}", flush=True)
    
    if len(results) < 5: return
    
    C_arr = np.array([r['C_full'] for r in results])
    I_arr = np.array([r['I_mutual'] for r in results])
    S_arr = np.array([r['S_total'] for r in results])
    P_arr = np.array([r['purity_sys'] for r in results])
    Conc_arr = np.array([r['concurrence'] for r in results])
    
    print(f"\n  CORRÉLATIONS FONDAMENTALES ({len(results)} configs) :")
    
    r_CI, p_CI = stats.pearsonr(C_arr, I_arr)
    print(f"    ρ(C[Ψ], I_mutual)    = {r_CI:+.4f}  p={p_CI:.4f}")
    verdict = "C MESURE ECHANGE INFO" if abs(r_CI) > 0.7 else "pas de lien direct" if abs(r_CI) < 0.3 else "lien modere"
    print(f"    → {verdict}")
    
    r_CS, p_CS = stats.pearsonr(C_arr, S_arr)
    print(f"    ρ(C[Ψ], S_total)     = {r_CS:+.4f}  p={p_CS:.4f}")
    
    r_CP, p_CP = stats.pearsonr(C_arr, P_arr)
    print(f"    ρ(C[Ψ], Pureté_sys)  = {r_CP:+.4f}  p={p_CP:.4f}")
    
    mask_conc = ~np.isnan(Conc_arr)
    if mask_conc.sum() > 3:
        r_CC, p_CC = stats.pearsonr(C_arr[mask_conc], Conc_arr[mask_conc])
        print(f"    ρ(C[Ψ], Concurrence) = {r_CC:+.4f}  p={p_CC:.4f}")
        if r_CC < -0.5:
            print(f"    → C anticorrele avec intrication = decoherence CREE la structure classique")
    
    # Le test clé : I(sys:env) prédit-il C MIEUX que S_total seul ?
    from numpy.linalg import lstsq
    X1 = np.column_stack([S_arr, np.ones(len(S_arr))])
    b1, _, _, _ = lstsq(X1, C_arr, rcond=None)
    r2_S = 1 - np.sum((C_arr - X1@b1)**2) / np.sum((C_arr - C_arr.mean())**2)
    
    X2 = np.column_stack([S_arr, I_arr, np.ones(len(S_arr))])
    b2, _, _, _ = lstsq(X2, C_arr, rcond=None)
    r2_SI = 1 - np.sum((C_arr - X2@b2)**2) / np.sum((C_arr - C_arr.mean())**2)
    
    print(f"\n  R²(S seul → C)     = {r2_S:.4f}")
    print(f"  R²(S + I → C)     = {r2_SI:.4f}")
    delta_r2 = r2_SI - r2_S
    extra = " <- I ajoute de info !" if delta_r2 > 0.01 else ""
    print(f"  dR2 = {delta_r2:+.4f}{extra}")
    
    with open(OUT / 'mutual_information.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)
    print(f"  Sauvé: {OUT / 'mutual_information.csv'}", flush=True)


# ═══════════════════════════════════════════════════
# PARTIE 2D : DÉCOHÉRENCE PROGRESSIVE
# "Comment la réalité émerge progressivement"
# ═══════════════════════════════════════════════════

def test_decoherence_progressive():
    """
    On part d'un état pur (superposition) et on augmente
    progressivement la décohérence.
    
    Question : C[Ψ] suit-il une courbe universelle
    en fonction de la "quantité de réalité créée" ?
    
    γ = 0 → état pur quantique (superposition)
    γ → ∞ → état classique (décohéré)
    
    Si C(γ) a une forme universelle → il y a une équation
    """
    print(f"\n{'='*60}")
    print(f"TEST D : DÉCOHÉRENCE PROGRESSIVE")
    print(f"Comment la 'réalité classique' émerge")
    print(f"{'='*60}", flush=True)
    
    N = 5
    gamma_values = np.logspace(-3, 0, 20)  # 0.001 à 1.0, 20 points log
    
    for hJ in [0.3, 0.6, 1.0, 2.0]:
        print(f"\n  h/J = {hJ}:", flush=True)
        results = []
        
        for gamma in gamma_values:
            t0 = time.time()
            H = ising_H(N, hJ)
            c_ops = []
            for i in range(N):
                o = [qt.qeye(2)]*N; o[i] = qt.destroy(2)
                c_ops.append(np.sqrt(gamma) * qt.tensor(o))
            
            try:
                rho = qt.steadystate(H, c_ops, method='direct')
                sz = []
                for i in range(N):
                    o = [qt.qeye(2)]*N; o[i] = qt.sigmaz()
                    sz.append(qt.expect(qt.tensor(o), rho))
                C, T = compute_cpsi(np.array(sz))
                S_vN = qt.entropy_vn(rho, 2)
                purity = (rho * rho).tr().real
                
                # L1 coherence (off-diagonal elements)
                rho_full = rho.full()
                l1 = np.sum(np.abs(rho_full)) - np.sum(np.abs(np.diag(rho_full)))
                
                print(f"    γ={gamma:.4f}  C={C:.4f}  S={S_vN:.3f}  L1={l1:.3f}  Pur={purity:.4f}", flush=True)
                results.append({'gamma':gamma, 'C':C, 'T':T, 'S_vN':S_vN, 'purity':purity, 'L1':l1, 'hJ':hJ})
            except:
                pass
        
        if len(results) < 5: continue
        
        C_arr = np.array([r['C'] for r in results])
        L1_arr = np.array([r['L1'] for r in results])
        gamma_arr = np.array([r['gamma'] for r in results])
        
        r_CL, p_CL = stats.pearsonr(C_arr, L1_arr)
        r_Cg, p_Cg = stats.pearsonr(C_arr, np.log10(gamma_arr))
        
        print(f"\n    ρ(C, L1_coherence) = {r_CL:+.4f}  p={p_CL:.4f}")
        print(f"    ρ(C, log10(γ))     = {r_Cg:+.4f}  p={p_Cg:.4f}")
        
        if r_CL < -0.7:
            print(f"    → ✅ C AUGMENTE quand L1 (cohérence quantique) DIMINUE")
            print(f"    → La décohérence CRÉE la structure classique mesurée par C[Ψ]")
        
        with open(OUT / f'decoherence_hJ{hJ}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader(); w.writerows(results)


# ═══════════════════════════════════════════════════
# PARTIE 2E : QUANTUM DARWINISM
# "L'univers copie l'information pour les observateurs"
# ═══════════════════════════════════════════════════

def test_quantum_darwinism():
    """
    Quantum Darwinism (Zurek 2009) : l'information sur un système 
    est copiée de manière redondante dans l'environnement.
    
    Test : C[Ψ] mesure-t-il cette redondance ?
    
    On calcule C[Ψ] sur des fractions croissantes de l'environnement
    et on regarde si C sature (= information redondante = "réalité objective")
    """
    print(f"\n{'='*60}")
    print(f"TEST E : QUANTUM DARWINISM")
    print(f"L'univers copie-t-il l'info pour les observateurs ?")
    print(f"{'='*60}", flush=True)
    
    N = 6  # 6 qubits
    gamma = 0.1
    
    for hJ in [0.3, 1.0]:
        print(f"\n  h/J = {hJ}:", flush=True)
        
        H = ising_H(N, hJ)
        c_ops = []
        for i in range(N):
            o = [qt.qeye(2)]*N; o[i] = qt.destroy(2)
            c_ops.append(np.sqrt(gamma) * qt.tensor(o))
        
        try:
            rho = qt.steadystate(H, c_ops, method='direct')
        except:
            print(f"    Échec steadystate"); continue
        
        # C[Ψ] sur des sous-systèmes de taille croissante
        for n_keep in range(2, N+1):
            keep = list(range(n_keep))
            rho_sub = rho.ptrace(keep)
            
            sz_sub = []
            for i in range(n_keep):
                o = [qt.qeye(2)]*n_keep; o[i] = qt.sigmaz()
                sz_sub.append(qt.expect(qt.tensor(o), rho_sub))
            
            C, T = compute_cpsi(np.array(sz_sub))
            S = qt.entropy_vn(rho_sub, 2)
            frac = n_keep / N
            
            print(f"    n={n_keep}/{N} ({frac:.0%})  C={C:.4f}  T={T:.4f}  S={S:.3f}", flush=True)
        
        print(f"    → Si C sature rapidement → redondance → 'réalité objective' accessible")
        print(f"    → Si C augmente linéairement → pas de redondance → 'réalité subjective'")


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"{'='*60}")
    print(f"C[Ψ] DEEP — TESTS FONDAMENTAUX")
    print(f"{'='*60}", flush=True)
    
    t0 = time.time()
    
    # Partie 1 : renforcer P3q (rapide)
    test_closed_symmetry_broken()    # ~30s
    
    # Partie 2 : l'équation observateur-univers
    test_mutual_information()         # ~5 min (N=6)
    test_decoherence_progressive()    # ~10 min (N=5, 20×4 points)
    test_quantum_darwinism()          # ~2 min (N=6)
    
    # Partie 1B : thermo N=7 (le plus long, en dernier)
    test_thermo_n7()                  # ~2-3h
    
    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"TOTAL: {total/60:.1f} min")
    print(f"Résultats: {OUT}")
    print(f"{'='*60}")
