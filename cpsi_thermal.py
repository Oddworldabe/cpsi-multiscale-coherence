#!/usr/bin/env python3
"""
cpsi_thermal.py — Tests fondamentaux restants

PARTIE 1 : C vs Pureté sur états THERMIQUES (Gibbs) N=7 à 12
  ρ_thermal = exp(-βH) / Z → états mixtes SANS Lindblad
  → Rapide (diagonalisation seule)
  → Si C > Pureté à N≥7 → PRL

PARTIE 2 : Pourquoi C_eq(B) = 1/3 ?
  Analyse analytique de la phase paramagnétique

PARTIE 3 : C comme principe variationnel ?
  Test : dC/dt ≥ 0 pendant la relaxation ?

python3 -u ~/Desktop/cpsi_thermal.py 2>&1 | tee ~/Desktop/thermal_log.txt
"""
import numpy as np
import time, warnings, csv
warnings.filterwarnings('ignore')
import qutip as qt
from scipy import stats
from pathlib import Path

print(f"qutip {qt.__version__}", flush=True)
OUT = Path.home() / "Desktop" / "cpsi_thermal"
OUT.mkdir(exist_ok=True)


def compute_cpsi(sz):
    N = len(sz)
    micro = np.mean([abs(sz[i]*sz[i+1]) for i in range(N-1)])
    macro = np.mean([abs(sz[i]*sz[j]) for i in range(N) for j in range(i+1,N)])
    meso = max(0, min(1, 1.0 - np.std(np.abs(sz))))
    C = (micro + macro + meso) / 3
    T = abs(micro-meso) + abs(meso-macro) + abs(micro-macro)
    return C, T


def ising_H(N, hJ, J=1.0, epsilon=0.01):
    h = hJ * J
    H = 0
    for i in range(N-1):
        o = [qt.qeye(2)]*N; o[i] = qt.sigmaz(); o[i+1] = qt.sigmaz()
        H += -J * qt.tensor(o)
    for i in range(N):
        o = [qt.qeye(2)]*N; o[i] = qt.sigmax()
        H += -h * qt.tensor(o)
        o2 = [qt.qeye(2)]*N; o2[i] = qt.sigmaz()
        H += -epsilon * qt.tensor(o2)
    return H


def thermal_state(H, beta):
    """État de Gibbs ρ = exp(-βH) / Z"""
    evals, estates = H.eigenstates()
    dim = len(evals)
    # Boltzmann weights
    weights = np.exp(-beta * (evals - evals.min()))  # shift pour stabilité
    Z = weights.sum()
    probs = weights / Z
    # ρ = Σ p_i |ψ_i⟩⟨ψ_i|
    rho = sum(probs[i] * qt.ket2dm(estates[i]) for i in range(dim))
    return rho


# ═══════════════════════════════════════════════════
# PARTIE 1 : C vs PURETÉ sur états thermiques
# ═══════════════════════════════════════════════════

def test_c_vs_purity():
    print(f"\n{'='*60}")
    print(f"PARTIE 1 : C vs PURETÉ — états thermiques Gibbs")
    print(f"N = 7 à 12, températures variées")
    print(f"{'='*60}", flush=True)
    
    # β = 1/T : petit β = haute température (mixte), grand β = basse température (pur)
    beta_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    hJ_values = [0.3, 0.6, 1.0]
    
    for N in [7, 8, 9, 10, 11, 12]:
        dim = 2**N
        if dim > 4096:
            print(f"\n  N={N} (dim={dim}): eigenstates complet trop lent, skip", flush=True)
            continue
            
        print(f"\n  N={N} (dim={dim}):", flush=True)
        results = []
        
        for hJ in hJ_values:
            t0 = time.time()
            H = ising_H(N, hJ)
            
            # Full diagonalisation (nécessaire pour état thermique)
            try:
                evals, estates = H.eigenstates()
            except Exception as e:
                print(f"    hJ={hJ} diag failed: {str(e)[:50]}", flush=True)
                continue
            
            for beta in beta_values:
                # Boltzmann weights
                weights = np.exp(-beta * (evals - evals.min()))
                Z = weights.sum()
                probs = weights / Z
                
                # ρ thermique
                rho = sum(probs[i] * qt.ket2dm(estates[i]) for i in range(len(evals)))
                
                # <σz> pour chaque qubit
                sz = []
                for i in range(N):
                    o = [qt.qeye(2)]*N; o[i] = qt.sigmaz()
                    sz.append(qt.expect(qt.tensor(o), rho))
                
                C, T = compute_cpsi(np.array(sz))
                purity = (rho * rho).tr().real
                S_vN = qt.entropy_vn(rho, 2)
                E = qt.expect(H, rho)
                
                results.append({
                    'N':N, 'hJ':hJ, 'beta':beta, 'C':C, 'T':T,
                    'purity':purity, 'S_vN':S_vN, 'E':E
                })
            
            dt = time.time() - t0
            print(f"    hJ={hJ:.1f}: {len(beta_values)} temperatures en {dt:.1f}s", flush=True)
        
        if len(results) < 10: continue
        
        # ANALYSE : C vs Pureté
        C_a = np.array([r['C'] for r in results])
        P_a = np.array([r['purity'] for r in results])
        E_a = np.array([r['E'] for r in results])
        S_a = np.array([r['S_vN'] for r in results])
        
        rCP, _ = stats.pearsonr(C_a, P_a)
        
        # Test clé : C prédit-il E MIEUX que Pureté ?
        from numpy.linalg import lstsq
        
        # R²(Pureté → E)
        X_p = np.column_stack([P_a, np.ones(len(P_a))])
        b_p, _, _, _ = lstsq(X_p, E_a, rcond=None)
        r2_P = 1 - np.sum((E_a - X_p@b_p)**2) / np.sum((E_a - E_a.mean())**2)
        
        # R²(C → E)
        X_c = np.column_stack([C_a, np.ones(len(C_a))])
        b_c, _, _, _ = lstsq(X_c, E_a, rcond=None)
        r2_C = 1 - np.sum((E_a - X_c@b_c)**2) / np.sum((E_a - E_a.mean())**2)
        
        # R²(C + Pureté → E)
        X_cp = np.column_stack([C_a, P_a, np.ones(len(C_a))])
        b_cp, _, _, _ = lstsq(X_cp, E_a, rcond=None)
        r2_CP = 1 - np.sum((E_a - X_cp@b_cp)**2) / np.sum((E_a - E_a.mean())**2)
        
        delta_r2 = r2_CP - r2_P
        
        print(f"\n    RÉSULTATS N={N} ({len(results)} configs) :")
        print(f"    rho(C, Pureté) = {rCP:+.4f}")
        print(f"    R2(Pureté -> E) = {r2_P:.4f}")
        print(f"    R2(C -> E)      = {r2_C:.4f}  {'C > P !' if r2_C > r2_P else 'P > C'}")
        print(f"    R2(C+P -> E)    = {r2_CP:.4f}")
        print(f"    Delta R2        = {delta_r2:+.4f}  {'C ajoute info !' if delta_r2 > 0.01 else ''}", flush=True)
        
        # Sauver
        with open(OUT / f'thermal_N{N}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader(); w.writerows(results)


# ═══════════════════════════════════════════════════
# PARTIE 2 : POURQUOI C_eq(B) = 1/3 ?
# ═══════════════════════════════════════════════════

def test_ceq_analysis():
    print(f"\n{'='*60}")
    print(f"PARTIE 2 : POURQUOI C_eq(B) = 1/3 ?")
    print(f"{'='*60}", flush=True)
    
    print(f"\n  ANALYSE ANALYTIQUE :")
    print(f"  En phase paramagnétique (h >> J), tous les spins s'alignent sur x.")
    print(f"  Donc <sigma_z^i> -> 0 pour tout i.")
    print(f"  ")
    print(f"  r_micro = mean(|sz_i * sz_(i+1)|) -> 0  (produits de ~0)")
    print(f"  r_macro = mean(|sz_i * sz_j|)     -> 0  (idem)")
    print(f"  r_meso  = 1 - std(|sz|)           -> 1  (std de valeurs ~0 = 0)")
    print(f"  ")
    print(f"  C = (0 + 0 + 1) / 3 = 1/3 = 0.333...")
    print(f"  ")
    print(f"  C'est DEFINITIONNELLEMENT 1/3 quand tous les observables sont nuls.")
    print(f"  Ce n'est PAS une constante physique profonde.")
    print(f"  C'est la valeur plancher de C quand le système est homogène et désordonné.", flush=True)
    
    # Vérification numérique
    print(f"\n  VÉRIFICATION NUMÉRIQUE :", flush=True)
    print(f"  Phase paramagnétique profonde (h/J=2.0) :")
    
    for N in [4, 7, 10, 14]:
        sz = np.zeros(N)  # tous les <σz> = 0
        # Ajouter un tout petit bruit (comme dans la réalité numérique)
        sz += np.random.randn(N) * 1e-8
        C, T = compute_cpsi(sz)
        print(f"    N={N:2d}: C = {C:.10f}  (deviation de 1/3 = {abs(C-1/3):.2e})")
    
    # Test : que se passe-t-il avec une définition ALTERNATIVE ?
    print(f"\n  DÉFINITIONS ALTERNATIVES :", flush=True)
    
    # Alternative 1 : C = (micro + macro + meso) / 3 mais meso = 1 - std/max_std
    # Alternative 2 : C = sqrt(micro * macro * meso)  (géométrique)
    # Alternative 3 : C = (micro + macro) / 2  (sans meso)
    
    for N in [7, 10, 14]:
        H = ising_H(N, 2.0)  # Phase B profonde
        _, estates = H.eigenstates(eigvals=1, sparse=True)
        psi = estates[0]
        sz = []
        for i in range(N):
            o = [qt.qeye(2)]*N; o[i] = qt.sigmaz()
            sz.append(qt.expect(qt.tensor(o), psi))
        sz = np.array(sz)
        
        micro = np.mean([abs(sz[i]*sz[i+1]) for i in range(N-1)])
        macro = np.mean([abs(sz[i]*sz[j]) for i in range(N) for j in range(i+1,N)])
        meso = max(0, min(1, 1.0 - np.std(np.abs(sz))))
        
        C_standard = (micro + macro + meso) / 3
        C_geom = (micro * macro * meso) ** (1/3) if micro > 0 and macro > 0 else 0
        C_sans_meso = (micro + macro) / 2
        
        print(f"    N={N:2d}: standard={C_standard:.4f}  geom={C_geom:.4f}  sans_meso={C_sans_meso:.4f}")
        print(f"          micro={micro:.6f}  macro={macro:.6f}  meso={meso:.6f}")
    
    # Phase A (ordonnée) — même exercice
    print(f"\n  PHASE ORDONNÉE (h/J=0.01) :", flush=True)
    for N in [7, 10, 14]:
        H = ising_H(N, 0.01)
        _, estates = H.eigenstates(eigvals=1, sparse=True)
        psi = estates[0]
        sz = []
        for i in range(N):
            o = [qt.qeye(2)]*N; o[i] = qt.sigmaz()
            sz.append(qt.expect(qt.tensor(o), psi))
        sz = np.array(sz)
        
        micro = np.mean([abs(sz[i]*sz[i+1]) for i in range(N-1)])
        macro = np.mean([abs(sz[i]*sz[j]) for i in range(N) for j in range(i+1,N)])
        meso = max(0, min(1, 1.0 - np.std(np.abs(sz))))
        
        C_standard = (micro + macro + meso) / 3
        print(f"    N={N:2d}: C={C_standard:.6f}  micro={micro:.6f}  macro={macro:.6f}  meso={meso:.6f}")


# ═══════════════════════════════════════════════════
# PARTIE 3 : C COMME PRINCIPE VARIATIONNEL
# dC/dt ≥ 0 pendant la relaxation ?
# ═══════════════════════════════════════════════════

def test_variational():
    """
    Si C[Ψ] est un principe variationnel (comme l'entropie),
    alors C devrait augmenter monotoniquement pendant la relaxation
    d'un état initial aléatoire vers le steady state.
    
    Test : on simule mesolve avec un état initial random
    et on vérifie que C(t) est monotone croissant.
    """
    print(f"\n{'='*60}")
    print(f"PARTIE 3 : C EST-IL VARIATIONNEL ?")
    print(f"dC/dt >= 0 pendant la relaxation ?")
    print(f"{'='*60}", flush=True)
    
    N = 4  # petit pour vitesse
    gamma = 0.1
    
    for hJ in [0.3, 0.6, 1.0]:
        print(f"\n  h/J = {hJ}:", flush=True)
        H = ising_H(N, hJ, epsilon=0)
        
        c_ops = []
        for i in range(N):
            o = [qt.qeye(2)]*N; o[i] = qt.destroy(2)
            c_ops.append(np.sqrt(gamma) * qt.tensor(o))
        
        # État initial aléatoire (état pur random)
        dim = 2**N
        psi0 = qt.rand_ket(dim)
        rho0 = qt.ket2dm(psi0)
        
        # Observables σz pour chaque qubit
        sz_ops = []
        for i in range(N):
            o = [qt.qeye(2)]*N; o[i] = qt.sigmaz()
            sz_ops.append(qt.tensor(o))
        
        # mesolve
        tlist = np.linspace(0, 50, 200)
        result = qt.mesolve(H, rho0, tlist, c_ops, sz_ops)
        
        # C(t)
        C_t = []
        for t_idx in range(len(tlist)):
            sz = np.array([result.expect[i][t_idx] for i in range(N)])
            C, T = compute_cpsi(sz)
            C_t.append(C)
        C_t = np.array(C_t)
        
        # Est-ce monotone ?
        dC = np.diff(C_t)
        n_positive = np.sum(dC > 0)
        n_negative = np.sum(dC < 0)
        monotone = n_negative == 0
        
        C_init = C_t[0]
        C_final = C_t[-1]
        C_min = C_t.min()
        C_max = C_t.max()
        
        print(f"    C(0) = {C_init:.4f}  C(inf) = {C_final:.4f}")
        print(f"    C_min = {C_min:.4f}  C_max = {C_max:.4f}")
        print(f"    dC > 0 : {n_positive}/{len(dC)}  dC < 0 : {n_negative}/{len(dC)}")
        
        if monotone:
            print(f"    --> MONOTONE CROISSANT (dC/dt >= 0 toujours)")
        elif n_negative < len(dC) * 0.05:
            print(f"    --> QUASI-MONOTONE ({n_negative} violations sur {len(dC)})")
        else:
            print(f"    --> PAS MONOTONE (C oscille pendant relaxation)")
    
    # Test avec 10 états initiaux aléatoires
    print(f"\n  TEST ROBUSTE : 10 états initiaux random, h/J=0.3 :", flush=True)
    hJ = 0.3
    H = ising_H(N, hJ, epsilon=0)
    c_ops = []
    for i in range(N):
        o = [qt.qeye(2)]*N; o[i] = qt.destroy(2)
        c_ops.append(np.sqrt(gamma) * qt.tensor(o))
    
    monotone_count = 0
    for trial in range(10):
        psi0 = qt.rand_ket(dim)
        rho0 = qt.ket2dm(psi0)
        result = qt.mesolve(H, rho0, tlist, c_ops, sz_ops)
        C_t = []
        for t_idx in range(len(tlist)):
            sz = np.array([result.expect[i][t_idx] for i in range(N)])
            C, _ = compute_cpsi(sz)
            C_t.append(C)
        dC = np.diff(C_t)
        n_neg = np.sum(dC < -1e-6)  # seuil numérique
        if n_neg == 0:
            monotone_count += 1
        print(f"    Trial {trial}: C(0)={C_t[0]:.3f} -> C(inf)={C_t[-1]:.3f}  violations={n_neg}  {'MONOTONE' if n_neg==0 else 'oscille'}", flush=True)
    
    print(f"\n    VERDICT : {monotone_count}/10 trajectoires monotones")
    if monotone_count >= 8:
        print(f"    --> C[Psi] se comporte COMME un potentiel variationnel")
        print(f"    --> dC/dt >= 0 est une bonne approximation")
        print(f"    --> Analogue à dS/dt >= 0 (2ème loi)")
    elif monotone_count >= 5:
        print(f"    --> Tendance variationnelle mais pas stricte")
    else:
        print(f"    --> C N'EST PAS variationnel (oscille)")


# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"{'='*60}")
    print(f"C[Psi] THERMAL — Tests fondamentaux")
    print(f"{'='*60}", flush=True)
    
    t0 = time.time()
    
    test_ceq_analysis()     # ~30s
    test_variational()      # ~2 min
    test_c_vs_purity()      # ~10-30 min selon N max
    
    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"TOTAL: {total/60:.1f} min")
    print(f"{'='*60}")
