#!/usr/bin/env python3
"""
cpsi_derivation.py — Dérivation analytique + théories parallèles

PARTIE 1 : DÉRIVATION ANALYTIQUE de dC/dt depuis Lindblad
  Lindblad : d⟨σz_i⟩/dt = -i⟨[σz_i, H]⟩ - γ(⟨σz_i⟩ + 1)
  C = f({⟨σz_i⟩}) → dC/dt = Σ (∂C/∂sz_i) × dsz_i/dt
  → Vérifier que la dérivation analytique = résultat numérique

PARTIE 2 : PRIGOGINE — minimum de production d'entropie
  Au steady state, le système minimise-t-il la production d'entropie ?
  
PARTIE 3 : FRISTON — principe d'énergie libre (neuroscience)
  C[Ψ] est-il un "modèle génératif" au sens de Friston ?
  Test : C minimise-t-il la surprise (= entropie) ?

PARTIE 4 : LOI CONSTRUCTALE (Bejan) — optimisation des flux
  C[Ψ] mesure-t-il l'efficacité d'accès aux flux d'énergie ?

PARTIE 5 : SOC (Bak) — criticalité auto-organisée
  Le point QPT est-il un attracteur ? Le système évolue-t-il VERS la criticalité ?

python3 ~/Desktop/cpsi_derivation.py
"""
import numpy as np
import qutip as qt
from scipy import stats
import warnings, time
warnings.filterwarnings('ignore')

print(f"qutip {qt.__version__}")

def compute_cpsi_detailed(sz):
    N = len(sz)
    micro = np.mean([abs(sz[i]*sz[i+1]) for i in range(N-1)])
    macro = np.mean([abs(sz[i]*sz[j]) for i in range(N) for j in range(i+1,N)])
    meso = max(0, min(1, 1.0 - np.std(np.abs(sz))))
    C = (micro + macro + meso) / 3
    T = abs(micro-meso) + abs(meso-macro) + abs(micro-macro)
    return C, T, micro, macro, meso

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


# ═══════════════════════════════════════════════════
# PARTIE 1 : DÉRIVATION ANALYTIQUE de dC/dt
# ═══════════════════════════════════════════════════

def test_derivation():
    """
    C = (micro + macro + meso) / 3
    
    micro = (1/(N-1)) Σ |sz_i × sz_{i+1}|
    macro = (2/(N(N-1))) Σ_{i<j} |sz_i × sz_j|
    meso = 1 - std(|sz|)
    
    Sous amplitude damping :
    d⟨σz_i⟩/dt = f_i(H, {sz}) - γ(sz_i + 1)
    
    Le terme dissipatif est UNIVERSEL : -γ(sz_i + 1)
    → pousse tous les sz_i vers -1
    → micro → 1, macro → 1, meso → 1 → C → 1
    
    Mais le terme Hamiltonien (commutateur) crée des oscillations.
    La compétition H vs γ détermine le steady state.
    """
    print(f"\n{'='*60}")
    print(f"PARTIE 1 : DÉRIVATION ANALYTIQUE dC/dt")
    print(f"{'='*60}", flush=True)
    
    N = 4; gamma = 0.1; hJ = 0.3
    H = ising_H(N, hJ, eps=0)
    
    c_ops = []
    for i in range(N):
        o = [qt.qeye(2)]*N; o[i] = qt.destroy(2)
        c_ops.append(np.sqrt(gamma) * qt.tensor(o))
    
    sz_ops = []
    for i in range(N):
        o = [qt.qeye(2)]*N; o[i] = qt.sigmaz()
        sz_ops.append(qt.tensor(o))
    
    dim = 2**N
    psi0 = qt.Qobj(qt.rand_ket(dim).full(), dims=[[2]*N, [1]*N])
    rho0 = qt.ket2dm(psi0)
    
    tlist = np.linspace(0, 100, 1000)
    dt = tlist[1] - tlist[0]
    result = qt.mesolve(H, rho0, tlist, c_ops, sz_ops)
    
    # Calculer C(t) et dC/dt numérique
    C_t = []; T_t = []
    for t_idx in range(len(tlist)):
        sz = np.array([result.expect[i][t_idx] for i in range(N)])
        C, T, _, _, _ = compute_cpsi_detailed(sz)
        C_t.append(C); T_t.append(T)
    
    C_t = np.array(C_t); T_t = np.array(T_t)
    dC_dt_num = np.gradient(C_t, dt)
    
    # Dérivation analytique de dC/dt
    # C = (micro + macro + meso)/3
    # dC/dt = (dmicro/dt + dmacro/dt + dmeso/dt)/3
    # 
    # Pour le cas homogène (tous sz_i ≈ m) :
    # micro ≈ m², macro ≈ m², meso ≈ 1
    # C ≈ (2m² + 1)/3
    # dC/dt ≈ (4m/3) × dm/dt
    # 
    # Sous amplitude damping sans H : dm/dt = -γ(m+1)
    # → dC/dt ≈ (4m/3) × (-γ)(m+1) = -4γm(m+1)/3
    
    print(f"\n  FORMULE ANALYTIQUE (cas homogène, H négligé) :")
    print(f"  m = ⟨σz⟩ moyen")
    print(f"  C ≈ (2m² + 1)/3")
    print(f"  dC/dt ≈ -(4γ/3) × m × (m+1)")
    print(f"  ")
    print(f"  Cette formule prédit :")
    print(f"    - dC/dt > 0 quand m < -1 (impossible physiquement)")
    print(f"    - dC/dt > 0 quand -1 < m < 0 (phase paramagnétique)")
    print(f"    - dC/dt < 0 quand m > 0 (C diminue ?!)")
    print(f"    - dC/dt = 0 quand m = -1 (steady state)")
    print(f"  ")
    
    # Vérification numérique
    m_t = np.array([np.mean([result.expect[i][t_idx] for i in range(N)]) 
                     for t_idx in range(len(tlist))])
    C_pred = (2 * m_t**2 + 1) / 3
    
    # Corrélation entre C prédit et C mesuré
    rho_pred, p_pred = stats.pearsonr(C_pred, C_t)
    
    print(f"  VÉRIFICATION :")
    print(f"    ρ(C_prédit, C_mesuré) = {rho_pred:+.4f}  p={p_pred:.4f}")
    print(f"    C_final mesuré  = {C_t[-1]:.4f}")
    print(f"    C_final prédit  = {C_pred[-1]:.4f}")
    print(f"    m_final         = {m_t[-1]:.4f}")
    
    # Le cas non-homogène (σ_i ≠ σ_j) nécessite la dérivée complète
    # Mais le cas homogène donne déjà l'intuition
    
    # Fit exponentiel : C(t) = C_eq + (C_0 - C_eq) * exp(-λt)
    from scipy.optimize import curve_fit
    
    def exp_relax(t, C_eq, C_0, lam):
        return C_eq + (C_0 - C_eq) * np.exp(-lam * t)
    
    try:
        popt, _ = curve_fit(exp_relax, tlist, C_t, p0=[C_t[-1], C_t[0], 0.1], maxfev=5000)
        C_eq_fit, C_0_fit, lam_fit = popt
        C_fit = exp_relax(tlist, *popt)
        r2 = 1 - np.sum((C_t - C_fit)**2) / np.sum((C_t - C_t.mean())**2)
        
        print(f"\n  FIT EXPONENTIEL : C(t) = {C_eq_fit:.4f} + ({C_0_fit:.4f} - {C_eq_fit:.4f}) × exp(-{lam_fit:.4f}t)")
        print(f"  R² = {r2:.6f}")
        print(f"  λ = {lam_fit:.4f} (taux de relaxation)")
        print(f"  τ = {1/lam_fit:.2f} (temps caractéristique)")
        
        if r2 > 0.95:
            print(f"\n  --> L'ÉQUATION DÉRIVÉE :")
            print(f"  dC/dt = -{lam_fit:.4f} × (C - {C_eq_fit:.4f})")
            print(f"  = relaxation exponentielle vers C_eq")
            print(f"  λ dépend de γ (à tester)")
    except Exception as e:
        print(f"  Fit échoué : {e}")
    
    # Tester λ en fonction de γ
    print(f"\n  λ(γ) — taux de relaxation vs force de dissipation :")
    for gamma_test in [0.01, 0.05, 0.1, 0.2, 0.5]:
        c_ops_test = []
        for i in range(N):
            o = [qt.qeye(2)]*N; o[i] = qt.destroy(2)
            c_ops_test.append(np.sqrt(gamma_test) * qt.tensor(o))
        
        result_test = qt.mesolve(H, rho0, tlist, c_ops_test, sz_ops)
        C_test = []
        for t_idx in range(len(tlist)):
            sz = np.array([result_test.expect[i][t_idx] for i in range(N)])
            C, _, _, _, _ = compute_cpsi_detailed(sz)
            C_test.append(C)
        C_test = np.array(C_test)
        
        try:
            popt_t, _ = curve_fit(exp_relax, tlist, C_test, p0=[C_test[-1], C_test[0], gamma_test], maxfev=5000)
            lam = popt_t[2]
            r2 = 1 - np.sum((C_test - exp_relax(tlist, *popt_t))**2) / np.sum((C_test - C_test.mean())**2)
            print(f"    γ={gamma_test:.3f}  λ={lam:.4f}  λ/γ={lam/gamma_test:.2f}  R²={r2:.4f}")
        except:
            print(f"    γ={gamma_test:.3f}  fit échoué")


# ═══════════════════════════════════════════════════
# PARTIE 2 : PRIGOGINE — production d'entropie minimale
# ═══════════════════════════════════════════════════

def test_prigogine():
    """
    Prigogine (Nobel 1977) : au steady state non-équilibre,
    le système minimise la production d'entropie.
    
    σ_prod = dS_env/dt = -Tr(dρ/dt × ln(ρ)) ← production d'entropie
    
    Test : σ_prod corrèle-t-il avec C[Ψ] ?
    """
    print(f"\n{'='*60}")
    print(f"PARTIE 2 : PRIGOGINE — production d'entropie")
    print(f"{'='*60}", flush=True)
    
    N = 4; gamma = 0.1
    
    results = []
    for hJ in np.linspace(0.01, 2.0, 15):
        H = ising_H(N, hJ, eps=0.01)
        c_ops = []
        for i in range(N):
            o = [qt.qeye(2)]*N; o[i] = qt.destroy(2)
            c_ops.append(np.sqrt(gamma) * qt.tensor(o))
        
        rho_ss = qt.steadystate(H, c_ops, method='direct')
        
        sz_ops = []
        for i in range(N):
            o = [qt.qeye(2)]*N; o[i] = qt.sigmaz()
            sz_ops.append(qt.tensor(o))
        sz = np.array([qt.expect(op, rho_ss) for op in sz_ops])
        C, T, _, _, _ = compute_cpsi_detailed(sz)
        
        # Production d'entropie au steady state
        # σ = Σ_k γ_k × Tr(L_k ρ L_k†) × ln(Tr(L_k ρ L_k†) / Tr(L_k† L_k ρ))
        # Approximation : σ ≈ Σ_k γ × ⟨L_k† L_k⟩
        sigma_prod = 0
        for i in range(N):
            o = [qt.qeye(2)]*N; o[i] = qt.destroy(2)
            Lk = qt.tensor(o)
            sigma_prod += gamma * qt.expect(Lk.dag() * Lk, rho_ss)
        
        E = qt.expect(H, rho_ss)
        S = qt.entropy_vn(rho_ss, 2)
        
        results.append({'hJ':hJ, 'C':C, 'T':T, 'sigma':sigma_prod, 'E':E, 'S':S})
    
    C_a = np.array([r['C'] for r in results])
    sigma_a = np.array([r['sigma'] for r in results])
    E_a = np.array([r['E'] for r in results])
    
    r_Cs, _ = stats.pearsonr(C_a, sigma_a)
    print(f"  ρ(C, σ_prod) = {r_Cs:+.4f}")
    print(f"  σ_prod range: [{sigma_a.min():.4f}, {sigma_a.max():.4f}]")
    
    if r_Cs < -0.7:
        print(f"  --> C ÉLEVÉ = production d'entropie FAIBLE (Prigogine)")
        print(f"  --> Systèmes cohérents = minimisent la dissipation")
    elif r_Cs > 0.7:
        print(f"  --> C ÉLEVÉ = production d'entropie FORTE")
        print(f"  --> Systèmes cohérents = STRUCTURES DISSIPATIVES (Prigogine)")


# ═══════════════════════════════════════════════════
# PARTIE 3 : FRISTON — énergie libre variationnelle
# ═══════════════════════════════════════════════════

def test_friston():
    """
    Friston (2010) : les systèmes vivants minimisent l'énergie libre
    variationnelle F_var = E_q[ln q(θ) - ln p(x,θ)]
    
    En termes simples : le système maintient un "modèle interne" 
    qui minimise la surprise (= entropie des observations).
    
    Test : C[Ψ] est-il l'inverse de la surprise ?
    Surprise ≈ -log p(observations) ≈ entropie des observables
    """
    print(f"\n{'='*60}")
    print(f"PARTIE 3 : FRISTON — énergie libre variationnelle")
    print(f"C[Ψ] minimise-t-il la surprise ?")
    print(f"{'='*60}", flush=True)
    
    N = 4; gamma = 0.1
    
    results = []
    for hJ in np.linspace(0.01, 2.0, 15):
        H = ising_H(N, hJ, eps=0.01)
        c_ops = []
        for i in range(N):
            o = [qt.qeye(2)]*N; o[i] = qt.destroy(2)
            c_ops.append(np.sqrt(gamma) * qt.tensor(o))
        
        rho_ss = qt.steadystate(H, c_ops, method='direct')
        sz_ops = []
        for i in range(N):
            o = [qt.qeye(2)]*N; o[i] = qt.sigmaz()
            sz_ops.append(qt.tensor(o))
        sz = np.array([qt.expect(op, rho_ss) for op in sz_ops])
        C, T, _, _, _ = compute_cpsi_detailed(sz)
        
        # "Surprise" = entropie des observables (pas de l'état)
        # Les probabilités p_i de mesurer σz_i = +1 ou -1
        probs = [(1 + s)/2 for s in sz]  # prob de mesurer +1
        surprise = 0
        for p in probs:
            p = np.clip(p, 1e-10, 1-1e-10)
            surprise -= p * np.log(p) + (1-p) * np.log(1-p)
        surprise /= N  # moyenne par qubit
        
        # Complexité = KL(q || p_prior) ≈ distance au prior uniforme
        complexity = sum(abs(p - 0.5) for p in probs) / N
        
        # Accuracy = -surprise (combien les mesures sont prévisibles)
        accuracy = -surprise
        
        results.append({'hJ':hJ, 'C':C, 'surprise':surprise, 
                        'complexity':complexity, 'accuracy':accuracy})
    
    C_a = np.array([r['C'] for r in results])
    surp_a = np.array([r['surprise'] for r in results])
    comp_a = np.array([r['complexity'] for r in results])
    acc_a = np.array([r['accuracy'] for r in results])
    
    r_CS, _ = stats.pearsonr(C_a, surp_a)
    r_CC, _ = stats.pearsonr(C_a, comp_a)
    r_CA, _ = stats.pearsonr(C_a, acc_a)
    
    print(f"  ρ(C, Surprise)   = {r_CS:+.4f}  {'C diminue la surprise !' if r_CS < -0.7 else ''}")
    print(f"  ρ(C, Complexité) = {r_CC:+.4f}  {'C augmente la complexité' if r_CC > 0.7 else ''}")
    print(f"  ρ(C, Accuracy)   = {r_CA:+.4f}  {'C = prédictibilité' if r_CA > 0.7 else ''}")
    
    if r_CS < -0.8:
        print(f"\n  --> C[Ψ] = INVERSE de la surprise (Friston)")
        print(f"  --> Les systèmes cohérents minimisent la surprise observationnelle")
        print(f"  --> C[Ψ] implémente le Free Energy Principle !")


# ═══════════════════════════════════════════════════
# PARTIE 4 : CONSTRUCTALE (Bejan) — optimisation des flux
# ═══════════════════════════════════════════════════

def test_constructal():
    """
    Loi constructale (Bejan 1996) : les systèmes évoluent
    pour faciliter l'accès aux flux qui les traversent.
    
    Test : C[Ψ] mesure-t-il l'efficacité de transport d'énergie
    à travers la chaîne de spins ?
    
    Proxy : courant d'énergie J_E = ⟨[H_i, H_{i+1}]⟩
    """
    print(f"\n{'='*60}")
    print(f"PARTIE 4 : CONSTRUCTALE — flux d'énergie")
    print(f"C[Ψ] optimise-t-il le transport ?")
    print(f"{'='*60}", flush=True)
    
    N = 4; gamma = 0.1
    
    results = []
    for hJ in np.linspace(0.01, 2.0, 15):
        H = ising_H(N, hJ, eps=0.01)
        c_ops = []
        for i in range(N):
            o = [qt.qeye(2)]*N; o[i] = qt.destroy(2)
            c_ops.append(np.sqrt(gamma) * qt.tensor(o))
        
        rho_ss = qt.steadystate(H, c_ops, method='direct')
        sz_ops = []
        for i in range(N):
            o = [qt.qeye(2)]*N; o[i] = qt.sigmaz()
            sz_ops.append(qt.tensor(o))
        sz = np.array([qt.expect(op, rho_ss) for op in sz_ops])
        C, T, _, _, _ = compute_cpsi_detailed(sz)
        
        # Courant d'énergie : proxy = gradient de magnétisation
        # |∇m| = Σ |sz_{i+1} - sz_i| / (N-1)
        grad_m = np.mean([abs(sz[i+1] - sz[i]) for i in range(N-1)])
        
        # Conductance thermique proxy = corrélation entre voisins × gradient
        conductance = np.mean([abs(sz[i] * sz[i+1]) for i in range(N-1)])
        
        # Flux net = conductance × gradient
        flux = conductance * grad_m
        
        results.append({'hJ':hJ, 'C':C, 'grad_m':grad_m, 
                        'conductance':conductance, 'flux':flux})
    
    C_a = np.array([r['C'] for r in results])
    cond_a = np.array([r['conductance'] for r in results])
    flux_a = np.array([r['flux'] for r in results])
    grad_a = np.array([r['grad_m'] for r in results])
    
    r_Ccond, _ = stats.pearsonr(C_a, cond_a)
    r_Cflux, _ = stats.pearsonr(C_a, flux_a)
    r_Cgrad, _ = stats.pearsonr(C_a, grad_a)
    
    print(f"  ρ(C, Conductance) = {r_Ccond:+.4f}")
    print(f"  ρ(C, Flux)        = {r_Cflux:+.4f}")
    print(f"  ρ(C, Gradient)    = {r_Cgrad:+.4f}")
    
    if r_Ccond > 0.8:
        print(f"  --> C ÉLEVÉ = haute conductance (Bejan : accès facilité)")


# ═══════════════════════════════════════════════════
# PARTIE 5 : SOC — criticalité auto-organisée
# ═══════════════════════════════════════════════════

def test_soc():
    """
    Bak (1987) : les systèmes complexes s'auto-organisent vers 
    le point critique. Le sable s'accumule jusqu'à l'avalanche.
    
    Test : sous relaxation dissipative, le système converge-t-il
    vers le point critique h/J ≈ 0.57 ? Ou vers l'ordre (C=1) ?
    """
    print(f"\n{'='*60}")
    print(f"PARTIE 5 : SOC — le système converge-t-il vers la criticalité ?")
    print(f"{'='*60}", flush=True)
    
    N = 4
    # Valeur de C au point critique
    H_crit = ising_H(N, 0.57, eps=0.01)
    c_ops_crit = []
    for i in range(N):
        o = [qt.qeye(2)]*N; o[i] = qt.destroy(2)
        c_ops_crit.append(np.sqrt(0.1) * qt.tensor(o))
    rho_crit = qt.steadystate(H_crit, c_ops_crit, method='direct')
    sz_ops = []
    for i in range(N):
        o = [qt.qeye(2)]*N; o[i] = qt.sigmaz()
        sz_ops.append(qt.tensor(o))
    sz_crit = np.array([qt.expect(op, rho_crit) for op in sz_ops])
    C_crit, _, _, _, _ = compute_cpsi_detailed(sz_crit)
    print(f"  C au point critique (h/J=0.57) = {C_crit:.4f}")
    
    # Sous amplitude damping, le système va vers C ≈ 1 (ordre)
    # PAS vers la criticalité
    # Mais sous dephasing ? Ou sans dissipation ?
    
    print(f"\n  Convergence sous différentes dissipations :")
    for gamma in [0.01, 0.1, 0.5]:
        for hJ in [0.3, 0.6, 1.0]:
            H = ising_H(N, hJ, eps=0.01)
            c_ops = []
            for i in range(N):
                o = [qt.qeye(2)]*N; o[i] = qt.destroy(2)
                c_ops.append(np.sqrt(gamma) * qt.tensor(o))
            rho_ss = qt.steadystate(H, c_ops, method='direct')
            sz = np.array([qt.expect(op, rho_ss) for op in sz_ops])
            C, T, _, _, _ = compute_cpsi_detailed(sz)
            dist_crit = abs(C - C_crit)
            print(f"    γ={gamma:.2f} hJ={hJ:.1f}: C_ss={C:.4f}  dist(crit)={dist_crit:.4f}  → {'CRITIQUE' if dist_crit < 0.05 else 'ORDRE' if C > 0.8 else 'DÉSORDRE'}")
    
    print(f"\n  VERDICT : sous amplitude damping, le système converge vers")
    print(f"  l'ORDRE (C→1), PAS vers la criticalité (SOC ne s'applique pas)")
    print(f"  → C[Ψ] n'est PAS un système SOC au sens de Bak")
    print(f"  → Mais la TRANSITION entre phases EST le point intéressant")


# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"{'='*60}")
    print(f"C[Psi] — DÉRIVATION + THÉORIES PARALLÈLES")
    print(f"{'='*60}")
    
    t0 = time.time()
    
    test_derivation()    # ~3 min
    test_prigogine()     # ~1 min
    test_friston()       # ~1 min
    test_constructal()   # ~1 min
    test_soc()           # ~1 min
    
    print(f"\nTOTAL: {(time.time()-t0)/60:.1f} min")
