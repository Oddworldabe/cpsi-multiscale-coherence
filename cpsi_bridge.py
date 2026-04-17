#!/usr/bin/env python3
"""
cpsi_bridge.py — Pont entre quantique, fréquences et applications

PARTIE 1 : FINITE-SIZE SCALING (ground state fermé, N=4 à N=14)
  → Instantané (diagonalisation), donne β critique propre
  → Renforce P3q massivement

PARTIE 2 : C[Ψ] SUR SIGNAUX ACOUSTIQUES
  → Son pur vs son dissonant vs bruit
  → Accord parfait vs désaccordé
  → Fréquences harmoniques vs inharmoniques
  → Test direct : la cohérence musicale = C[Ψ] élevé ?

PARTIE 3 : RÉSONANCE ET FRÉQUENCES LIMITES
  → À quelle fréquence C[Ψ] s'effondre ?
  → Lien perception humaine : 20 Hz - 20 kHz
  → Les limites physiologiques = limites de réalité perçue ?

python3 -u ~/Desktop/cpsi_bridge.py 2>&1 | tee ~/Desktop/bridge_log.txt
"""
import numpy as np
import time, warnings, csv
warnings.filterwarnings('ignore')
from scipy import stats, signal, fft
from pathlib import Path

OUT = Path.home() / "Desktop" / "cpsi_bridge"
OUT.mkdir(exist_ok=True)


def compute_cpsi(sz):
    N = len(sz)
    if N < 2: return None, None
    micro = np.mean([abs(sz[i]*sz[i+1]) for i in range(N-1)])
    macro = np.mean([abs(sz[i]*sz[j]) for i in range(N) for j in range(i+1,N)])
    meso = max(0, min(1, 1.0 - np.std(np.abs(sz))))
    C = (micro + macro + meso) / 3
    T = abs(micro-meso) + abs(meso-macro) + abs(micro-macro)
    return C, T


def compute_cpsi_matrix(data_matrix):
    """C[Ψ] depuis matrice T×N (signaux multi-channels)."""
    if data_matrix.shape[1] < 3: 
        # 2 channels: use simple correlation
        data = (data_matrix - data_matrix.mean(axis=0)) / (data_matrix.std(axis=0) + 1e-10)
        r = abs(np.corrcoef(data.T)[0,1])
        return (2*r + 1)/3, 2*(1-r)/3
    data = (data_matrix - data_matrix.mean(axis=0)) / (data_matrix.std(axis=0) + 1e-10)
    C_mat = np.corrcoef(data.T)
    N = C_mat.shape[0]
    micro = np.mean([abs(C_mat[i,i+1]) for i in range(N-1)])
    pairs = [abs(C_mat[i,j]) for i in range(N) for j in range(i+1,N)]
    macro = np.mean(pairs)
    neighbor = [abs(C_mat[i,i+1]) for i in range(N-1)]
    meso = max(0, min(1, 1.0 - np.std(neighbor)))
    C = (micro + macro + meso) / 3
    T = abs(micro-meso) + abs(meso-macro) + abs(micro-macro)
    return C, T


# ═══════════════════════════════════════════════════
# PARTIE 1 : FINITE-SIZE SCALING (N=4 à N=14)
# ═══════════════════════════════════════════════════

def test_finite_size():
    """
    Ground state fermé (avec ε=0.01) pour N=4 à N=14.
    Diagonalisation exacte → instantané jusqu'à N~12.
    Donne le vrai β critique par finite-size scaling.
    """
    import qutip as qt
    
    print(f"\n{'='*60}")
    print(f"PARTIE 1 : FINITE-SIZE SCALING (ground state)")
    print(f"N = 4 à 14, diagonalisation exacte")
    print(f"{'='*60}", flush=True)
    
    epsilon = 0.01
    hJ_values = [0.01, 0.2, 0.4, 0.5, 0.57, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]
    
    all_results = []
    
    for N in range(4, 15):
        dim = 2**N
        if dim > 32768:  # 2^15 = 32768, skip N>14
            print(f"\n  N={N}: dim={dim} trop grand, skip", flush=True)
            break
        
        print(f"\n  N={N} (dim={dim}):", flush=True)
        t_total = time.time()
        
        for hJ in hJ_values:
            t0 = time.time()
            try:
                J = 1.0; h = hJ * J
                H = 0
                for i in range(N-1):
                    o = [qt.qeye(2)]*N; o[i] = qt.sigmaz(); o[i+1] = qt.sigmaz()
                    H += -J * qt.tensor(o)
                for i in range(N):
                    o = [qt.qeye(2)]*N; o[i] = qt.sigmax()
                    H += -h * qt.tensor(o)
                    o2 = [qt.qeye(2)]*N; o2[i] = qt.sigmaz()
                    H += -epsilon * qt.tensor(o2)
                
                evals, estates = H.eigenstates(eigvals=2, sparse=True)
                psi = estates[0]
                gap = evals[1] - evals[0]
                
                sz = []
                for i in range(N):
                    o = [qt.qeye(2)]*N; o[i] = qt.sigmaz()
                    sz.append(qt.expect(qt.tensor(o), psi))
                
                C, T = compute_cpsi(np.array(sz))
                dt = time.time() - t0
                print(f"    hJ={hJ:.2f}  C={C:.4f}  T={T:.4f}  gap={gap:.4f}  ({dt:.1f}s)", flush=True)
                all_results.append({'N':N, 'hJ':hJ, 'C':C, 'T':T, 'gap':gap})
            except Exception as e:
                print(f"    hJ={hJ:.2f}  X {str(e)[:50]}", flush=True)
        
        dt_total = time.time() - t_total
        print(f"  N={N} terminé en {dt_total:.0f}s", flush=True)
    
    if len(all_results) < 20: return
    
    # Analyse finite-size scaling
    print(f"\n  FINITE-SIZE SCALING :")
    print(f"  {'N':<5} {'C_eq(A)':<12} {'C_eq(B)':<12} {'Ratio':<8}")
    for N_val in sorted(set(r['N'] for r in all_results)):
        sub = [r for r in all_results if r['N'] == N_val]
        C_A = np.mean([r['C'] for r in sub if r['hJ'] < 0.3])
        C_B = np.mean([r['C'] for r in sub if r['hJ'] > 1.0])
        ratio = C_A / C_B if C_B > 0.01 else np.inf
        print(f"  {N_val:<5} {C_A:<12.4f} {C_B:<12.4f} {ratio:<8.2f}")
    
    # β par fit power-law pour chaque N
    print(f"\n  EXPOSANT CRITIQUE β par taille :")
    for N_val in sorted(set(r['N'] for r in all_results)):
        sub = [r for r in all_results if r['N'] == N_val]
        hJ_arr = np.array([r['hJ'] for r in sub])
        C_arr = np.array([r['C'] for r in sub])
        C_B = np.mean(C_arr[hJ_arr > 1.0])
        # Fit sur la partie transition
        mask = (hJ_arr >= 0.4) & (hJ_arr <= 0.8) & (C_arr > C_B * 1.1)
        if mask.sum() >= 3:
            x = np.log(np.abs(hJ_arr[mask] - 0.57) + 0.01)
            y = np.log(C_arr[mask] - C_B + 0.01)
            try:
                slope, intercept, r_val, _, _ = stats.linregress(x, y)
                print(f"  N={N_val:2d}  beta_eff = {slope:.3f}  (R2 = {r_val**2:.3f})")
            except:
                pass
    
    with open(OUT / 'finite_size.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_results[0].keys())
        w.writeheader(); w.writerows(all_results)
    print(f"\n  Sauve: {OUT / 'finite_size.csv'}", flush=True)


# ═══════════════════════════════════════════════════
# PARTIE 2 : C[Ψ] SUR SIGNAUX ACOUSTIQUES
# ═══════════════════════════════════════════════════

def test_acoustic():
    """
    Hypothèse : l'harmonie musicale/acoustique = C[Ψ] élevé
    Dissonance = C[Ψ] faible
    
    On génère des signaux multi-channels avec différentes
    structures harmoniques et on mesure C[Ψ].
    """
    print(f"\n{'='*60}")
    print(f"PARTIE 2 : C[Ψ] SUR SIGNAUX ACOUSTIQUES")
    print(f"Harmonie vs Dissonance vs Bruit")
    print(f"{'='*60}", flush=True)
    
    fs = 44100  # Hz audio standard
    duration = 1.0  # 1 seconde
    T_samples = int(fs * duration)
    t = np.arange(T_samples) / fs
    np.random.seed(42)
    
    results = []
    
    # === A. Accords musicaux ===
    print(f"\n  A. ACCORDS MUSICAUX (8 harmoniques par note) :", flush=True)
    
    def make_note(freq, t, n_harmonics=8):
        """Génère une note avec harmoniques naturelles."""
        sig = np.zeros_like(t)
        for h in range(1, n_harmonics+1):
            sig += (1.0/h) * np.sin(2*np.pi*freq*h*t)
        return sig
    
    accords = {
        # Accord parfait majeur (Do-Mi-Sol) : rapports 4:5:6
        'Majeur (4:5:6)': [261.63, 329.63, 392.00],
        # Accord parfait mineur (Do-Mib-Sol) : rapports 10:12:15
        'Mineur (10:12:15)': [261.63, 311.13, 392.00],
        # Quinte juste (Do-Sol) : rapport 2:3
        'Quinte (2:3)': [261.63, 392.00, 523.25],
        # Octave (Do-Do) : rapport 1:2
        'Octave (1:2)': [261.63, 523.25, 261.63*4],
        # Triton (diabolus in musica) : rapport sqrt(2)
        'Triton (sqrt2)': [261.63, 369.99, 523.25],
        # Cluster dissonant (demi-tons adjacents)
        'Cluster (demi-tons)': [261.63, 277.18, 293.66],
        # Quart de ton (micro-intervalle)
        'Quart de ton': [261.63, 269.29, 277.18],
        # Unisson parfait
        'Unisson': [261.63, 261.63, 261.63],
        # Aléatoire (fréquences non harmoniques)
        'Aleatoire': [261.63, 317.42, 428.91],
        # Bruit blanc
        'Bruit blanc': None,
    }
    
    for name, freqs in accords.items():
        if freqs is None:
            # Bruit blanc
            channels = [np.random.randn(T_samples) for _ in range(8)]
        else:
            # 8 channels = 8 harmoniques du fondamental
            channels = []
            for f in freqs:
                for h_idx in range(1, 4):  # 3 harmoniques par note
                    sig = make_note(f * h_idx, t, n_harmonics=3)
                    sig += np.random.randn(T_samples) * 0.01  # petit bruit
                    channels.append(sig)
            # Pad to 8 channels si nécessaire
            while len(channels) < 8:
                channels.append(channels[-1] * 0.5 + np.random.randn(T_samples) * 0.01)
            channels = channels[:8]
        
        data = np.column_stack(channels)
        C, T = compute_cpsi_matrix(data)
        
        # Dissonance proxy : ratio des fréquences
        if freqs:
            ratios = [freqs[i]/freqs[0] for i in range(1, len(freqs))]
            # Proche d'un rapport simple (1, 1.5, 2, etc.) = consonant
            consonance = np.mean([min(abs(r - round(r*6)/6) for r in [0.5,0.667,0.75,1,1.333,1.5,2]) for r in ratios])
        else:
            consonance = 1.0  # bruit = max dissonance
        
        print(f"    {name:<22s}  C={C:.4f}  T={T:.4f}", flush=True)
        results.append({'type': name, 'C': C, 'T': T, 'consonance': consonance})
    
    # === B. Fréquences pures ===
    print(f"\n  B. FRÉQUENCES PURES (même fréquence, phases décalées) :", flush=True)
    
    freq_results = []
    for freq in [20, 50, 100, 261, 440, 1000, 4000, 10000, 20000]:
        channels = []
        for ch in range(8):
            phase = 2*np.pi * ch / 8
            sig = np.sin(2*np.pi*freq*t + phase) + np.random.randn(T_samples) * 0.05
            channels.append(sig)
        data = np.column_stack(channels)
        C, T = compute_cpsi_matrix(data)
        print(f"    f={freq:6d} Hz  C={C:.4f}  T={T:.4f}", flush=True)
        freq_results.append({'freq': freq, 'C': C, 'T': T})
    
    # Analyse : C dépend-il de la fréquence ?
    freqs_arr = np.array([r['freq'] for r in freq_results])
    C_arr = np.array([r['C'] for r in freq_results])
    r, p = stats.pearsonr(np.log10(freqs_arr), C_arr)
    print(f"\n    rho(log10(freq), C) = {r:+.4f}  p={p:.4f}")
    if abs(r) < 0.2:
        print(f"    --> C[Psi] est INDEPENDANT de la frequence pure")
        print(f"    --> Confirme scale-invariance (P2)")
    
    # === C. Battements (phasage/déphasage) ===
    print(f"\n  C. BATTEMENTS (phasage/dephasage) :", flush=True)
    
    for detune in [0, 0.5, 1, 2, 5, 10, 50]:
        channels = []
        f_base = 440  # La
        for ch in range(8):
            f = f_base + detune * (ch - 4)  # spread autour de 440
            sig = np.sin(2*np.pi*f*t) + np.random.randn(T_samples) * 0.02
            channels.append(sig)
        data = np.column_stack(channels)
        C, T = compute_cpsi_matrix(data)
        print(f"    detune={detune:5.1f} Hz  C={C:.4f}  T={T:.4f}", flush=True)
    
    # === D. Résonance : signal + résonateur ===
    print(f"\n  D. RESONANCE (excitation + cavite) :", flush=True)
    
    for Q_factor in [1, 5, 10, 50, 100, 500]:
        # Excitation = impulsion
        excitation = np.zeros(T_samples)
        excitation[100] = 1.0  # impulsion à t=100
        
        # Résonateur = filtre passe-bande centré sur 440 Hz
        f_res = 440
        bw = f_res / Q_factor
        try:
            b, a = signal.butter(2, [max(1, f_res-bw/2), min(fs/2-1, f_res+bw/2)], btype='band', fs=fs)
            resonance = signal.lfilter(b, a, excitation)
        except:
            resonance = excitation
        
        # 8 channels = résonance + bruit variable
        channels = []
        for ch in range(8):
            noise_level = 0.001 * (ch + 1)
            sig = resonance + np.random.randn(T_samples) * noise_level
            channels.append(sig)
        data = np.column_stack(channels)
        C, T = compute_cpsi_matrix(data)
        print(f"    Q={Q_factor:4d}  C={C:.4f}  T={T:.4f}", flush=True)
    
    with open(OUT / 'acoustic.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['type','C','T','consonance'])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\n  Sauve: {OUT / 'acoustic.csv'}", flush=True)


# ═══════════════════════════════════════════════════
# PARTIE 3 : LIMITES DE FRÉQUENCE = LIMITES DE RÉALITÉ ?
# ═══════════════════════════════════════════════════

def test_frequency_limits():
    """
    Hypothèse : la perception humaine (20 Hz - 20 kHz pour l'audio,
    0.5 - 80 Hz pour l'EEG) définit une fenêtre de "réalité perçue".
    
    C[Ψ] capture-t-il cette fenêtre ?
    → Signaux dans la bande perceptible vs hors bande
    → Structure harmonique vs inharmonique
    """
    print(f"\n{'='*60}")
    print(f"PARTIE 3 : LIMITES FREQUENCE = LIMITES REALITE ?")
    print(f"{'='*60}", flush=True)
    
    fs = 44100
    duration = 0.5
    T_samples = int(fs * duration)
    t = np.arange(T_samples) / fs
    np.random.seed(42)
    
    # A. Signal harmonique à différentes fréquences fondamentales
    print(f"\n  A. HARMONIQUES NATURELLES (f, 2f, 3f, 4f) :", flush=True)
    print(f"  {'Freq (Hz)':<12} {'Bande':<15} {'C':<8} {'T':<8}")
    
    results = []
    for f0 in [1, 5, 20, 50, 100, 200, 440, 1000, 4000, 10000, 20000, 40000]:
        channels = []
        for h in range(1, 9):  # 8 harmoniques
            f = f0 * h
            if f > fs/2: f = fs/2 - 100  # Nyquist
            sig = np.sin(2*np.pi*f*t) / h + np.random.randn(T_samples) * 0.01
            channels.append(sig)
        data = np.column_stack(channels)
        C, T = compute_cpsi_matrix(data)
        
        if f0 < 20: band = "infra-son"
        elif f0 <= 20000: band = "audible"
        else: band = "ultra-son"
        
        print(f"  {f0:<12d} {band:<15s} {C:.4f}   {T:.4f}")
        results.append({'f0':f0, 'band':band, 'C':C, 'T':T})
    
    # B. Rapport signal/bruit et cohérence
    print(f"\n  B. RAPPORT SIGNAL/BRUIT vs C[Psi] (f=440 Hz) :", flush=True)
    
    for snr_db in [-20, -10, -6, -3, 0, 3, 6, 10, 20, 40]:
        snr_linear = 10**(snr_db/20)
        channels = []
        for ch in range(8):
            phase = 2*np.pi * ch / 8
            sig = snr_linear * np.sin(2*np.pi*440*t + phase) + np.random.randn(T_samples)
            channels.append(sig)
        data = np.column_stack(channels)
        C, T = compute_cpsi_matrix(data)
        print(f"    SNR={snr_db:+4d} dB  C={C:.4f}  T={T:.4f}")
    
    # C. Transition ordre/désordre sonore
    print(f"\n  C. TRANSITION ORDRE/DESORDRE (ratio harmonique/bruit) :", flush=True)
    
    for ratio in np.linspace(0, 1, 11):
        channels = []
        for ch in range(8):
            harmonic = np.sin(2*np.pi*440*t + 2*np.pi*ch/8)
            noise = np.random.randn(T_samples)
            sig = ratio * harmonic + (1-ratio) * noise
            channels.append(sig)
        data = np.column_stack(channels)
        C, T = compute_cpsi_matrix(data)
        print(f"    harmonic={ratio:.1f}  C={C:.4f}  T={T:.4f}")
    
    with open(OUT / 'frequency_limits.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"{'='*60}")
    print(f"C[Psi] BRIDGE — Quantique / Frequences / Son")
    print(f"{'='*60}", flush=True)
    
    t0 = time.time()
    
    # Partie 2 et 3 d'abord (instantané, pas de qutip)
    test_acoustic()
    test_frequency_limits()
    
    # Partie 1 en dernier (nécessite qutip, peut être long pour N>12)
    test_finite_size()
    
    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"TOTAL: {total/60:.1f} min")
    print(f"Resultats: {OUT}")
    print(f"{'='*60}")
