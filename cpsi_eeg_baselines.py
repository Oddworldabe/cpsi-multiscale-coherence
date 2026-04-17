#!/usr/bin/env python3
"""
cpsi_eeg_baselines.py — Baselines entropiques pour P_EEG (Entropy submission)

Compare C[Ψ] à 6 mesures établies sur les MÊMES données EEG :
1. Sample Entropy (SampEn)
2. Permutation Entropy (PermEn)  
3. Spectral Entropy
4. Lempel-Ziv Complexity
5. DFA exponent (α)
6. Mean absolute coherence (baseline simple)

Sur 4 sujets × 500 epochs, même données que dans le papier.

python3 ~/Desktop/cpsi_eeg_baselines.py
"""
import numpy as np
import pandas as pd
from scipy import stats, signal
import warnings, time
warnings.filterwarnings('ignore')

def compute_cpsi(data_matrix):
    data = (data_matrix - data_matrix.mean(axis=0)) / (data_matrix.std(axis=0) + 1e-10)
    C_mat = np.corrcoef(data.T)
    N = C_mat.shape[0]
    micro = np.mean([abs(C_mat[i,i+1]) for i in range(N-1)])
    pairs = [abs(C_mat[i,j]) for i in range(N) for j in range(i+1,N)]
    macro = np.mean(pairs)
    neighbor = [abs(C_mat[i,i+1]) for i in range(N-1)]
    meso = max(0, min(1, 1.0 - np.std(neighbor)))
    C = (micro+macro+meso)/3
    return C

def sample_entropy(sig, m=2, r_mult=0.2):
    """Sample entropy (Richman & Moorman 2000)."""
    N = len(sig)
    r = r_mult * np.std(sig)
    if r < 1e-10: return 0
    def count_matches(template_len):
        count = 0
        templates = np.array([sig[i:i+template_len] for i in range(N-template_len)])
        for i in range(len(templates)):
            dists = np.max(np.abs(templates[i] - templates), axis=1)
            count += np.sum(dists < r) - 1  # exclude self
        return count
    A = count_matches(m+1)
    B = count_matches(m)
    if B == 0 or A == 0: return 0
    return -np.log(A / B)

def permutation_entropy(sig, order=3, delay=1):
    """Permutation entropy (Bandt & Pompe 2002)."""
    N = len(sig)
    n_perms = 0
    perm_counts = {}
    for i in range(N - (order-1)*delay):
        pattern = tuple(np.argsort(sig[i:i+order*delay:delay]))
        perm_counts[pattern] = perm_counts.get(pattern, 0) + 1
        n_perms += 1
    if n_perms == 0: return 0
    probs = np.array(list(perm_counts.values())) / n_perms
    return -np.sum(probs * np.log2(probs + 1e-10)) / np.log2(np.math.factorial(order))

def spectral_entropy(sig, fs):
    """Spectral entropy (Inouye et al. 1991)."""
    freqs, psd = signal.welch(sig, fs=fs, nperseg=min(len(sig), int(fs*4)))
    psd_norm = psd / (psd.sum() + 1e-10)
    psd_norm = psd_norm[psd_norm > 0]
    return -np.sum(psd_norm * np.log2(psd_norm)) / np.log2(len(psd_norm))

def lempel_ziv(sig):
    """Lempel-Ziv complexity (binary sequence)."""
    median = np.median(sig)
    binary = ''.join(['1' if x > median else '0' for x in sig])
    n = len(binary)
    i, c, l = 0, 1, 1
    while i + l <= n:
        sub = binary[i:i+l]
        if sub in binary[:i]:
            l += 1
        else:
            c += 1
            i += l
            l = 1
    return c / (n / np.log2(n + 1e-10) + 1e-10)

def dfa_exponent(sig, min_box=4, max_box=None):
    """Detrended Fluctuation Analysis exponent."""
    N = len(sig)
    if max_box is None: max_box = N // 4
    y = np.cumsum(sig - np.mean(sig))
    box_sizes = np.unique(np.logspace(np.log10(min_box), np.log10(max_box), 15).astype(int))
    flucts = []
    for bs in box_sizes:
        n_boxes = N // bs
        if n_boxes < 2: continue
        f = 0
        for i in range(n_boxes):
            segment = y[i*bs:(i+1)*bs]
            x = np.arange(bs)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            f += np.mean((segment - trend)**2)
        flucts.append(np.sqrt(f / n_boxes))
    if len(flucts) < 3: return 0.5
    log_bs = np.log(box_sizes[:len(flucts)])
    log_fl = np.log(np.array(flucts) + 1e-10)
    slope, _, _, _, _ = stats.linregress(log_bs, log_fl)
    return slope

def band_power(sig, fs, band):
    freqs, psd = signal.welch(sig, fs=fs, nperseg=min(len(sig), int(fs*4)))
    total = np.trapz(psd, freqs) + 1e-10
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.trapz(psd[mask], freqs[mask]) / total if mask.sum() > 0 else 0


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

print("=" * 60)
print("C[Ψ] vs BASELINES ENTROPIQUES — même données EEG")
print("=" * 60)

try:
    import mne
    from mne.datasets.sleep_physionet.age import fetch_data
except ImportError:
    print("pip3 install mne"); exit()

bands = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,50)}
all_results = []

for subj in range(4):
    print(f"\nSujet {subj}:", flush=True)
    files = fetch_data(subjects=[subj], recording=[1], on_missing='warn')
    raw = mne.io.read_raw_edf(files[0][0], preload=True, verbose=False)
    data = raw.get_data().T
    fs = raw.info['sfreq']
    ch_names = raw.ch_names
    eeg_idx = [i for i, ch in enumerate(ch_names) if 'EEG' in ch.upper()]
    data = data[:, eeg_idx[:2]]
    
    epoch_len = int(30 * fs)
    n_epochs = min(500, data.shape[0] // epoch_len)
    
    for ep in range(n_epochs):
        epoch = data[ep*epoch_len:(ep+1)*epoch_len, :]
        sig = epoch[:, 0]  # Channel 1 pour mesures univariées
        
        # C[Ψ] (multivarié)
        cpsi = compute_cpsi(epoch)
        
        # Baselines (univariées sur channel 1)
        # SampEn sur sous-échantillon pour vitesse
        sampen = sample_entropy(sig[::10], m=2, r_mult=0.2)
        permen = permutation_entropy(sig, order=3, delay=1)
        specen = spectral_entropy(sig, fs)
        lz = lempel_ziv(sig)
        dfa = dfa_exponent(sig)
        
        # Coherence simple (baseline multivarié)
        mean_coh = abs(np.corrcoef(epoch.T)[0,1])
        
        # Band powers
        bp = {b: band_power(sig, fs, r) for b, r in bands.items()}
        
        all_results.append({
            'subject': subj, 'epoch': ep,
            'C_psi': cpsi,
            'SampEn': sampen,
            'PermEn': permen,
            'SpecEn': specen,
            'LZ': lz,
            'DFA': dfa,
            'MeanCoh': mean_coh,
            **bp
        })
    
    print(f"  {n_epochs} epochs done", flush=True)

df = pd.DataFrame(all_results)

# ═══════════════════════════════════════════════════
# COMPARAISON : corrélation avec les bandes de fréquence
# ═══════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"CORRÉLATIONS AVEC LES BANDES (n={len(df)})")
print(f"{'='*60}")

methods = ['C_psi', 'SampEn', 'PermEn', 'SpecEn', 'LZ', 'DFA', 'MeanCoh']

print(f"\n{'Method':<12}", end="")
for b in ['delta','alpha','beta','gamma']:
    print(f"  {'ρ('+b+')':<12}", end="")
print(f"  {'|ρ| moyen':<12}")

print("-" * 72)

for method in methods:
    print(f"{method:<12}", end="")
    rhos = []
    for b in ['delta','alpha','beta','gamma']:
        r, p = stats.pearsonr(df[method], df[b])
        print(f"  {r:+.4f}      ", end="")
        rhos.append(abs(r))
    print(f"  {np.mean(rhos):.4f}")

# ═══════════════════════════════════════════════════
# COMPARAISON PAR SUJET
# ═══════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"ρ(method, gamma) PAR SUJET")
print(f"{'='*60}")

print(f"\n{'Method':<12}", end="")
for s in range(4):
    print(f"  {'Subj '+str(s):<10}", end="")
print(f"  {'Neg/4':<8}")

for method in methods:
    print(f"{method:<12}", end="")
    neg_count = 0
    for s in range(4):
        sub = df[df['subject']==s]
        r, _ = stats.pearsonr(sub[method], sub['gamma'])
        print(f"  {r:+.4f}    ", end="")
        if r < 0: neg_count += 1
    print(f"  {neg_count}/4")

# ═══════════════════════════════════════════════════
# SCORE COMPOSITE : quelle méthode discrimine le mieux les bandes ?
# ═══════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"CLASSEMENT GLOBAL (|ρ| moyen sur delta, alpha, beta, gamma)")
print(f"{'='*60}\n")

ranking = []
for method in methods:
    rhos = []
    for b in ['delta','alpha','beta','gamma']:
        r, _ = stats.pearsonr(df[method], df[b])
        rhos.append(abs(r))
    ranking.append((method, np.mean(rhos)))

ranking.sort(key=lambda x: -x[1])

for i, (method, score) in enumerate(ranking):
    marker = " ← C[Ψ]" if method == 'C_psi' else ""
    print(f"  {i+1}. {method:<12} |ρ| moyen = {score:.4f}{marker}")

# Save
df.to_csv('/Users/oddworldabe/Desktop/cpsi_eeg_baselines.csv' if False else 'eeg_baselines.csv', index=False)
print(f"\nSauvé: eeg_baselines.csv")
