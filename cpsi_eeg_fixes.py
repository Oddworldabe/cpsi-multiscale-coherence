#!/usr/bin/env python3
"""
cpsi_eeg_fixes.py — Calcule les valeurs manquantes pour le papier
1. Alpha per-subject (4 sujets)
2. Phase-randomization null model (meilleur que scramble)

python3 ~/Desktop/cpsi_eeg_fixes.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats, signal, fft
import warnings
warnings.filterwarnings('ignore')

def compute_cpsi(data_matrix):
    if data_matrix.shape[1] < 2: return None
    data = (data_matrix - data_matrix.mean(axis=0)) / (data_matrix.std(axis=0) + 1e-10)
    C_mat = np.corrcoef(data.T)
    N = C_mat.shape[0]
    micro = np.mean([abs(C_mat[i,i+1]) for i in range(N-1)])
    pairs = [abs(C_mat[i,j]) for i in range(N) for j in range(i+1,N)]
    macro = np.mean(pairs) if pairs else 0
    neighbor = [abs(C_mat[i,i+1]) for i in range(N-1)]
    meso = max(0, min(1, 1.0 - np.std(neighbor))) if neighbor else 0
    return (micro+macro+meso)/3

def band_power(sig, fs, band):
    freqs, psd = signal.welch(sig, fs=fs, nperseg=min(len(sig), int(fs*4)))
    total = np.trapz(psd, freqs) + 1e-10
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.trapz(psd[mask], freqs[mask]) / total if mask.sum() > 0 else 0

def phase_randomize(sig):
    """Phase-randomization surrogate: preserves power spectrum, destroys correlations."""
    N = len(sig)
    ft = fft.fft(sig)
    phases = np.random.uniform(0, 2*np.pi, len(ft))
    phases[0] = 0  # keep DC
    if N % 2 == 0: phases[N//2] = 0  # keep Nyquist
    # Impose conjugate symmetry
    for i in range(1, N//2):
        phases[N-i] = -phases[i]
    ft_surr = np.abs(ft) * np.exp(1j * phases)
    return np.real(fft.ifft(ft_surr))

print("=" * 60)
print("FIXES POUR P_EEG")
print("=" * 60)

try:
    import mne
    from mne.datasets.sleep_physionet.age import fetch_data
except ImportError:
    print("pip3 install mne"); exit()

# ═══════════════════════════════════════════════════
# FIX 1 : ALPHA PER-SUBJECT
# ═══════════════════════════════════════════════════

print(f"\nFIX 1 : ALPHA PER-SUBJECT (4 sujets)")
print("-" * 40, flush=True)

bands = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,50)}

for subj in range(4):
    files = fetch_data(subjects=[subj], recording=[1], on_missing='warn')
    raw = mne.io.read_raw_edf(files[0][0], preload=True, verbose=False)
    data = raw.get_data().T
    fs = raw.info['sfreq']
    ch = raw.ch_names
    eeg_idx = [i for i, c in enumerate(ch) if 'EEG' in c.upper()]
    data = data[:, eeg_idx[:2]]
    
    epoch_len = int(30 * fs)
    n_ep = min(500, data.shape[0] // epoch_len)
    
    Cs = []; bps = {b: [] for b in bands}
    for ep in range(n_ep):
        epoch = data[ep*epoch_len:(ep+1)*epoch_len, :]
        c = compute_cpsi(epoch)
        if c is None: continue
        Cs.append(c)
        for b, rng in bands.items():
            bps[b].append(band_power(epoch[:, 0], fs, rng))
    
    Cs = np.array(Cs)
    print(f"\n  Sujet {subj} (n={len(Cs)}):")
    for b in bands:
        r, p = stats.pearsonr(np.array(bps[b]), Cs)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"    ρ({b:6s}, C) = {r:+.4f}  p={p:.4f}  {sig}")

# ═══════════════════════════════════════════════════
# FIX 2 : PHASE-RANDOMIZATION NULL MODEL
# ═══════════════════════════════════════════════════

print(f"\n\nFIX 2 : PHASE-RANDOMIZATION NULL (100 perms, sujet 0)")
print("-" * 40, flush=True)

files = fetch_data(subjects=[0], recording=[1], on_missing='warn')
raw = mne.io.read_raw_edf(files[0][0], preload=True, verbose=False)
data = raw.get_data().T
fs = raw.info['sfreq']
ch = raw.ch_names
eeg_idx = [i for i, c in enumerate(ch) if 'EEG' in c.upper()]
data = data[:, eeg_idx[:2]]

epoch_len = int(30 * fs)
n_ep = min(500, data.shape[0] // epoch_len)

# Real
real_Cs = []; real_bps = {b: [] for b in ['delta','beta','gamma']}
for ep in range(n_ep):
    epoch = data[ep*epoch_len:(ep+1)*epoch_len, :]
    c = compute_cpsi(epoch)
    if c is None: continue
    real_Cs.append(c)
    for b in ['delta','beta','gamma']:
        real_bps[b].append(band_power(epoch[:, 0], fs, bands[b]))

real_corrs = {}
for b in ['delta','beta','gamma']:
    real_corrs[b] = stats.pearsonr(np.array(real_bps[b]), np.array(real_Cs))[0]

# Phase-randomized surrogates
n_perm = 100
surr_corrs = {b: [] for b in ['delta','beta','gamma']}
np.random.seed(42)

for perm in range(n_perm):
    perm_Cs = []
    for ep in range(n_ep):
        epoch = data[ep*epoch_len:(ep+1)*epoch_len, :].copy()
        # Phase-randomize each channel independently
        for ch_i in range(epoch.shape[1]):
            epoch[:, ch_i] = phase_randomize(epoch[:, ch_i])
        c = compute_cpsi(epoch)
        if c is not None:
            perm_Cs.append(c)
    
    for b in ['delta','beta','gamma']:
        # Use REAL band powers (spectrum preserved by phase-rand)
        r = stats.pearsonr(np.array(real_bps[b])[:len(perm_Cs)], np.array(perm_Cs))[0]
        surr_corrs[b].append(r)
    
    if (perm+1) % 25 == 0:
        print(f"  {perm+1}/100 done", flush=True)

print(f"\n  RÉSULTATS PHASE-RANDOMIZATION :")
print(f"  {'Bande':<8} {'ρ réel':<10} {'ρ surr (moy±sd)':<20} {'Rang':<8}")
for b in ['delta','beta','gamma']:
    rank = np.mean([abs(s) >= abs(real_corrs[b]) for s in surr_corrs[b]])
    print(f"  {b:<8} {real_corrs[b]:+.4f}     {np.mean(surr_corrs[b]):+.4f}±{np.std(surr_corrs[b]):.4f}      {rank:.4f}")

print(f"\n{'='*60}")
print("TERMINÉ")
print(f"{'='*60}")
