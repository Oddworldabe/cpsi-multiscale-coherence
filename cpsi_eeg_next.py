#!/usr/bin/env python3
"""
cpsi_eeg_next.py — Tests suivants pour renforcer P_EEG_Frequency

TEST 1 : NULL MODEL — scramble des channels (le test que ChatGPT va demander)
TEST 2 : Full Sleep-EDF Expanded (jusqu'à 20 sujets)
TEST 3 : Lag-embedding pour créer N>2 channels virtuels (fixe le problème ρ(C,T)=-1)

python3 -u ~/Desktop/cpsi_eeg_next.py 2>&1 | tee ~/Desktop/eeg_next_log.txt
"""
import numpy as np
import pandas as pd
import time, warnings, sys
warnings.filterwarnings('ignore')
from pathlib import Path
from scipy import stats, signal, fft

OUT = Path.home() / "Desktop" / "cpsi_eeg_next"
OUT.mkdir(exist_ok=True)


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
    C = (micro+macro+meso)/3
    T = abs(micro-meso) + abs(meso-macro) + abs(micro-macro)
    return {'C':C, 'T':T, 'micro':micro, 'macro':macro, 'meso':meso}


def spectrum_info(sig_1d, fs):
    bands = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,50)}
    N = len(sig_1d)
    if N < 50: return None
    sig_1d = sig_1d - np.mean(sig_1d)
    freqs, psd = signal.welch(sig_1d, fs=fs, nperseg=min(N, int(fs*4)))
    total = np.trapz(psd, freqs) + 1e-10
    bp = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        bp[name] = np.trapz(psd[mask], freqs[mask]) / total if mask.sum() > 0 else 0
    return bp


def lag_embed(signal_2ch, lags=[1, 2, 5, 10]):
    """Crée N > 2 channels virtuels par lag-embedding."""
    T, N = signal_2ch.shape
    embedded = [signal_2ch]
    for lag in lags:
        if lag < T:
            shifted = np.roll(signal_2ch, lag, axis=0)
            shifted[:lag, :] = signal_2ch[:lag, :]
            embedded.append(shifted)
    return np.hstack(embedded)  # T × (N * (1+len(lags)))


# ═══════════════════════════════════════════════════
# TEST 1 : NULL MODEL — scramble temporel
# ═══════════════════════════════════════════════════

def test_null_model():
    """
    Scramble : on mélange l'ordre des échantillons dans chaque channel
    indépendamment. Ça détruit les corrélations inter-channels
    tout en préservant le contenu spectral de chaque channel.
    
    Si C[Ψ] scramblé corrèle ENCORE avec les bandes → artefact.
    Si C[Ψ] scramblé ne corrèle PLUS → le signal est réel.
    """
    print(f"\n{'='*60}")
    print(f"TEST 1 : NULL MODEL (scramble temporel)")
    print(f"{'='*60}", flush=True)
    
    try:
        import mne
        from mne.datasets.sleep_physionet.age import fetch_data
    except ImportError:
        print("  mne requis"); return
    
    files = fetch_data(subjects=[0], recording=[1], on_missing='warn')
    raw = mne.io.read_raw_edf(files[0][0], preload=True, verbose=False)
    data = raw.get_data().T
    fs = raw.info['sfreq']
    ch_names = raw.ch_names
    eeg_idx = [i for i, ch in enumerate(ch_names) if 'EEG' in ch.upper()]
    data = data[:, eeg_idx[:2]]
    
    epoch_len = int(30 * fs)
    n_epochs = min(500, data.shape[0] // epoch_len)
    
    # 1. Résultats réels
    real_results = []
    for ep in range(n_epochs):
        epoch = data[ep*epoch_len:(ep+1)*epoch_len, :]
        cpsi = compute_cpsi(epoch)
        spec = spectrum_info(epoch[:, 0], fs)
        if cpsi and spec:
            real_results.append({**cpsi, **spec})
    df_real = pd.DataFrame(real_results)
    
    # 2. Résultats scramblés (100 permutations)
    n_perm = 100
    scramble_corrs = {'delta':[], 'beta':[], 'gamma':[]}
    
    print(f"  {n_perm} permutations...", flush=True)
    np.random.seed(42)
    for perm in range(n_perm):
        perm_results = []
        for ep in range(n_epochs):
            epoch = data[ep*epoch_len:(ep+1)*epoch_len, :].copy()
            # Scramble chaque channel indépendamment
            for ch in range(epoch.shape[1]):
                np.random.shuffle(epoch[:, ch])
            cpsi = compute_cpsi(epoch)
            spec = spectrum_info(data[ep*epoch_len:(ep+1)*epoch_len, 0], fs)  # spectre ORIGINAL
            if cpsi and spec:
                perm_results.append({**cpsi, **spec})
        
        df_perm = pd.DataFrame(perm_results)
        for band in ['delta', 'beta', 'gamma']:
            r, _ = stats.pearsonr(df_perm[band], df_perm['C'])
            scramble_corrs[band].append(r)
        
        if (perm+1) % 20 == 0:
            print(f"    {perm+1}/{n_perm} permutations done", flush=True)
    
    # Comparaison
    print(f"\n  RÉSULTATS :")
    print(f"  {'Bande':<8} {'ρ réel':<10} {'ρ scramblé (moy)':<20} {'p (rang)':<10}")
    for band in ['delta', 'beta', 'gamma']:
        r_real = stats.pearsonr(df_real[band], df_real['C'])[0]
        r_scr = np.mean(scramble_corrs[band])
        # Rank test: combien de scrambles dépassent le réel ?
        rank = np.mean([abs(s) >= abs(r_real) for s in scramble_corrs[band]])
        print(f"  {band:<8} {r_real:+.4f}     {r_scr:+.4f} ± {np.std(scramble_corrs[band]):.4f}        {rank:.4f}")
    
    print(f"\n  VERDICT :")
    for band in ['delta', 'beta', 'gamma']:
        r_real = abs(stats.pearsonr(df_real[band], df_real['C'])[0])
        r_scr_max = max(abs(s) for s in scramble_corrs[band])
        if r_real > r_scr_max:
            print(f"    {band}: ✅ RÉEL — ρ réel ({r_real:.3f}) dépasse TOUTES les permutations")
        else:
            rank = np.mean([abs(s) >= r_real for s in scramble_corrs[band]])
            print(f"    {band}: {'✅ RÉEL' if rank < 0.05 else '❌ ARTEFACT'} — rang = {rank:.3f}")


# ═══════════════════════════════════════════════════
# TEST 2 : MULTI-SUJET ÉTENDU (jusqu'à 20)
# ═══════════════════════════════════════════════════

def test_extended_subjects():
    print(f"\n{'='*60}")
    print(f"TEST 2 : REPLICATION ÉTENDUE (20 sujets)")
    print(f"{'='*60}", flush=True)
    
    try:
        import mne
        from mne.datasets.sleep_physionet.age import fetch_data
    except ImportError:
        print("  mne requis"); return
    
    all_results = []
    subjects_done = 0
    
    for subj in range(20):
        try:
            files = fetch_data(subjects=[subj], recording=[1], on_missing='warn')
            if not files: continue
            raw = mne.io.read_raw_edf(files[0][0], preload=True, verbose=False)
            data = raw.get_data().T
            fs = raw.info['sfreq']
            ch_names = raw.ch_names
            eeg_idx = [i for i, ch in enumerate(ch_names) if 'EEG' in ch.upper()]
            if len(eeg_idx) < 2: continue
            data = data[:, eeg_idx[:2]]
            
            epoch_len = int(30 * fs)
            n_epochs = min(200, data.shape[0] // epoch_len)  # 200 par sujet pour vitesse
            
            subj_corrs = []
            for ep in range(n_epochs):
                epoch = data[ep*epoch_len:(ep+1)*epoch_len, :]
                cpsi = compute_cpsi(epoch)
                spec = spectrum_info(epoch[:, 0], fs)
                if cpsi and spec:
                    all_results.append({'subject': subj, **cpsi, **spec})
                    subj_corrs.append((cpsi['C'], spec.get('gamma', 0)))
            
            if len(subj_corrs) > 10:
                Cs, Gs = zip(*subj_corrs)
                r = stats.pearsonr(Cs, Gs)[0]
                subjects_done += 1
                print(f"  Sujet {subj:2d}: n={len(subj_corrs):3d}  ρ(γ,C)={r:+.3f}  {'✅' if r < -0.1 else '⚠️'}", flush=True)
        except Exception as e:
            print(f"  Sujet {subj:2d}: ❌ {str(e)[:60]}", flush=True)
    
    if len(all_results) < 100:
        print("  Pas assez de données"); return
    
    df = pd.DataFrame(all_results)
    print(f"\n  Total: {len(df)} epochs, {subjects_done} sujets", flush=True)
    
    # Global
    for band in ['delta','theta','alpha','beta','gamma']:
        r, p = stats.pearsonr(df[band], df['C'])
        print(f"    {band:6s} : ρ = {r:+.4f}  p = {p:.2e}")
    
    # Combien de sujets ont ρ(γ,C) < 0 ?
    neg_count = 0
    for s in df['subject'].unique():
        sub = df[df['subject']==s]
        if len(sub) < 10: continue
        r = stats.pearsonr(sub['gamma'], sub['C'])[0]
        if r < 0: neg_count += 1
    total_s = len(df['subject'].unique())
    print(f"\n  Sujets avec ρ(γ,C) < 0 : {neg_count}/{total_s} ({neg_count/total_s*100:.0f}%)")
    print(f"  Binomial test: p = {stats.binom_test(neg_count, total_s, 0.5):.4f}")
    
    df.to_csv(OUT / 'extended_subjects.csv', index=False)


# ═══════════════════════════════════════════════════
# TEST 3 : LAG-EMBEDDING (N > 2 channels)
# ═══════════════════════════════════════════════════

def test_lag_embedding():
    """
    Avec 2 channels, ρ(C,T) = -1 trivialement.
    Lag-embedding crée N = 2*(1+4) = 10 channels virtuels.
    Sur 10 channels, ρ(C,T) devrait être < 1 en valeur absolue.
    """
    print(f"\n{'='*60}")
    print(f"TEST 3 : LAG-EMBEDDING (10 channels virtuels)")
    print(f"Fixe le problème ρ(C,T) = -1 avec 2 channels")
    print(f"{'='*60}", flush=True)
    
    try:
        import mne
        from mne.datasets.sleep_physionet.age import fetch_data
    except ImportError:
        print("  mne requis"); return
    
    files = fetch_data(subjects=[0], recording=[1], on_missing='warn')
    raw = mne.io.read_raw_edf(files[0][0], preload=True, verbose=False)
    data = raw.get_data().T
    fs = raw.info['sfreq']
    ch_names = raw.ch_names
    eeg_idx = [i for i, ch in enumerate(ch_names) if 'EEG' in ch.upper()]
    data = data[:, eeg_idx[:2]]
    
    epoch_len = int(30 * fs)
    n_epochs = min(500, data.shape[0] // epoch_len)
    
    lags = [1, 3, 10, 30]  # En échantillons (à 100 Hz : 10ms, 30ms, 100ms, 300ms)
    
    results_2ch = []
    results_embed = []
    
    for ep in range(n_epochs):
        epoch = data[ep*epoch_len:(ep+1)*epoch_len, :]
        
        # 2 channels (classique)
        cpsi_2 = compute_cpsi(epoch)
        spec = spectrum_info(epoch[:, 0], fs)
        if cpsi_2 and spec:
            results_2ch.append({**cpsi_2, **spec})
        
        # Lag-embedded (10 channels)
        epoch_embed = lag_embed(epoch, lags=lags)
        cpsi_e = compute_cpsi(epoch_embed)
        if cpsi_e and spec:
            results_embed.append({**cpsi_e, **spec})
    
    df_2 = pd.DataFrame(results_2ch)
    df_e = pd.DataFrame(results_embed)
    
    print(f"\n  COMPARAISON 2 channels vs 10 channels (lag-embedded) :")
    print(f"  {'Métrique':<25} {'2 ch':<15} {'10 ch (embed)':<15}")
    print(f"  {'-'*55}")
    
    r_ct_2 = stats.pearsonr(df_2['C'], df_2['T'])[0]
    r_ct_e = stats.pearsonr(df_e['C'], df_e['T'])[0]
    print(f"  {'ρ(C, T)':<25} {r_ct_2:+.4f}         {r_ct_e:+.4f}")
    
    for band in ['delta','beta','gamma']:
        r_2 = stats.pearsonr(df_2[band], df_2['C'])[0]
        r_e = stats.pearsonr(df_e[band], df_e['C'])[0]
        print(f"  {'ρ('+band+', C)':<25} {r_2:+.4f}         {r_e:+.4f}")
    
    print(f"\n  C moyen 2ch : {df_2['C'].mean():.4f} ± {df_2['C'].std():.4f}")
    print(f"  C moyen 10ch: {df_e['C'].mean():.4f} ± {df_e['C'].std():.4f}")
    
    if abs(r_ct_e) < 0.99:
        print(f"\n  ✅ ρ(C,T) = {r_ct_e:+.4f} (plus trivialement -1 avec lag-embedding)")
    else:
        print(f"\n  ⚠️ ρ(C,T) encore très élevé — lag-embedding insuffisant")
    
    df_e.to_csv(OUT / 'lag_embedded.csv', index=False)


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"{'='*60}")
    print(f"C[Ψ] EEG — TESTS SUIVANTS")
    print(f"{'='*60}", flush=True)
    
    t0 = time.time()
    
    test_null_model()       # ~5 min (100 permutations)
    test_lag_embedding()    # ~2 min
    test_extended_subjects() # ~20 min (téléchargement)
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {(time.time()-t0)/60:.1f} min")
    print(f"Résultats: {OUT}")
    print(f"{'='*60}")
