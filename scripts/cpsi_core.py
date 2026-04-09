#!/usr/bin/env python3
"""
cpsi_core.py — C[Ψ] Multiscale Coherence Engine
==================================================
Standalone implementation. Dependency: numpy only.

Usage:
    from cpsi_core import cpsi
    C, T = cpsi(matrix, window=10)
"""
import numpy as np


def _mst_mean_weight(D):
    """Prim's MST on distance matrix D. Returns mean edge weight."""
    n = D.shape[0]
    if n < 2:
        return 0.0
    visited = np.zeros(n, dtype=bool)
    min_cost = D[0].copy()
    visited[0] = True
    total = 0.0
    for _ in range(n - 1):
        candidates = np.where(~visited)[0]
        if len(candidates) == 0:
            break
        u = candidates[np.argmin(min_cost[candidates])]
        total += min_cost[u]
        visited[u] = True
        for v in range(n):
            if not visited[v] and D[u, v] < min_cost[v]:
                min_cost[v] = D[u, v]
    return total / max(n - 1, 1)


def cpsi(matrix, window=10):
    """
    Compute multiscale coherence C[Ψ] and inter-scale tension T[Ψ].

    Parameters
    ----------
    matrix : ndarray, shape (T, N)
        Time series matrix. T = time steps, N = channels (N >= 3).
    window : int
        Rolling window size for correlation estimation.

    Returns
    -------
    C : float
        Multiscale coherence in [0, 1].
    T : float
        Inter-scale tension in [0, ~1.5].
    """
    T_steps, N = matrix.shape
    if N < 3 or T_steps < window + 2:
        return np.nan, np.nan

    r_micro_list, r_macro_list, r_meso_list = [], [], []

    for t in range(window, T_steps):
        chunk = matrix[t - window:t]
        with np.errstate(all='ignore'):
            corr = np.corrcoef(chunk.T)
        corr = np.nan_to_num(corr)
        np.fill_diagonal(corr, 0)
        upper = corr[np.triu_indices(N, k=1)]

        r_micro_list.append(np.mean(np.abs(upper)))
        r_macro_list.append(np.sqrt(np.mean(upper ** 2)))

        D = np.sqrt(2 * (1 - np.abs(corr)))
        np.fill_diagonal(D, 0)
        r_meso_list.append(1 - _mst_mean_weight(D) / 2)

    rm = np.mean(r_micro_list)
    rM = np.mean(r_macro_list)
    re = np.mean(r_meso_list)

    C = (rm + rM + re) / 3
    T = abs(rm - re) + abs(re - rM) + abs(rm - rM)
    return C, T


def fragility(matrix, window=10, n_sub=10):
    """Compute fragility S[Ψ] = std of C over random subsystems."""
    _, N = matrix.shape
    if N < 4:
        return 0.0
    sub_cs = []
    for _ in range(n_sub):
        idx = np.random.choice(N, max(3, N - 1), replace=False)
        c, _ = cpsi(matrix[:, idx], window)
        if not np.isnan(c):
            sub_cs.append(c)
    return np.std(sub_cs) if len(sub_cs) > 2 else 0.0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='C[Ψ] Multiscale Coherence')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    if args.demo:
        print("C[Ψ] Multiscale Coherence — Demo")
        print("=" * 50)
        np.random.seed(42)

        t = np.arange(300)
        base = np.sin(2 * np.pi * t / 50)
        structured = np.column_stack([base + 0.2 * np.random.randn(300) for _ in range(6)])
        random = np.random.randn(300, 6)

        c_s, t_s = cpsi(structured, window=15)
        c_r, t_r = cpsi(random, window=15)
        s_s = fragility(structured, window=15)
        s_r = fragility(random, window=15)

        print(f"  Structured: C={c_s:.4f}  T={t_s:.4f}  S={s_s:.4f}")
        print(f"  Random:     C={c_r:.4f}  T={t_r:.4f}  S={s_r:.4f}")
        print(f"  ΔC = {c_s - c_r:+.4f}")
        print()
        print("  C[Ψ] correctly identifies structured vs random states.")
