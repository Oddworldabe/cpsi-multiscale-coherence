# C[Ψ] — Multiscale Coherence Framework

**Empirical Scale Invariance of Multiscale Coherence Across 31 Orders of Magnitude**

Alexandre Fogel-Reinert — Independent Researcher, Thionville, France

---

## Paper

Fogel-Reinert (2026). *Empirical Scale Invariance of Multiscale Coherence Across 31 Orders of Magnitude.* Submitted to Physical Review X.

Preprint: [Zenodo DOI — à ajouter après upload]

## What is C[Ψ]?

```
C[Ψ] = (r_micro + r_macro + r_meso) / 3
```

A multiscale coherence measure combining:
- **r_micro**: local pairwise correlation = (2/N(N−1)) Σ |ρ_ij|
- **r_macro**: global synchrony = √(mean(ρ²))
- **r_meso**: hierarchical compactness = 1 − MST_Mantegna / 2

Additionally:
- **T[Ψ]**: inter-scale tension = |r_micro−r_meso| + |r_meso−r_macro| + |r_micro−r_macro|
- **S[Ψ]**: fragility = std of C over subsystems

## Key Result

Across 18 real-data domains spanning **31 orders of magnitude** in characteristic timescale:

| Statistic | Value | p-value |
|---|---|---|
| Pearson ρ | −0.002 | 0.994 |
| Spearman ρ | −0.018 | 0.945 |
| Kendall τ | −0.014 | 0.939 |
| Bootstrap 95% CI | [−0.46, +0.45] | — |

C[Ψ] discriminates structured from unstructured states equally well from millisecond (cardiac) to billion-year (galactic) timescales.

## Quick Start

```bash
pip install numpy scipy
python scripts/cpsi_core.py --demo
```

## Repository Structure

```
├── README.md
├── LICENSE                    # MIT
├── paper/
│   └── P2_Scale_Invariance.pdf
├── scripts/
│   ├── cpsi_core.py           # C[Ψ] engine (numpy only)
│   └── reproduce_table1.py    # Reproduce all statistical tests
└── data/
    └── README_data.md         # Data sources and access
```

## Citation

```bibtex
@article{fogelreinert2026scale,
  title={Empirical Scale Invariance of Multiscale Coherence Across 31 Orders of Magnitude},
  author={Fogel-Reinert, Alexandre},
  journal={Physical Review X},
  year={2026},
  note={Submitted}
}
```

## Related Work

- [P1] Universal Coherence Functional. *Phys. Rev. E* (2026). In review. DOI:10.5281/zenodo.19444148
- [P3] Continuous Classical-Quantum Order Parameter. In preparation.
- [P7] Coherence Predicts Quantum Cycle Efficiency. In preparation.

## License

MIT
