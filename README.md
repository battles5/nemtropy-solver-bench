# ERGM / Maximum-Entropy: Solver Benchmark & Null-Model Analysis

Reproducible benchmark of numerical solvers for **Exponential Random Graph Models** (ERGM), plus an applied null-model analysis with z-scores.  
Developed for the **Complex Network Analysis** course — [2nd Level Master in Data Science and Statistical Learning (MD2SL)](https://md2sl-eng.imtlucca.it/), IMT School for Advanced Studies Lucca & University of Florence.

---

## Background

Maximum-entropy models (ERGM) provide statistically principled *null models* for real networks.
Given a set of observed constraints (e.g. the degree sequence), the ERGM assigns probabilities to all possible graphs by maximising Shannon entropy subject to those constraints.
The resulting distribution takes the exponential form

$$P^*(\mathbf{G}) = \frac{e^{-H(\mathbf{G},\,\boldsymbol{\theta})}}{Z(\boldsymbol{\theta})}, \qquad H(\mathbf{G},\,\boldsymbol{\theta}) = \sum_i \theta_i\,C_i(\mathbf{G}),$$

where the Lagrange multipliers $\boldsymbol{\theta}$ are estimated by maximising the log-likelihood numerically.

**Undirected Binary Configuration Model (UBCM):**
For an undirected network with degree constraints $\{k_i\}$, each link probability factorises as:

$$p_{ij} = \frac{x_i\,x_j}{1 + x_i\,x_j}, \qquad \langle k_i \rangle = \sum_{j\neq i} p_{ij} = k_i^{\text{obs}}$$

**Directed Binary Configuration Model (DBCM):**
For a directed network with out-degree and in-degree constraints:

$$p_{ij} = \frac{x_i\,y_j}{1 + x_i\,y_j}$$

Different solvers offer different trade-offs between speed, memory and convergence:

| Method | Algorithm | Memory | Convergence |
|---|---|---|---|
| `newton` | Newton–Raphson, full Hessian | $O(n^2)$ | Quadratic |
| `quasinewton` | Newton–Raphson, diagonal Hessian | $O(n)$ | Super-linear |
| `fixed-point` | Fixed-point iteration | $O(n)$ | Linear |

**Reference:** Vallarano *et al.* (2021), *Scientific Reports* **11**, 15227. Implementation: [NEMtropy](https://github.com/nicoloval/NEMtropy).

---

## Part 1 — Solver Benchmark (`src/experiment.py`)

### Networks

| Network | Type | n | m | Model |
|---|---|---|---|---|
| Zachary Karate Club | empirical, undirected | 34 | 78 | UBCM |
| Les Misérables | empirical, undirected | 77 | 254 | UBCM |
| G(n, p = 0.05) undirected | synthetic | 50 – 2 000 | variable | UBCM |
| G(n, p = 0.05) directed | synthetic | 50 – 2 000 | variable | DBCM |

### Key benchmark results (seed = 42, 5 runs)

**UBCM (undirected)**

| Network | Method | n | Runtime (s) | Max rel. error |
|---|---|---|---|---|
| Karate Club | `newton` | 34 | 0.401 ± 0.89 | 3.9 × 10⁻⁸ |
| Karate Club | `quasinewton` | 34 | 0.050 ± 0.09 | 2.2 × 10⁻⁸ |
| Karate Club | `fixed-point` | 34 | 0.082 ± 0.18 | 4.0 × 10⁻⁹ |
| Les Misérables | `newton` | 77 | 0.002 ± 0.000 | 1.3 × 10⁻⁸ |
| Les Misérables | `quasinewton` | 77 | 0.005 ± 0.001 | 6.2 × 10⁻⁸ |
| Les Misérables | `fixed-point` | 77 | 0.001 ± 0.000 | 4.8 × 10⁻⁹ |
| G(n, p) | `newton` | 2 000 | 0.011 ± 0.001 | 4.3 × 10⁻⁸ |
| G(n, p) | `quasinewton` | 2 000 | 0.029 ± 0.006 | 1.0 × 10⁻⁸ |
| G(n, p) | `fixed-point` | 2 000 | 0.012 ± 0.000 | 1.1 × 10⁻⁹ |

**DBCM (directed)**

| Network | Method | n | Runtime (s) | Max rel. error |
|---|---|---|---|---|
| G(n, p) | `newton` | 2 000 | 26.7 ± 5.8 | 3.2 × 10⁻⁷ |
| G(n, p) | `quasinewton` | 2 000 | 1.45 ± 0.17 | 3.5 × 10⁻⁸ |
| G(n, p) | `fixed-point` | 2 000 | 0.83 ± 0.007 | 1.9 × 10⁻¹⁰ |

### Key findings

- All solvers achieve errors of order $10^{-8}$–$10^{-10}$, confirming the results of Vallarano *et al.* (2021).
- **`newton`** scales poorly on DBCM because the full Hessian grows as $O(n^2)$: at n = 2 000 it is **~32× slower** than `fixed-point`.
- **`fixed-point`** is the fastest and most stable solver across all sizes, with the lowest variance.
- The solver choice is a **speed vs. robustness** trade-off, not a speed vs. accuracy one.

---

## Part 2 — Null-Model Analysis with Z-Scores (`src/analysis.py`)

Beyond benchmarking solver speed, the second script demonstrates the *application* of the UBCM as a statistical null model, following the methodology of Squartini & Garlaschelli (2011).

### Methodology

For each real-world network:

1. **Fit UBCM** → obtain parameters $\{x_i\}$.
2. **Reconstruct probability matrix** → $p_{ij} = x_i x_j / (1 + x_i x_j)$.
3. **Sample an ensemble** of 500 random graphs from the UBCM via independent Bernoulli trials.
4. **Compute observed network properties**:
   - **ANND** (Average Nearest-Neighbour Degree): $k_{nn,i} = \frac{1}{k_i}\sum_j a_{ij} k_j$
   - **Local clustering coefficient**: $c_i = \frac{2 t_i}{k_i(k_i - 1)}$ where $t_i$ = triangles at node $i$
   - **Transitivity** (global clustering): $\frac{3 \times \text{triangles}}{\text{connected triples}}$
   - **Degree assortativity**: Pearson correlation of degrees at edge endpoints
5. **Z-scores**: $z_X = \frac{X^{\text{obs}} - \langle X \rangle_{\text{UBCM}}}{\sigma_X^{\text{UBCM}}}$
   - $|z| > 2$: the property is **statistically significant** — not explained by the degree sequence alone.

### Z-score results (ensemble = 500, seed = 42)

**Karate Club (n = 34)**

| Property | Observed | ⟨X⟩_UBCM | σ | z-score | Significant? |
|---|---|---|---|---|---|
| avg_clustering | 0.571 | 0.357 | 0.069 | **+3.08** | **Yes** |
| transitivity | 0.256 | 0.272 | 0.036 | −0.45 | No |
| n_triangles | 45 | 53.0 | 13.5 | −0.59 | No |
| assortativity | −0.476 | −0.327 | 0.067 | **−2.21** | **Yes** |
| density | 0.139 | 0.139 | 0.012 | −0.02 | No |

**Les Misérables (n = 77)**

| Property | Observed | ⟨X⟩_UBCM | σ | z-score | Significant? |
|---|---|---|---|---|---|
| avg_clustering | 0.573 | 0.238 | 0.030 | **+11.0** | **Yes** |
| transitivity | 0.499 | 0.236 | 0.019 | **+13.6** | **Yes** |
| n_triangles | 467 | 237.4 | 38.9 | **+5.9** | **Yes** |
| assortativity | −0.165 | −0.185 | 0.037 | +0.53 | No |
| density | 0.087 | 0.087 | 0.005 | −0.01 | No |

### Interpretation

- **Density is always z ≈ 0** — expected, since the degree constraints implicitly fix the total number of edges.
- **Karate Club** shows significantly higher clustering ($z = +3.1$) and significantly more disassortative mixing ($z = -2.2$) than explained by degrees alone → evidence of genuine community structure.
- **Les Misérables** has *massively* higher clustering and transitivity ($z > 11$) than the UBCM predicts → the co-appearance network has strong triadic closure *beyond* what heterogeneous degrees produce. This aligns with the known presence of dense character groups (communities).
- **Assortativity** in Les Misérables is *not* significant ($z = +0.5$) → degree-degree correlations are well-explained by the degree sequence alone.

---

## Figures

### Benchmark figures (`src/experiment.py`)

| Figure | Description |
|---|---|
| `runtime_vs_n.png` | Log-log runtime scaling (UBCM / DBCM panels) |
| `error_vs_n.png` | Log-log mean relative error vs n |
| `karate_runtime_methods.png` | Bar chart: solver comparison on Karate Club |
| `error_heatmap.png` | Heatmap of max relative error by (n, method) |

### Analysis figures (`src/analysis.py`)

| Figure | Description |
|---|---|
| `*_degree_scatter.png` | Observed vs expected degree (should align on y = x) |
| `*_annd_comparison.png` | ANND: observed vs UBCM ensemble, by node degree |
| `*_clustering_comparison.png` | Clustering: observed vs UBCM ensemble, by node degree |
| `*_zscore_summary.png` | Bar chart of z-scores per property |
| `*_transitivity_distribution.png` | Ensemble histogram with observed value |
| `*_avg_clustering_distribution.png` | Ensemble histogram with observed value |
| `*_n_triangles_distribution.png` | Ensemble histogram with observed value |

---

## Quickstart

```bash
# Create virtualenv
python3.11 -m venv .venv
source .venv/bin/activate            # Linux / macOS
# .\.venv\Scripts\Activate.ps1       # Windows PowerShell

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Run solver benchmark (default: 5 runs, sizes up to 2000)
python src/experiment.py --outdir results --seed 42

# Run null-model analysis (default: 500 ensemble samples)
python src/analysis.py --outdir results --seed 42

# Customise
python src/experiment.py --runs 10 --sizes "100,500,1000,5000" --p 0.1
python src/analysis.py --ensemble 1000
```

### Conda alternative

```bash
conda env create -f environment.yml
conda activate cna-nemtropy
python src/experiment.py
python src/analysis.py
```

---

## Output

```
results/
├── figures/
│   ├── runtime_vs_n.png                         # benchmark
│   ├── error_vs_n.png
│   ├── karate_runtime_methods.png
│   ├── error_heatmap.png
│   ├── karate_club_degree_scatter.png            # analysis
│   ├── karate_club_annd_comparison.png
│   ├── karate_club_clustering_comparison.png
│   ├── karate_club_zscore_summary.png
│   ├── karate_club_transitivity_distribution.png
│   ├── les_miserables_degree_scatter.png
│   ├── les_miserables_annd_comparison.png
│   ├── les_miserables_clustering_comparison.png
│   ├── les_miserables_zscore_summary.png
│   └── ...
├── tables/
│   ├── benchmark.csv
│   ├── summary.csv
│   ├── zscore_analysis.csv
│   ├── metadata.json
│   └── analysis_metadata.json
└── samples/
```

## Reproducibility

- **Global seed:** `--seed 42` (default). NumPy's `default_rng` derives per-run seeds deterministically.
- **`metadata.json`** and **`analysis_metadata.json`** record Python version, platform, library versions and all CLI parameters.
- **Freeze dependencies:** `pip freeze > pip_freeze.txt`

## Repository structure

```
.
├── README.md
├── LICENSE                        # MIT
├── requirements.txt
├── environment.yml
├── .gitignore
├── src/
│   ├── experiment.py              # Solver benchmark
│   └── analysis.py                # Null-model analysis (z-scores)
├── data/                          # Reserved for future datasets
└── results/                       # Generated (gitignored)
    ├── figures/
    ├── tables/
    └── samples/
```

## References

1. Vallarano, N., Bruno, M., Marchese, E., Trapani, G., Saracco, F., Cimini, G., Zanon, M. & Squartini, T. (2021). *Fast and scalable likelihood maximization for Exponential Random Graph Models with local constraints.* Scientific Reports **11**, 15227. [doi:10.1038/s41598-021-93830-4](https://doi.org/10.1038/s41598-021-93830-4) — [arXiv:2101.12625](https://arxiv.org/abs/2101.12625)
2. Squartini, T. & Garlaschelli, D. (2011). *Analytical maximum-likelihood method to detect patterns in real networks.* New Journal of Physics **13**, 083001. [doi:10.1088/1367-2630/13/8/083001](https://doi.org/10.1088/1367-2630/13/8/083001)
3. Zachary, W. W. (1977). *An information flow model for conflict and fission in small groups.* Journal of Anthropological Research **33**(4), 452–473.
4. Knuth, D. E. (1993). *The Stanford GraphBase: A Platform for Combinatorial Computing.* ACM Press. (Les Misérables dataset)
