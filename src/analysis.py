#!/usr/bin/env python
"""
Null-model analysis: z-scores and observed vs expected properties.

Demonstrates the *application* of the Undirected Binary Configuration
Model (UBCM) as a statistical null model, following the methodology
of Squartini & Garlaschelli (2011) and Sections 8A-8B of the course.

For each real-world network this script:
  1. Fits the UBCM via NEMtropy (fixed-point solver).
  2. Reconstructs the link-probability matrix  p_ij = x_i x_j / (1 + x_i x_j).
  3. Computes observed network properties (ANND, clustering, triangles,
     assortativity).
  4. Samples an ensemble of random graphs preserving expected degrees.
  5. Computes z-scores:  z = (X_obs - <X>) / sigma_X.
  6. Generates diagnostic and comparison plots.

Networks
--------
- Zachary Karate Club  (n = 34,  m = 78)
- Les Miserables co-appearance  (n = 77,  m ~254)

Usage
-----
    python src/analysis.py --help
    python src/analysis.py --outdir results --seed 42 --ensemble 500
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
except Exception:
    sns = None  # type: ignore

from NEMtropy import UndirectedGraph

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

UBCM_MODEL = "cm_exp"


# ===================================================================
# Network loaders
# ===================================================================

def load_karate() -> tuple[nx.Graph, str]:
    """Load Zachary's Karate Club (n=34, m=78)."""
    G = nx.karate_club_graph()
    return G, "Karate Club"


def load_les_miserables() -> tuple[nx.Graph, str]:
    """Load Les Miserables co-appearance network (n=77).

    The original graph is weighted; we binarise it and
    relabel nodes to integers for compatibility with adjacency matrices.
    """
    G = nx.les_miserables_graph()
    # Binarise (drop weights)
    G_bin = nx.Graph()
    G_bin.add_edges_from(G.edges())
    G_bin = nx.convert_node_labels_to_integers(G_bin)
    return G_bin, "Les Miserables"


# ===================================================================
# UBCM fitting and probability matrix
# ===================================================================

def fit_ubcm(A: np.ndarray, method: str = "fixed-point") -> np.ndarray:
    """Fit the UBCM and return the x_i parameter vector.

    Parameters
    ----------
    A : np.ndarray
        Binary symmetric adjacency matrix.
    method : str
        Solver name (default: ``"fixed-point"``).

    Returns
    -------
    np.ndarray
        Fitted x_i parameters (one per node).
    """
    g = UndirectedGraph(A)
    g.solve_tool(
        model=UBCM_MODEL,
        method=method,
        initial_guess="random",
        max_steps=500,
        full_return=False,
        verbose=False,
    )
    x = np.asarray(g.x).ravel()
    return x


def pij_matrix(x: np.ndarray) -> np.ndarray:
    """Build the probability matrix from UBCM parameters.

    .. math::
        p_{ij} = \\frac{x_i \\, x_j}{1 + x_i \\, x_j}
    """
    xx = np.outer(x, x)
    P = xx / (1.0 + xx)
    np.fill_diagonal(P, 0.0)
    return P


def sample_ensemble(
    P: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Sample binary undirected graphs from the probability matrix.

    Each graph is drawn independently: for every pair (i, j) with i < j,
    a Bernoulli trial with probability p_ij determines the link.
    """
    n = P.shape[0]
    P_upper = np.triu(P, k=1)          # only upper triangle matters
    graphs: list[np.ndarray] = []
    for _ in range(n_samples):
        R = rng.random((n, n))
        A = (R < P_upper).astype(float)
        A = A + A.T                     # symmetrise
        graphs.append(A)
    return graphs


# ===================================================================
# Network property computation
# ===================================================================

def compute_degrees(A: np.ndarray) -> np.ndarray:
    """Degree sequence."""
    return A.sum(axis=1)


def compute_annd(A: np.ndarray) -> np.ndarray:
    """Average Nearest-Neighbour Degree (ANND) for each node.

    .. math::
        k_{nn,i} = \\frac{1}{k_i} \\sum_j a_{ij} \\, k_j
    """
    k = A.sum(axis=1)
    n = A.shape[0]
    annd = np.zeros(n)
    for i in range(n):
        nbrs = np.where(A[i] > 0)[0]
        if len(nbrs) > 0:
            annd[i] = k[nbrs].mean()
    return annd


def compute_clustering(A: np.ndarray) -> np.ndarray:
    """Local clustering coefficient for each node."""
    G = nx.from_numpy_array(A)
    cc = nx.clustering(G)
    return np.array([cc[i] for i in range(len(cc))])


def count_triangles(A: np.ndarray) -> int:
    r"""Total number of triangles: :math:`\mathrm{tr}(A^3) / 6`."""
    A3 = A @ A @ A
    return int(round(np.trace(A3))) // 6


def compute_global_properties(A: np.ndarray) -> dict[str, float]:
    """Compute a suite of scalar network properties."""
    G = nx.from_numpy_array(A)
    n = A.shape[0]
    m = int(A.sum()) // 2
    props: dict[str, float] = {
        "avg_clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
        "n_triangles": float(count_triangles(A)),
        "density": float(A.sum()) / (n * (n - 1)) if n > 1 else 0.0,
    }
    # Degree assortativity (requires >= 1 edge)
    if m > 0:
        try:
            props["assortativity"] = nx.degree_assortativity_coefficient(G)
        except Exception:
            pass
    return props


# ===================================================================
# Z-score
# ===================================================================

def compute_zscore(
    obs: float,
    ens_vals: np.ndarray,
) -> tuple[float, float, float]:
    """Return (z_score, ensemble_mean, ensemble_std)."""
    mu = float(np.mean(ens_vals))
    sigma = float(np.std(ens_vals, ddof=0))
    if sigma < 1e-15:
        z = 0.0 if abs(obs - mu) < 1e-15 else float(np.sign(obs - mu)) * np.inf
    else:
        z = (obs - mu) / sigma
    return z, mu, sigma


# ===================================================================
# Plots
# ===================================================================

def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", path)


def plot_degree_scatter(
    k_obs: np.ndarray,
    k_exp: np.ndarray,
    name: str,
    figpath: Path,
) -> None:
    """Scatter of observed vs model-expected degrees."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(k_obs, k_exp, alpha=0.7, edgecolors="black", linewidth=0.5, s=60)
    lim = max(k_obs.max(), k_exp.max()) * 1.1
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="y = x")
    ax.set_xlabel(r"Observed degree $k_i$")
    ax.set_ylabel(r"Expected degree $\langle k_i \rangle$")
    ax.set_title(f"Degree: observed vs expected — {name}")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    _savefig(fig, figpath)


def plot_annd_comparison(
    annd_obs: np.ndarray,
    annd_exp_mean: np.ndarray,
    annd_exp_std: np.ndarray,
    k_obs: np.ndarray,
    name: str,
    figpath: Path,
) -> None:
    """ANND observed vs UBCM ensemble, plotted against node degree."""
    fig, ax = plt.subplots(figsize=(7, 5))
    mask = k_obs > 0
    ax.errorbar(
        k_obs[mask], annd_exp_mean[mask], yerr=annd_exp_std[mask],
        fmt="o", color="steelblue", alpha=0.6, label="UBCM ensemble",
        capsize=3,
    )
    ax.scatter(
        k_obs[mask], annd_obs[mask], color="red", marker="x", s=80,
        linewidth=2, label="Observed", zorder=5,
    )
    ax.set_xlabel(r"Node degree $k_i$")
    ax.set_ylabel(r"$k_{nn,i}$  (ANND)")
    ax.set_title(f"ANND: observed vs UBCM — {name}")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, figpath)


def plot_clustering_comparison(
    cc_obs: np.ndarray,
    cc_exp_mean: np.ndarray,
    cc_exp_std: np.ndarray,
    k_obs: np.ndarray,
    name: str,
    figpath: Path,
) -> None:
    """Clustering coefficient: observed vs UBCM ensemble vs degree."""
    fig, ax = plt.subplots(figsize=(7, 5))
    mask = k_obs > 0
    ax.errorbar(
        k_obs[mask], cc_exp_mean[mask], yerr=cc_exp_std[mask],
        fmt="o", color="steelblue", alpha=0.6, label="UBCM ensemble",
        capsize=3,
    )
    ax.scatter(
        k_obs[mask], cc_obs[mask], color="red", marker="x", s=80,
        linewidth=2, label="Observed", zorder=5,
    )
    ax.set_xlabel(r"Node degree $k_i$")
    ax.set_ylabel(r"Clustering coefficient $c_i$")
    ax.set_title(f"Clustering: observed vs UBCM — {name}")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, figpath)


def plot_zscore_summary(
    zscores: dict[str, float],
    name: str,
    figpath: Path,
) -> None:
    """Horizontal bar chart of z-scores for global properties."""
    fig, ax = plt.subplots(figsize=(8, 4))
    props = list(zscores.keys())
    vals = [zscores[p] for p in props]
    colors = ["#d62728" if abs(v) > 2 else "#1f77b4" for v in vals]
    y_pos = range(len(props))
    ax.barh(y_pos, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(x=2,  color="gray", linestyle="--", linewidth=0.8, label=r"$|z| = 2$")
    ax.axvline(x=-2, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(props)
    ax.set_xlabel("z-score")
    ax.set_title(f"Z-scores under UBCM null model — {name}")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _savefig(fig, figpath)


def plot_ensemble_histogram(
    obs_val: float,
    ens_vals: np.ndarray,
    prop_name: str,
    name: str,
    figpath: Path,
) -> None:
    """Histogram of an ensemble-level property with observed value marked."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(
        ens_vals, bins=35, alpha=0.7, color="steelblue",
        edgecolor="black", density=True,
    )
    ax.axvline(
        obs_val, color="red", linewidth=2, linestyle="--",
        label=f"Observed = {obs_val:.4f}",
    )
    mu = float(np.mean(ens_vals))
    ax.axvline(
        mu, color="green", linewidth=1.5, linestyle=":",
        label=f"Ensemble mean = {mu:.4f}",
    )
    ax.set_xlabel(prop_name)
    ax.set_ylabel("Density")
    ax.set_title(f"{prop_name} distribution under UBCM — {name}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(fig, figpath)


# ===================================================================
# Main analysis pipeline — one network
# ===================================================================

def analyse_network(
    G: nx.Graph,
    name: str,
    dirs: dict[str, Path],
    n_ensemble: int,
    rng: np.random.Generator,
) -> dict:
    """Full UBCM null-model analysis on a single network.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    name : str
        Human-readable network name (used in titles).
    dirs : dict
        Output directories (``"fig"``, ``"tab"``).
    n_ensemble : int
        Number of ensemble graphs to sample.
    rng : np.random.Generator
        Seeded RNG.

    Returns
    -------
    dict
        Summary results including z-scores and fit time.
    """
    log.info("=" * 60)
    log.info("Analysing: %s  (n=%d, m=%d)", name, G.number_of_nodes(), G.number_of_edges())
    log.info("=" * 60)

    # — Adjacency matrix (binarised, no self-loops) —
    A = nx.to_numpy_array(G, dtype=float)
    A = (A > 0).astype(float)
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]

    # ------------------------------------------------------------------
    # 1)  Fit UBCM
    # ------------------------------------------------------------------
    log.info("Fitting UBCM (fixed-point)...")
    t0 = time.perf_counter()
    x = fit_ubcm(A, method="fixed-point")
    fit_time = time.perf_counter() - t0
    log.info("  Fit completed in %.3f s", fit_time)

    # ------------------------------------------------------------------
    # 2)  Probability matrix
    # ------------------------------------------------------------------
    P = pij_matrix(x)

    # ------------------------------------------------------------------
    # 3)  Observed properties
    # ------------------------------------------------------------------
    log.info("Computing observed properties...")
    k_obs = compute_degrees(A)
    k_exp = P.sum(axis=1)              # model-expected degrees
    annd_obs = compute_annd(A)
    cc_obs = compute_clustering(A)
    obs_global = compute_global_properties(A)
    log.info("  Observed:  %s", {k: f"{v:.4f}" for k, v in obs_global.items()})

    # ------------------------------------------------------------------
    # 4)  Sample ensemble
    # ------------------------------------------------------------------
    log.info("Sampling %d ensemble graphs...", n_ensemble)
    t0 = time.perf_counter()
    ensemble = sample_ensemble(P, n_ensemble, rng)
    log.info("  Sampling completed in %.1f s", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # 5)  Compute properties on ensemble
    # ------------------------------------------------------------------
    log.info("Computing ensemble properties...")
    ens_annd = np.zeros((n_ensemble, n))
    ens_cc   = np.zeros((n_ensemble, n))
    ens_global: dict[str, np.ndarray] = {k: np.zeros(n_ensemble) for k in obs_global}

    for i, A_s in enumerate(ensemble):
        ens_annd[i] = compute_annd(A_s)
        ens_cc[i]   = compute_clustering(A_s)
        gp = compute_global_properties(A_s)
        for k in obs_global:
            if k in gp:
                ens_global[k][i] = gp[k]

    # ------------------------------------------------------------------
    # 6)  Z-scores
    # ------------------------------------------------------------------
    log.info("Z-scores:")
    zscores:       dict[str, float] = {}
    zscore_detail: dict[str, dict]  = {}
    for prop in obs_global:
        if prop not in ens_global:
            continue
        z, mu, sigma = compute_zscore(obs_global[prop], ens_global[prop])
        zscores[prop] = z
        zscore_detail[prop] = {
            "observed":      obs_global[prop],
            "ensemble_mean": mu,
            "ensemble_std":  sigma,
            "z_score":       z,
        }
        log.info(
            "  %-20s  obs=%8.4f   <X>=%8.4f   σ=%8.4f   z=%+.2f  %s",
            prop, obs_global[prop], mu, sigma, z,
            " ***" if abs(z) > 2 else "",
        )

    # ------------------------------------------------------------------
    # 7)  Plots
    # ------------------------------------------------------------------
    log.info("Generating plots...")
    tag = name.lower().replace(" ", "_")

    # 7a — Degree scatter
    plot_degree_scatter(
        k_obs, k_exp, name,
        dirs["fig"] / f"{tag}_degree_scatter.png",
    )

    # 7b — ANND comparison
    plot_annd_comparison(
        annd_obs,
        ens_annd.mean(axis=0),
        ens_annd.std(axis=0),
        k_obs, name,
        dirs["fig"] / f"{tag}_annd_comparison.png",
    )

    # 7c — Clustering comparison
    plot_clustering_comparison(
        cc_obs,
        ens_cc.mean(axis=0),
        ens_cc.std(axis=0),
        k_obs, name,
        dirs["fig"] / f"{tag}_clustering_comparison.png",
    )

    # 7d — Z-score bar chart
    plot_zscore_summary(
        zscores, name,
        dirs["fig"] / f"{tag}_zscore_summary.png",
    )

    # 7e — Ensemble histograms for key properties
    for prop in ["transitivity", "avg_clustering", "n_triangles"]:
        if prop in obs_global and prop in ens_global:
            plot_ensemble_histogram(
                obs_global[prop], ens_global[prop], prop, name,
                dirs["fig"] / f"{tag}_{prop}_distribution.png",
            )

    return {
        "network":    name,
        "n":          n,
        "m":          int(A.sum()) // 2,
        "fit_time_s": fit_time,
        "z_scores":   zscore_detail,
    }


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="UBCM null-model analysis with z-scores",
    )
    ap.add_argument("--outdir",   type=str, default="results")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument(
        "--ensemble", type=int, default=500,
        help="Number of ensemble graphs to sample (default: 500)",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    dirs: dict[str, Path] = {
        "out": outdir,
        "fig": outdir / "figures",
        "tab": outdir / "tables",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # --- Load networks ---
    networks = [load_karate(), load_les_miserables()]

    all_results: list[dict] = []
    for G, name in networks:
        result = analyse_network(G, name, dirs, args.ensemble, rng)
        all_results.append(result)

    # --- Save z-score summary table ---
    rows: list[dict] = []
    for result in all_results:
        for prop, details in result["z_scores"].items():
            rows.append({
                "network": result["network"],
                "n":       result["n"],
                "m":       result["m"],
                "property": prop,
                **details,
            })
    zscore_df = pd.DataFrame(rows)
    zscore_path = dirs["tab"] / "zscore_analysis.csv"
    zscore_df.to_csv(zscore_path, index=False)
    log.info("Z-score table saved: %s", zscore_path)

    # --- Metadata ---
    meta = {
        "script":         "analysis.py",
        "seed":           args.seed,
        "n_ensemble":     args.ensemble,
        "python_version": platform.python_version(),
        "platform":       platform.platform(),
        "numpy_version":  np.__version__,
        "networkx_version": nx.__version__,
    }
    meta_path = dirs["tab"] / "analysis_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info("Metadata saved: %s", meta_path)

    log.info("=" * 60)
    log.info("Analysis completed — %d networks processed.", len(all_results))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
