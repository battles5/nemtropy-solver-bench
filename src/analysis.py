#!/usr/bin/env python
"""
Null-model analysis: z-scores, analytical expectations, and observed vs
expected properties under Configuration Models.

Demonstrates the *application* of:
  - UBCM (Undirected Binary Configuration Model) on undirected networks
  - DBCM (Directed Binary Configuration Model)   on directed networks

following Squartini & Garlaschelli (2011, NJP) and Sections 8A-8B of the
Complex Network Analysis course.

For each network this script:
  1. Fits the appropriate Configuration Model via NEMtropy.
  2. Reconstructs the link-probability matrix.
  3. Computes **analytical** expected values where closed-form expressions
     exist (ANND, clustering under UBCM; reciprocity under DBCM).
  4. Samples a large ensemble for numerical z-scores on global properties.
  5. Compares analytical vs numerical expectations.
  6. Generates diagnostic and comparison plots.

Networks
--------
Undirected (UBCM):
  - Zachary Karate Club  (n = 34,  m = 78)
  - Les Miserables       (n = 77,  m ~ 254)

Directed (DBCM):
  - email-Eu-core (SNAP) (n = 1005, m = 25 571) — auto-downloaded

Usage
-----
    python src/analysis.py --help
    python src/analysis.py --outdir results --seed 42 --ensemble 1000
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import platform
import time
import urllib.request
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

from NEMtropy import UndirectedGraph, DirectedGraph

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
DBCM_MODEL = "dcm_exp"


# ===================================================================
# Network loaders
# ===================================================================

def load_karate() -> tuple[nx.Graph, str]:
    """Load Zachary's Karate Club (n=34, m=78)."""
    G = nx.karate_club_graph()
    return G, "Karate Club"


def load_les_miserables() -> tuple[nx.Graph, str]:
    """Load Les Miserables co-appearance (n=77). Binarised, int-labelled."""
    G = nx.les_miserables_graph()
    G_bin = nx.Graph()
    G_bin.add_edges_from(G.edges())
    G_bin = nx.convert_node_labels_to_integers(G_bin)
    return G_bin, "Les Miserables"


_EMAIL_EU_CORE_URL = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"


def load_email_eu_core(data_dir: Path = Path("data")) -> tuple[nx.DiGraph, str]:
    """Download (if needed) and load email-Eu-core from SNAP.

    Directed e-mail network of a large European research institution
    (n = 1 005, m = 25 571).

    Parameters
    ----------
    data_dir : Path
        Directory to cache the raw edge list.

    Returns
    -------
    tuple[nx.DiGraph, str]
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "email-Eu-core.txt"
    if not cache.exists():
        log.info("Downloading email-Eu-core from SNAP (%s)...", _EMAIL_EU_CORE_URL)
        resp = urllib.request.urlopen(_EMAIL_EU_CORE_URL, timeout=60)
        compressed = resp.read()
        raw = gzip.decompress(compressed).decode("utf-8")
        cache.write_text(raw, encoding="utf-8")
        log.info("  Saved to %s", cache)
    else:
        log.info("Loading email-Eu-core from cache: %s", cache)
        raw = cache.read_text(encoding="utf-8")

    G = nx.DiGraph()
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            u, v = int(parts[0]), int(parts[1])
            if u != v:  # no self-loops
                G.add_edge(u, v)
    G = nx.convert_node_labels_to_integers(G)
    return G, "email-Eu-core"


# ===================================================================
# UBCM fitting and probability matrix
# ===================================================================

def fit_ubcm(A: np.ndarray, method: str = "fixed-point") -> np.ndarray:
    """Fit UBCM and return x_i parameter vector (one per node)."""
    g = UndirectedGraph(A)
    g.solve_tool(
        model=UBCM_MODEL,
        method=method,
        initial_guess="random",
        max_steps=500,
        full_return=False,
        verbose=False,
    )
    return np.asarray(g.x).ravel()


def pij_matrix_ubcm(x: np.ndarray) -> np.ndarray:
    r"""UBCM probability matrix: :math:`p_{ij} = x_i x_j / (1 + x_i x_j)`."""
    xx = np.outer(x, x)
    P = xx / (1.0 + xx)
    np.fill_diagonal(P, 0.0)
    return P


# ===================================================================
# DBCM fitting and probability matrix
# ===================================================================

def fit_dbcm(
    A: np.ndarray, method: str = "fixed-point",
) -> tuple[np.ndarray, np.ndarray]:
    """Fit DBCM and return (x_out, y_in) parameter vectors.

    x_out controls out-degrees, y_in controls in-degrees.
    NEMtropy stores them in separate attributes ``g.x`` and ``g.y``.
    """
    g = DirectedGraph(A)
    g.solve_tool(
        model=DBCM_MODEL,
        method=method,
        initial_guess="random",
        max_steps=500,
        full_return=False,
        verbose=False,
    )
    x_out = np.asarray(g.x).ravel()
    y_in = np.asarray(g.y).ravel()
    return x_out, y_in


def pij_matrix_dbcm(x_out: np.ndarray, y_in: np.ndarray) -> np.ndarray:
    r"""DBCM probability matrix: :math:`p_{ij} = x_i y_j / (1 + x_i y_j)`."""
    xy = np.outer(x_out, y_in)
    P = xy / (1.0 + xy)
    np.fill_diagonal(P, 0.0)
    return P


# ===================================================================
# Ensemble sampling
# ===================================================================

def sample_ensemble_undirected(
    P: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Sample binary undirected graphs via Bernoulli trials on p_ij."""
    n = P.shape[0]
    P_upper = np.triu(P, k=1)
    graphs: list[np.ndarray] = []
    for _ in range(n_samples):
        R = rng.random((n, n))
        A = (R < P_upper).astype(float)
        A = A + A.T
        graphs.append(A)
    return graphs


def sample_ensemble_directed(
    P: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Sample binary directed graphs via Bernoulli trials on p_ij.

    For each ordered pair (i, j) with i != j, an independent Bernoulli
    trial with probability p_ij determines the link.
    """
    n = P.shape[0]
    graphs: list[np.ndarray] = []
    for _ in range(n_samples):
        R = rng.random((n, n))
        A = (R < P).astype(float)
        np.fill_diagonal(A, 0.0)
        graphs.append(A)
    return graphs


# ===================================================================
# Property computation — undirected
# ===================================================================

def compute_degrees(A: np.ndarray) -> np.ndarray:
    """Degree sequence."""
    return A.sum(axis=1)


def compute_annd(A: np.ndarray) -> np.ndarray:
    """Average Nearest-Neighbour Degree per node."""
    k = A.sum(axis=1)
    n = A.shape[0]
    annd = np.zeros(n)
    for i in range(n):
        nbrs = np.where(A[i] > 0)[0]
        if len(nbrs) > 0:
            annd[i] = k[nbrs].mean()
    return annd


def compute_annd_analytical(P: np.ndarray, k_obs: np.ndarray) -> np.ndarray:
    r"""Analytical expected ANND under UBCM.

    .. math::
        \langle k_{nn,i} \rangle_{CM} = \frac{1}{k_i}
        \sum_{j \neq i} p_{ij} \, k_j^{\mathrm{obs}}

    Exact to first order because the UBCM fixes
    :math:`\langle k_j \rangle = k_j^{\mathrm{obs}}`.
    """
    annd = np.zeros_like(k_obs, dtype=float)
    mask = k_obs > 0
    annd[mask] = (P[mask] @ k_obs) / k_obs[mask]
    return annd


def compute_clustering(A: np.ndarray) -> np.ndarray:
    """Local clustering coefficient per node."""
    G = nx.from_numpy_array(A)
    cc = nx.clustering(G)
    return np.array([cc[i] for i in range(len(cc))])


def compute_clustering_analytical(
    P: np.ndarray, k_obs: np.ndarray,
) -> np.ndarray:
    r"""Analytical expected local clustering under UBCM.

    Using the factorisation of link probabilities:

    .. math::
        \langle t_i \rangle = \frac{1}{2} \sum_j P_{ij}
        \bigl(P^2\bigr)_{ij}

    because each triple of links :math:`(a_{ij}, a_{il}, a_{jl})` is
    independent under the CM, so
    :math:`\langle a_{ij}\,a_{il}\,a_{jl} \rangle = p_{ij}\,p_{il}\,p_{jl}`.

    Then :math:`\langle c_i \rangle \approx 2\,\langle t_i \rangle
    \,/\, [k_i\,(k_i - 1)]`.
    """
    Q = P @ P  # Q_ij = sum_l P_il P_lj = P_i . P_j  (P is symmetric)
    t_exp = 0.5 * np.sum(P * Q, axis=1)  # expected triangles per node
    n = len(k_obs)
    cc_ana = np.zeros(n)
    denom = k_obs * (k_obs - 1)
    mask = denom > 0
    cc_ana[mask] = 2.0 * t_exp[mask] / denom[mask]
    return cc_ana


def count_triangles_nx(G: nx.Graph) -> int:
    """Total number of triangles (efficient NetworkX implementation)."""
    return sum(nx.triangles(G).values()) // 3


def compute_global_properties_undirected(A: np.ndarray) -> dict[str, float]:
    """Suite of scalar properties for an undirected network."""
    G = nx.from_numpy_array(A)
    n = A.shape[0]
    props: dict[str, float] = {
        "avg_clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
        "n_triangles": float(count_triangles_nx(G)),
        "density": nx.density(G),
    }
    if G.number_of_edges() > 0:
        try:
            props["assortativity"] = nx.degree_assortativity_coefficient(G)
        except Exception:
            pass
    return props


# ===================================================================
# Property computation — directed
# ===================================================================

def count_reciprocated_pairs(A: np.ndarray) -> int:
    """Number of reciprocated link pairs: L↔ = sum_{i<j} a_ij * a_ji."""
    return int(np.sum(A * A.T)) // 2


def reciprocity_analytical(P: np.ndarray) -> tuple[float, float]:
    r"""Analytical expected reciprocity and std under DBCM.

    Under the DCM, directed links are independent, so:

    .. math::
        \langle L^{\leftrightarrow} \rangle = \sum_{i<j} p_{ij}\,p_{ji}

    .. math::
        \sigma^2[L^{\leftrightarrow}] = \sum_{i<j}
        p_{ij}\,p_{ji}\,(1 - p_{ij}\,p_{ji})

    Returns
    -------
    tuple[float, float]
        (expected_reciprocity, std_reciprocity).
    """
    r = P * P.T  # r_ij = p_ij * p_ji
    np.fill_diagonal(r, 0.0)
    r_upper = np.triu(r, k=1)
    mean = float(np.sum(r_upper))
    var = float(np.sum(r_upper * (1.0 - r_upper)))
    return mean, np.sqrt(max(var, 0.0))


def compute_global_properties_directed(A: np.ndarray) -> dict[str, float]:
    """Suite of scalar properties for a directed network.

    Clustering and transitivity are computed on the undirected projection.
    """
    n = A.shape[0]
    A_und = np.maximum(A, A.T)
    G_und = nx.from_numpy_array(A_und)

    props: dict[str, float] = {
        "density": float(A.sum()) / (n * (n - 1)) if n > 1 else 0.0,
        "reciprocity": float(count_reciprocated_pairs(A)),
        "avg_clustering": nx.average_clustering(G_und),
        "transitivity": nx.transitivity(G_und),
        "n_triangles": float(count_triangles_nx(G_und)),
    }
    if G_und.number_of_edges() > 0:
        try:
            props["assortativity"] = nx.degree_assortativity_coefficient(G_und)
        except Exception:
            pass
    return props


# ===================================================================
# Z-score
# ===================================================================

def compute_zscore(
    obs: float, ens_vals: np.ndarray,
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
# Plots — helpers
# ===================================================================

def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", path)


# ===================================================================
# Plots — undirected
# ===================================================================

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
    annd_analytical: np.ndarray,
    k_obs: np.ndarray,
    name: str,
    figpath: Path,
) -> None:
    """ANND: observed vs ensemble vs analytical, plotted against degree."""
    fig, ax = plt.subplots(figsize=(7, 5))
    mask = k_obs > 0
    ax.errorbar(
        k_obs[mask], annd_exp_mean[mask], yerr=annd_exp_std[mask],
        fmt="o", color="steelblue", alpha=0.6, label="UBCM ensemble",
        capsize=3,
    )
    ax.scatter(
        k_obs[mask], annd_analytical[mask], color="green", marker="D",
        s=50, alpha=0.8, zorder=4, label="Analytical $\\langle k_{nn,i}\\rangle_{CM}$",
    )
    ax.scatter(
        k_obs[mask], annd_obs[mask], color="red", marker="x", s=80,
        linewidth=2, label="Observed", zorder=5,
    )
    ax.set_xlabel(r"Node degree $k_i$")
    ax.set_ylabel(r"$k_{nn,i}$  (ANND)")
    ax.set_title(f"ANND: observed vs UBCM — {name}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(fig, figpath)


def plot_clustering_comparison(
    cc_obs: np.ndarray,
    cc_exp_mean: np.ndarray,
    cc_exp_std: np.ndarray,
    cc_analytical: np.ndarray,
    k_obs: np.ndarray,
    name: str,
    figpath: Path,
) -> None:
    """Clustering coefficient: observed vs ensemble vs analytical, by degree."""
    fig, ax = plt.subplots(figsize=(7, 5))
    mask = k_obs > 0
    ax.errorbar(
        k_obs[mask], cc_exp_mean[mask], yerr=cc_exp_std[mask],
        fmt="o", color="steelblue", alpha=0.6, label="UBCM ensemble",
        capsize=3,
    )
    ax.scatter(
        k_obs[mask], cc_analytical[mask], color="green", marker="D",
        s=50, alpha=0.8, zorder=4, label="Analytical $\\langle c_i\\rangle_{CM}$",
    )
    ax.scatter(
        k_obs[mask], cc_obs[mask], color="red", marker="x", s=80,
        linewidth=2, label="Observed", zorder=5,
    )
    ax.set_xlabel(r"Node degree $k_i$")
    ax.set_ylabel(r"Clustering coefficient $c_i$")
    ax.set_title(f"Clustering: observed vs UBCM — {name}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(fig, figpath)


def plot_analytical_vs_numerical(
    analytical: np.ndarray,
    numerical_mean: np.ndarray,
    prop_name: str,
    name: str,
    figpath: Path,
) -> None:
    """Scatter: analytical expected value vs numerical ensemble mean."""
    fig, ax = plt.subplots(figsize=(6, 6))
    mask = (analytical > 0) | (numerical_mean > 0)
    ax.scatter(
        analytical[mask], numerical_mean[mask],
        alpha=0.7, edgecolors="black", linewidth=0.5, s=60, color="steelblue",
    )
    vals = np.concatenate([analytical[mask], numerical_mean[mask]])
    if len(vals) > 0:
        lo, hi = vals.min() * 0.9, vals.max() * 1.1
        lo = min(lo, 0)
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
    ax.set_xlabel(f"Analytical $\\langle {prop_name} \\rangle_{{CM}}$")
    ax.set_ylabel(f"Ensemble mean $\\langle {prop_name} \\rangle_{{ens}}$")
    ax.set_title(f"Analytical vs numerical: {prop_name} — {name}")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    _savefig(fig, figpath)


def plot_zscore_summary(
    zscores: dict[str, float],
    name: str,
    figpath: Path,
) -> None:
    """Horizontal bar chart of z-scores for global properties."""
    fig, ax = plt.subplots(figsize=(8, max(4, len(zscores) * 0.8)))
    props = list(zscores.keys())
    vals = [zscores[p] for p in props]
    colors = ["#d62728" if abs(v) > 2 else "#1f77b4" for v in vals]
    y_pos = range(len(props))
    ax.barh(y_pos, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(x=2, color="gray", linestyle="--", linewidth=0.8, label=r"$|z| = 2$")
    ax.axvline(x=-2, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(props)
    ax.set_xlabel("z-score")
    ax.set_title(f"Z-scores under null model — {name}")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _savefig(fig, figpath)


def plot_ensemble_histogram(
    obs_val: float,
    ens_vals: np.ndarray,
    prop_name: str,
    name: str,
    figpath: Path,
    analytical_val: float | None = None,
) -> None:
    """Histogram of ensemble property with observed (and analytical) markers."""
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
    if analytical_val is not None:
        ax.axvline(
            analytical_val, color="orange", linewidth=1.5, linestyle="-.",
            label=f"Analytical = {analytical_val:.4f}",
        )
    ax.set_xlabel(prop_name)
    ax.set_ylabel("Density")
    ax.set_title(f"{prop_name} distribution under null model — {name}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(fig, figpath)


# ===================================================================
# Plots — directed
# ===================================================================

def plot_degree_scatter_directed(
    kout_obs: np.ndarray,
    kout_exp: np.ndarray,
    kin_obs: np.ndarray,
    kin_exp: np.ndarray,
    name: str,
    figpath: Path,
) -> None:
    """Two-panel scatter: out-degree and in-degree observed vs expected."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    ax1.scatter(kout_obs, kout_exp, alpha=0.5, edgecolors="black", linewidth=0.3, s=30)
    lim = max(kout_obs.max(), kout_exp.max()) * 1.1
    ax1.plot([0, lim], [0, lim], "r--", linewidth=1)
    ax1.set_xlabel(r"Observed $k_i^{out}$")
    ax1.set_ylabel(r"Expected $\langle k_i^{out} \rangle$")
    ax1.set_title(f"Out-degree — {name}")
    ax1.set_aspect("equal")

    ax2.scatter(kin_obs, kin_exp, alpha=0.5, edgecolors="black", linewidth=0.3, s=30)
    lim = max(kin_obs.max(), kin_exp.max()) * 1.1
    ax2.plot([0, lim], [0, lim], "r--", linewidth=1)
    ax2.set_xlabel(r"Observed $k_i^{in}$")
    ax2.set_ylabel(r"Expected $\langle k_i^{in} \rangle$")
    ax2.set_title(f"In-degree — {name}")
    ax2.set_aspect("equal")

    fig.tight_layout()
    _savefig(fig, figpath)


def plot_reciprocity_analysis(
    recip_obs: int,
    recip_mean_ana: float,
    recip_std_ana: float,
    recip_ens: np.ndarray,
    name: str,
    figpath: Path,
) -> None:
    """Histogram of ensemble reciprocity + analytical + observed markers."""
    z_ana = (recip_obs - recip_mean_ana) / recip_std_ana if recip_std_ana > 1e-15 else 0
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        recip_ens, bins=35, alpha=0.7, color="steelblue",
        edgecolor="black", density=True, label="DBCM ensemble",
    )
    ax.axvline(
        recip_obs, color="red", linewidth=2.5, linestyle="--",
        label=f"Observed = {recip_obs}",
    )
    ax.axvline(
        recip_mean_ana, color="orange", linewidth=2, linestyle="-.",
        label=f"Analytical ⟨L↔⟩ = {recip_mean_ana:.1f}",
    )
    mu_ens = float(np.mean(recip_ens))
    ax.axvline(
        mu_ens, color="green", linewidth=1.5, linestyle=":",
        label=f"Ensemble mean = {mu_ens:.1f}",
    )
    ax.set_xlabel(r"Reciprocated pairs $L^{\leftrightarrow}$")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Reciprocity under DBCM — {name}\n"
        f"$z_{{analytical}} = {z_ana:+.2f}$"
        + (" ***" if abs(z_ana) > 2 else ""),
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(fig, figpath)


# ===================================================================
# Analysis pipeline — undirected (UBCM)
# ===================================================================

def analyse_undirected(
    G: nx.Graph,
    name: str,
    dirs: dict[str, Path],
    n_ensemble: int,
    rng: np.random.Generator,
) -> dict:
    """Full UBCM null-model analysis on an undirected network."""

    log.info("=" * 60)
    log.info("UBCM analysis: %s  (n=%d, m=%d)", name, G.number_of_nodes(), G.number_of_edges())
    log.info("=" * 60)

    A = nx.to_numpy_array(G, dtype=float)
    A = (A > 0).astype(float)
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]

    # 1) Fit UBCM
    log.info("Fitting UBCM (fixed-point)...")
    t0 = time.perf_counter()
    x = fit_ubcm(A, method="fixed-point")
    fit_time = time.perf_counter() - t0
    log.info("  Fit completed in %.3f s", fit_time)

    # 2) Probability matrix
    P = pij_matrix_ubcm(x)

    # 3) Observed properties
    log.info("Computing observed properties...")
    k_obs = compute_degrees(A)
    k_exp = P.sum(axis=1)
    annd_obs = compute_annd(A)
    cc_obs = compute_clustering(A)
    obs_global = compute_global_properties_undirected(A)
    log.info("  Observed: %s", {k: f"{v:.4f}" for k, v in obs_global.items()})

    # 4) Analytical expected values
    log.info("Computing analytical expected ANND and clustering...")
    annd_analytical = compute_annd_analytical(P, k_obs)
    cc_analytical = compute_clustering_analytical(P, k_obs)

    # 5) Sample ensemble
    log.info("Sampling %d ensemble graphs...", n_ensemble)
    t0 = time.perf_counter()
    ensemble = sample_ensemble_undirected(P, n_ensemble, rng)
    log.info("  Sampling completed in %.1f s", time.perf_counter() - t0)

    # 6) Compute ensemble properties
    log.info("Computing ensemble properties...")
    ens_annd = np.zeros((n_ensemble, n))
    ens_cc = np.zeros((n_ensemble, n))
    ens_global: dict[str, np.ndarray] = {k: np.zeros(n_ensemble) for k in obs_global}

    for i, A_s in enumerate(ensemble):
        if (i + 1) % 200 == 0:
            log.info("  Ensemble sample %d / %d", i + 1, n_ensemble)
        ens_annd[i] = compute_annd(A_s)
        ens_cc[i] = compute_clustering(A_s)
        gp = compute_global_properties_undirected(A_s)
        for k in obs_global:
            if k in gp:
                ens_global[k][i] = gp[k]

    # 7) Z-scores
    log.info("Z-scores (numerical ensemble):")
    zscores: dict[str, float] = {}
    zscore_detail: dict[str, dict] = {}
    for prop in obs_global:
        if prop not in ens_global:
            continue
        z, mu, sigma = compute_zscore(obs_global[prop], ens_global[prop])
        zscores[prop] = z
        zscore_detail[prop] = {
            "observed": obs_global[prop],
            "ensemble_mean": mu,
            "ensemble_std": sigma,
            "z_score": z,
        }
        log.info(
            "  %-20s  obs=%8.4f   <X>=%8.4f   sigma=%8.4f   z=%+.2f  %s",
            prop, obs_global[prop], mu, sigma, z,
            " ***" if abs(z) > 2 else "",
        )

    # 8) Plots
    log.info("Generating plots...")
    tag = name.lower().replace(" ", "_")

    plot_degree_scatter(
        k_obs, k_exp, name,
        dirs["fig"] / f"{tag}_degree_scatter.png",
    )
    plot_annd_comparison(
        annd_obs, ens_annd.mean(axis=0), ens_annd.std(axis=0),
        annd_analytical, k_obs, name,
        dirs["fig"] / f"{tag}_annd_comparison.png",
    )
    plot_clustering_comparison(
        cc_obs, ens_cc.mean(axis=0), ens_cc.std(axis=0),
        cc_analytical, k_obs, name,
        dirs["fig"] / f"{tag}_clustering_comparison.png",
    )
    plot_analytical_vs_numerical(
        annd_analytical, ens_annd.mean(axis=0),
        "k_{nn,i}", name,
        dirs["fig"] / f"{tag}_annd_analytical_vs_numerical.png",
    )
    plot_analytical_vs_numerical(
        cc_analytical, ens_cc.mean(axis=0),
        "c_i", name,
        dirs["fig"] / f"{tag}_clustering_analytical_vs_numerical.png",
    )
    plot_zscore_summary(
        zscores, name,
        dirs["fig"] / f"{tag}_zscore_summary.png",
    )
    for prop in ["transitivity", "avg_clustering", "n_triangles"]:
        if prop in obs_global and prop in ens_global:
            plot_ensemble_histogram(
                obs_global[prop], ens_global[prop], prop, name,
                dirs["fig"] / f"{tag}_{prop}_distribution.png",
            )

    return {
        "network": name,
        "type": "undirected",
        "model": "UBCM",
        "n": n,
        "m": int(A.sum()) // 2,
        "fit_time_s": fit_time,
        "z_scores": zscore_detail,
    }


# ===================================================================
# Analysis pipeline — directed (DBCM)
# ===================================================================

def analyse_directed(
    G: nx.DiGraph,
    name: str,
    dirs: dict[str, Path],
    n_ensemble: int,
    rng: np.random.Generator,
) -> dict:
    """Full DBCM null-model analysis on a directed network.

    Includes analytical reciprocity z-score and numerical z-scores for
    clustering, transitivity etc. on the undirected projection.
    """

    log.info("=" * 60)
    log.info("DBCM analysis: %s  (n=%d, m=%d)", name, G.number_of_nodes(), G.number_of_edges())
    log.info("=" * 60)

    A = nx.to_numpy_array(G, dtype=float)
    A = (A > 0).astype(float)
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]

    # 1) Fit DBCM
    log.info("Fitting DBCM (fixed-point)...")
    t0 = time.perf_counter()
    x_out, y_in = fit_dbcm(A, method="fixed-point")
    fit_time = time.perf_counter() - t0
    log.info("  Fit completed in %.3f s", fit_time)

    # 2) Probability matrix
    P = pij_matrix_dbcm(x_out, y_in)

    # 3) Verify degree reproduction
    kout_obs = A.sum(axis=1)
    kin_obs = A.sum(axis=0)
    kout_exp = P.sum(axis=1)
    kin_exp = P.sum(axis=0)
    max_err_out = float(np.max(np.abs(kout_obs - kout_exp) / np.maximum(kout_obs, 1)))
    max_err_in = float(np.max(np.abs(kin_obs - kin_exp) / np.maximum(kin_obs, 1)))
    log.info("  Degree reproduction: max rel err out=%.2e, in=%.2e", max_err_out, max_err_in)

    # 4) Observed properties
    log.info("Computing observed properties...")
    obs_global = compute_global_properties_directed(A)
    recip_obs = count_reciprocated_pairs(A)
    log.info("  Observed: %s", {k: f"{v:.4f}" for k, v in obs_global.items()})

    # 5) Analytical reciprocity z-score
    log.info("Computing analytical reciprocity z-score...")
    recip_mean_ana, recip_std_ana = reciprocity_analytical(P)
    if recip_std_ana > 1e-15:
        z_recip_ana = (recip_obs - recip_mean_ana) / recip_std_ana
    else:
        z_recip_ana = 0.0
    log.info(
        "  Reciprocity (analytical):  obs=%d   <L>=%8.1f   sigma=%8.1f   z=%+.2f  %s",
        recip_obs, recip_mean_ana, recip_std_ana, z_recip_ana,
        " ***" if abs(z_recip_ana) > 2 else "",
    )

    # 6) Sample directed ensemble
    log.info("Sampling %d directed ensemble graphs...", n_ensemble)
    t0 = time.perf_counter()
    ensemble = sample_ensemble_directed(P, n_ensemble, rng)
    log.info("  Sampling completed in %.1f s", time.perf_counter() - t0)

    # 7) Compute ensemble properties
    log.info("Computing ensemble properties (this may take a few minutes)...")
    ens_global: dict[str, np.ndarray] = {k: np.zeros(n_ensemble) for k in obs_global}

    for i, A_s in enumerate(ensemble):
        if (i + 1) % 100 == 0:
            log.info("  Ensemble sample %d / %d", i + 1, n_ensemble)
        gp = compute_global_properties_directed(A_s)
        for k in obs_global:
            if k in gp:
                ens_global[k][i] = gp[k]

    # 8) Z-scores (numerical)
    log.info("Z-scores (numerical ensemble):")
    zscores: dict[str, float] = {}
    zscore_detail: dict[str, dict] = {}
    for prop in obs_global:
        if prop not in ens_global:
            continue
        z, mu, sigma = compute_zscore(obs_global[prop], ens_global[prop])
        zscores[prop] = z
        zscore_detail[prop] = {
            "observed": obs_global[prop],
            "ensemble_mean": mu,
            "ensemble_std": sigma,
            "z_score": z,
        }
        log.info(
            "  %-20s  obs=%8.4f   <X>=%8.4f   sigma=%8.4f   z=%+.2f  %s",
            prop, obs_global[prop], mu, sigma, z,
            " ***" if abs(z) > 2 else "",
        )

    # Add analytical reciprocity z-score to the detail record
    zscore_detail["reciprocity_analytical"] = {
        "observed": float(recip_obs),
        "analytical_mean": recip_mean_ana,
        "analytical_std": recip_std_ana,
        "z_score_analytical": z_recip_ana,
    }

    # 9) Plots
    log.info("Generating plots...")
    tag = name.lower().replace(" ", "_").replace("-", "_")

    plot_degree_scatter_directed(
        kout_obs, kout_exp, kin_obs, kin_exp, name,
        dirs["fig"] / f"{tag}_degree_scatter.png",
    )
    plot_reciprocity_analysis(
        recip_obs, recip_mean_ana, recip_std_ana,
        ens_global.get("reciprocity", np.array([])),
        name,
        dirs["fig"] / f"{tag}_reciprocity_analysis.png",
    )
    plot_zscore_summary(
        {**zscores, "reciprocity (analytical)": z_recip_ana},
        name,
        dirs["fig"] / f"{tag}_zscore_summary.png",
    )
    for prop in ["avg_clustering", "transitivity", "n_triangles"]:
        if prop in obs_global and prop in ens_global:
            plot_ensemble_histogram(
                obs_global[prop], ens_global[prop], prop, name,
                dirs["fig"] / f"{tag}_{prop}_distribution.png",
            )

    return {
        "network": name,
        "type": "directed",
        "model": "DBCM",
        "n": n,
        "m": int(A.sum()),
        "fit_time_s": fit_time,
        "z_scores": zscore_detail,
        "reciprocity_analytical": {
            "z_score": z_recip_ana,
            "observed": recip_obs,
            "expected": recip_mean_ana,
            "std": recip_std_ana,
        },
    }


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Null-model analysis with z-scores (UBCM + DBCM).",
    )
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--ensemble", type=int, default=1000,
        help="Number of ensemble graphs to sample (default: 1000)",
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

    # ------------------------------------------------------------------
    # Undirected networks — UBCM
    # ------------------------------------------------------------------
    undirected_networks = [load_karate(), load_les_miserables()]

    all_results: list[dict] = []
    for G, name in undirected_networks:
        result = analyse_undirected(G, name, dirs, args.ensemble, rng)
        all_results.append(result)

    # ------------------------------------------------------------------
    # Directed networks — DBCM
    # ------------------------------------------------------------------
    G_email, name_email = load_email_eu_core()
    result_dir = analyse_directed(G_email, name_email, dirs, args.ensemble, rng)
    all_results.append(result_dir)

    # ------------------------------------------------------------------
    # Save z-score summary table
    # ------------------------------------------------------------------
    rows: list[dict] = []
    for result in all_results:
        for prop, details in result["z_scores"].items():
            rows.append({
                "network": result["network"],
                "type": result["type"],
                "model": result["model"],
                "n": result["n"],
                "m": result["m"],
                "property": prop,
                **{k: v for k, v in details.items() if not isinstance(v, (dict, list))},
            })
    zscore_df = pd.DataFrame(rows)
    zscore_path = dirs["tab"] / "zscore_analysis.csv"
    zscore_df.to_csv(zscore_path, index=False)
    log.info("Z-score table saved: %s", zscore_path)

    # --- Metadata ---
    meta = {
        "script": "analysis.py",
        "seed": args.seed,
        "n_ensemble": args.ensemble,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
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
