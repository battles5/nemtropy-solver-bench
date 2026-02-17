#!/usr/bin/env python
"""
ERGM / Maximum-Entropy solver benchmark (Complex Network Analysis course).

Reference: Vallarano et al. 2021 – runtime/accuracy comparison of
newton, quasinewton and fixed-point solvers on UBCM and DBCM models.
Toolbox: NEMtropy (Maximum Entropy Hub, IMT Lucca).

Networks: Karate Club (UBCM) + synthetic G(n,p) (UBCM/DBCM).

Usage:
    python src/experiment.py --help
    python src/experiment.py --outdir results --seed 42 --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# Optional styling
try:
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
except Exception:
    sns = None  # type: ignore

from NEMtropy import UndirectedGraph, DirectedGraph
from NEMtropy.models_functions import (
    expected_degree_cm,
    expected_out_degree_dcm_exp,
    expected_in_degree_dcm_exp,
    expected_decm_exp,
)

# ---------------------------------------------------------------------------
# Constants (from NEMtropy docs)
# ---------------------------------------------------------------------------
METHODS: List[str] = ["newton", "quasinewton", "fixed-point"]
UBCM_MODEL = "cm_exp"       # Undirected Binary Configuration Model (exp parametrization)
DBCM_MODEL = "dcm_exp"      # Directed Binary Configuration Model  (exp parametrization)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    network: str
    model: str
    method: str
    n: int
    m: int
    run: int
    runtime_s: float
    max_rel_err: Optional[float]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def ensure_dirs(outdir: Path) -> Dict[str, Path]:
    """Create sub-directories for figures, tables and samples."""
    dirs = {
        "out": outdir,
        "fig": outdir / "figures",
        "tab": outdir / "tables",
        "samples": outdir / "samples",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def adjacency_from_graph(G: nx.Graph | nx.DiGraph) -> np.ndarray:
    """Convert a NetworkX graph to a binary adjacency matrix."""
    A = nx.to_numpy_array(G, dtype=float)
    A = (A > 0).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def rel_error(obs: np.ndarray, exp: np.ndarray) -> float:
    """Maximum relative error, excluding nodes with observed degree 0."""
    mask = obs > 0
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(obs[mask] - exp[mask]) / obs[mask]))


def _get_solution(graph_obj) -> Optional[np.ndarray]:
    """Retrieve the solution vector from a NEMtropy graph object after solve_tool."""
    # Try solution_array (reduced) first, then x/xy
    for attr in ("solution_array", "x", "xy", "solution", "sol", "r_x", "theta"):
        val = getattr(graph_obj, attr, None)
        if val is not None:
            return np.asarray(val).ravel()
    return None


# ---------------------------------------------------------------------------
# Solver wrappers
# ---------------------------------------------------------------------------
def solve_ubcm(
    A: np.ndarray,
    method: str,
    seed: int,
    sample_n: int = 0,
    sample_outdir: Optional[Path] = None,
) -> Tuple[float, Optional[float]]:
    """Solve UBCM (cm_exp) and return (runtime_s, max_rel_err)."""
    g = UndirectedGraph(A)
    t0 = time.perf_counter()
    g.solve_tool(
        model=UBCM_MODEL,
        method=method,
        initial_guess="random",
        max_steps=300,
        full_return=False,
        verbose=False,
    )
    t1 = time.perf_counter()

    theta = _get_solution(g)
    maxerr: Optional[float] = None
    if theta is not None:
        k_obs = np.asarray(A.sum(axis=1)).ravel()
        try:
            k_exp = expected_degree_cm(theta)
            maxerr = rel_error(k_obs, k_exp)
        except Exception:
            # Fallback: use internally computed expected_dseq
            exp_dseq = getattr(g, "expected_dseq", None)
            if exp_dseq is not None:
                maxerr = rel_error(k_obs, np.asarray(exp_dseq).ravel())

    if sample_n > 0 and sample_outdir is not None:
        sample_outdir.mkdir(parents=True, exist_ok=True)
        try:
            g.ensemble_sampler(n=sample_n, cpu_n=1, output_dir=str(sample_outdir) + "/", seed=seed)
        except Exception as exc:
            warnings.warn(f"ensemble_sampler failed: {exc}")

    return (t1 - t0), maxerr


def solve_dbcm(
    A: np.ndarray,
    method: str,
    seed: int,
    sample_n: int = 0,
    sample_outdir: Optional[Path] = None,
) -> Tuple[float, Optional[float]]:
    """Solve DBCM (dcm_exp) and return (runtime_s, max_rel_err)."""
    g = DirectedGraph(A)
    t0 = time.perf_counter()
    g.solve_tool(
        model=DBCM_MODEL,
        method=method,
        initial_guess="random",
        max_steps=300,
        full_return=False,
        verbose=False,
    )
    t1 = time.perf_counter()

    maxerr: Optional[float] = None
    # Use internally computed expected_dseq from NEMtropy
    exp_dseq = getattr(g, "expected_dseq", None)
    if exp_dseq is not None:
        kout_obs = np.asarray(A.sum(axis=1)).ravel()
        kin_obs = np.asarray(A.sum(axis=0)).ravel()
        n = A.shape[0]
        exp_arr = np.asarray(exp_dseq).ravel()
        kout_exp = exp_arr[:n]
        kin_exp = exp_arr[n:2 * n]
        maxerr = max(rel_error(kout_obs, kout_exp),
                     rel_error(kin_obs, kin_exp))
    else:
        theta = _get_solution(g)
        if theta is not None:
            kout_obs = np.asarray(A.sum(axis=1)).ravel()
            kin_obs = np.asarray(A.sum(axis=0)).ravel()
            n = A.shape[0]
            try:
                exp_all = expected_decm_exp(theta)
                kout_exp = exp_all[:n]
                kin_exp = exp_all[n:]
                maxerr = max(rel_error(kout_obs, kout_exp),
                             rel_error(kin_obs, kin_exp))
            except Exception:
                pass

    if sample_n > 0 and sample_outdir is not None:
        sample_outdir.mkdir(parents=True, exist_ok=True)
        try:
            g.ensemble_sampler(n=sample_n, cpu_n=1, output_dir=str(sample_outdir) + "/", seed=seed)
        except Exception as exc:
            warnings.warn(f"ensemble_sampler failed: {exc}")

    return (t1 - t0), maxerr


# ---------------------------------------------------------------------------
# Synthetic network generators
# ---------------------------------------------------------------------------
def gen_synthetic_undirected(n: int, p: float, seed: int) -> nx.Graph:
    return nx.gnp_random_graph(n=n, p=p, seed=seed, directed=False)


def gen_synthetic_directed(n: int, p: float, seed: int) -> nx.DiGraph:
    return nx.gnp_random_graph(n=n, p=p, seed=seed, directed=True)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_runtime_comparison(
    df: pd.DataFrame,
    figpath: Path,
    title: str = "Runtime vs n (NEMtropy solve_tool)",
) -> None:
    """Line plot of runtime grouped by (model, method)."""
    fig, ax = plt.subplots(figsize=(9, 5))

    agg = df.groupby(["model", "method", "n"])["runtime_s"].mean().reset_index()
    for (model, method), sub in agg.groupby(["model", "method"]):
        sub = sub.sort_values("n")
        ax.plot(sub["n"], sub["runtime_s"], marker="o", linewidth=2, label=f"{model} | {method}")

    ax.set_xlabel("n (number of nodes)")
    ax.set_ylabel("mean runtime (s)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {figpath}")


def plot_karate_methods(df: pd.DataFrame, figpath: Path) -> None:
    """Bar plot of runtime for the three methods on Karate Club."""
    fig, ax = plt.subplots(figsize=(7, 5))

    agg = df.groupby("method")["runtime_s"].agg(["mean", "std"]).reset_index()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = ax.bar(agg["method"], agg["mean"], yerr=agg["std"], capsize=5, color=colors[: len(agg)])
    ax.set_ylabel("mean runtime (s)")
    ax.set_title("UBCM (cm_exp) on Karate Club – method comparison")
    fig.tight_layout()
    fig.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {figpath}")


def plot_error_heatmap(df: pd.DataFrame, figpath: Path) -> None:
    """Heatmap of max_rel_err by (n, method) – only non-null rows."""
    sub = df.dropna(subset=["max_rel_err"])
    if sub.empty:
        return
    pivot = sub.groupby(["n", "method"])["max_rel_err"].mean().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        import seaborn as _sns
        _sns.heatmap(pivot, annot=True, fmt=".2e", cmap="YlOrRd", ax=ax)
    except Exception:
        ax.imshow(pivot.values, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
    ax.set_title("Max relative error on degree constraints")
    fig.tight_layout()
    fig.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {figpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="NEMtropy benchmark: ERGM solver comparison (Vallarano et al. 2021)"
    )
    ap.add_argument("--outdir", type=str, default="results", help="Output directory (default: results)")
    ap.add_argument("--seed", type=int, default=42, help="Global seed")
    ap.add_argument("--runs", type=int, default=3, help="Runs per method")
    ap.add_argument("--sample_n", type=int, default=0, help="Number of ensemble graphs to sample (0=skip)")
    ap.add_argument("--sizes", type=str, default="50,100,200", help="Synthetic network sizes, e.g. 50,100,200")
    ap.add_argument("--p", type=float, default=0.05, help="Edge probability for synthetic G(n,p) networks")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    dirs = ensure_dirs(outdir)
    rng = np.random.default_rng(args.seed)
    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]

    results: List[RunResult] = []

    # -----------------------------------------------------------------------
    # 1) Karate Club (UBCM)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("1) Karate Club – UBCM (cm_exp)")
    print("=" * 60)
    Gk = nx.karate_club_graph()
    Ak = adjacency_from_graph(Gk)
    n_k = Ak.shape[0]
    m_k = int(Ak.sum() // 2)

    for method in METHODS:
        for r in range(args.runs):
            seed_i = int(rng.integers(0, 1_000_000))
            rt, err = solve_ubcm(
                Ak,
                method=method,
                seed=seed_i,
                sample_n=args.sample_n if r == 0 else 0,
                sample_outdir=(dirs["samples"] / f"karate_{UBCM_MODEL}_{method}") if args.sample_n > 0 else None,
            )
            results.append(RunResult("karate_club", UBCM_MODEL, method, n_k, m_k, r, rt, err))
            print(f"  {method:14s} run={r}  t={rt:.4f}s  max_err={err}")

    # -----------------------------------------------------------------------
    # 2) Reti sintetiche – UBCM
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2) Synthetic undirected networks – UBCM (cm_exp)")
    print("=" * 60)
    for n in sizes:
        Gu = gen_synthetic_undirected(n=n, p=args.p, seed=int(rng.integers(0, 1_000_000)))
        Au = adjacency_from_graph(Gu)
        m_u = int(Au.sum() // 2)
        for method in METHODS:
            for r in range(args.runs):
                rt, err = solve_ubcm(Au, method=method, seed=int(rng.integers(0, 1_000_000)))
                results.append(RunResult(f"gnp_undirected_p{args.p}", UBCM_MODEL, method, n, m_u, r, rt, err))
                print(f"  n={n:4d}  {method:14s} run={r}  t={rt:.4f}s  max_err={err}")

    # -----------------------------------------------------------------------
    # 3) Reti sintetiche – DBCM
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3) Synthetic directed networks – DBCM (dcm_exp)")
    print("=" * 60)
    for n in sizes:
        Gd = gen_synthetic_directed(n=n, p=args.p, seed=int(rng.integers(0, 1_000_000)))
        Ad = adjacency_from_graph(Gd)
        m_d = int(Ad.sum())
        for method in METHODS:
            for r in range(args.runs):
                rt, err = solve_dbcm(Ad, method=method, seed=int(rng.integers(0, 1_000_000)))
                results.append(RunResult(f"gnp_directed_p{args.p}", DBCM_MODEL, method, n, m_d, r, rt, err))
                print(f"  n={n:4d}  {method:14s} run={r}  t={rt:.4f}s  max_err={err}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    df = pd.DataFrame([asdict(r) for r in results])
    csv_path = dirs["tab"] / "benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Table: {csv_path}")

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    print("\nGenerating figures...")

    # Runtime vs n (solo reti sintetiche)
    df_synth = df[df["network"].str.contains("gnp")]
    if not df_synth.empty:
        plot_runtime_comparison(df_synth, dirs["fig"] / "runtime_vs_n.png")

    # Karate Club – method comparison
    df_karate = df[df["network"] == "karate_club"]
    if not df_karate.empty:
        plot_karate_methods(df_karate, dirs["fig"] / "karate_runtime_methods.png")

    # Error heatmap (optional)
    plot_error_heatmap(df, dirs["fig"] / "error_heatmap.png")

    # -----------------------------------------------------------------------
    # Reproducibility metadata
    # -----------------------------------------------------------------------
    meta = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seed": args.seed,
        "runs": args.runs,
        "sizes": sizes,
        "p": args.p,
        "nemtropy_ref": "Vallarano et al. 2021, NEMtropy (PyPI 3.0.3+)",
    }
    meta_path = dirs["tab"] / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[OK] Metadata: {meta_path}")

    print("\n[DONE] Experiment completed.")

    # -----------------------------------------------------------------------
    # Switch path (comment): Saracco 2015 (BiCM + MovieLens)
    # -----------------------------------------------------------------------
    # TODO: alternative pipeline:
    # 1) bash data/download_data.sh
    # 2) import bicm
    # 3) load data/raw/ml-100k/u.data  →  binary bipartite matrix users×movies
    # 4) bicm.BipartiteGraph(...)  →  fit BiCM, p-values
    # 5) FDR via statsmodels.stats.multitest.multipletests
    # 6) movie-movie projection and community detection


if __name__ == "__main__":
    main()
