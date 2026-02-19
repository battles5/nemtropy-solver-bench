#!/usr/bin/env python
"""
ERGM / Maximum-Entropy solver benchmark.

Reproduces and extends the solver comparison from Vallarano et al. (2021):
  newton vs quasinewton vs fixed-point on UBCM and DBCM models.

Toolbox: NEMtropy  –  https://github.com/nicoloval/NEMtropy

Networks
--------
- Zachary's Karate Club  (n=34,  empirical, UBCM)
- Les Miserables         (n=77,  empirical, UBCM)
- Synthetic G(n, p) undirected  (UBCM)
- Synthetic G(n, p) directed    (DBCM)

Metrics
-------
- Wall-clock runtime (seconds)
- Maximum relative error on degree constraints
- Mean relative error on degree constraints
- Convergence flag and number of solver steps (when exposed by NEMtropy)

Usage
-----
    python src/experiment.py --help
    python src/experiment.py --outdir results --seed 42 --runs 5
    python src/experiment.py --sizes "50,100,200,500,1000,2000" --runs 5
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METHODS: list[str] = ["newton", "quasinewton", "fixed-point"]
UBCM_MODEL = "cm_exp"
DBCM_MODEL = "dcm_exp"
MAX_STEPS = 500  # generous upper bound for all solvers


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    """Container for a single solver run."""

    network: str
    model: str
    method: str
    n: int
    m: int
    run: int
    runtime_s: float
    max_rel_err: float | None = None
    mean_rel_err: float | None = None
    converged: bool | None = None
    n_steps: int | None = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def ensure_dirs(outdir: Path) -> dict[str, Path]:
    """Create sub-directories for figures, tables and samples.

    Parameters
    ----------
    outdir : Path
        Root output directory.

    Returns
    -------
    dict[str, Path]
        Mapping name -> directory path.
    """
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
    """Convert a NetworkX graph to a binary adjacency matrix.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Input graph (potentially weighted).

    Returns
    -------
    np.ndarray
        Binary (0/1) adjacency matrix with zeros on the diagonal.
    """
    A = nx.to_numpy_array(G, dtype=float)
    A = (A > 0).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def _rel_errors(obs: np.ndarray, exp: np.ndarray) -> tuple[float, float]:
    """Compute max and mean relative error, ignoring zero-degree nodes.

    Parameters
    ----------
    obs : np.ndarray
        Observed degree sequence.
    exp : np.ndarray
        Expected degree sequence from the fitted model.

    Returns
    -------
    tuple[float, float]
        (max_relative_error, mean_relative_error).
    """
    mask = obs > 0
    if not mask.any():
        return 0.0, 0.0
    rel = np.abs(obs[mask] - exp[mask]) / obs[mask]
    return float(np.max(rel)), float(np.mean(rel))


def _extract_convergence(graph_obj) -> tuple[bool | None, int | None]:
    """Try to extract convergence info from a NEMtropy graph object.

    Parameters
    ----------
    graph_obj : UndirectedGraph or DirectedGraph
        Graph object after ``solve_tool()`` has been called.

    Returns
    -------
    tuple[bool | None, int | None]
        (converged, n_steps) — None if the attribute is not exposed.
    """
    converged = getattr(graph_obj, "is_converged", None)
    n_steps = getattr(graph_obj, "n_steps", None)
    if n_steps is None:
        n_steps = getattr(graph_obj, "nit", None)
    return converged, n_steps


# ---------------------------------------------------------------------------
# Solver wrappers
# ---------------------------------------------------------------------------
def solve_ubcm(
    A: np.ndarray,
    method: str,
    seed: int,
    sample_n: int = 0,
    sample_outdir: Path | None = None,
) -> RunResult:
    """Solve UBCM and collect timing + accuracy metrics.

    Parameters
    ----------
    A : np.ndarray
        Binary adjacency matrix (symmetric).
    method : str
        One of ``"newton"``, ``"quasinewton"``, ``"fixed-point"``.
    seed : int
        Random seed for initial guess.
    sample_n : int
        Number of ensemble samples to draw (0 = skip).
    sample_outdir : Path or None
        Directory for ensemble samples.

    Returns
    -------
    RunResult
        Populated result (network/run fields left as defaults to be set
        by the caller).
    """
    n = A.shape[0]
    g = UndirectedGraph(A)

    t0 = time.perf_counter()
    g.solve_tool(
        model=UBCM_MODEL,
        method=method,
        initial_guess="random",
        max_steps=MAX_STEPS,
        full_return=False,
        verbose=False,
    )
    runtime = time.perf_counter() - t0

    # --- error computation ---
    max_err: float | None = None
    mean_err: float | None = None
    k_obs = np.asarray(A.sum(axis=1)).ravel()

    exp_dseq = getattr(g, "expected_dseq", None)
    if exp_dseq is not None:
        k_exp = np.asarray(exp_dseq).ravel()
        max_err, mean_err = _rel_errors(k_obs, k_exp)

    converged, n_steps = _extract_convergence(g)

    # --- optional sampling ---
    if sample_n > 0 and sample_outdir is not None:
        sample_outdir.mkdir(parents=True, exist_ok=True)
        try:
            g.ensemble_sampler(
                n=sample_n, cpu_n=1,
                output_dir=str(sample_outdir) + "/",
                seed=seed,
            )
        except Exception as exc:
            log.warning("ensemble_sampler failed: %s", exc)

    return RunResult(
        network="", model=UBCM_MODEL, method=method,
        n=n, m=int(A.sum()) // 2, run=0,
        runtime_s=runtime,
        max_rel_err=max_err, mean_rel_err=mean_err,
        converged=converged, n_steps=n_steps,
    )


def solve_dbcm(
    A: np.ndarray,
    method: str,
    seed: int,
    sample_n: int = 0,
    sample_outdir: Path | None = None,
) -> RunResult:
    """Solve DBCM and collect timing + accuracy metrics.

    Parameters
    ----------
    A : np.ndarray
        Binary adjacency matrix (asymmetric / directed).
    method : str
        One of ``"newton"``, ``"quasinewton"``, ``"fixed-point"``.
    seed : int
        Random seed for initial guess.
    sample_n : int
        Number of ensemble samples to draw (0 = skip).
    sample_outdir : Path or None
        Directory for ensemble samples.

    Returns
    -------
    RunResult
        Populated result (network/run fields left as defaults to be set
        by the caller).
    """
    n = A.shape[0]
    g = DirectedGraph(A)

    t0 = time.perf_counter()
    g.solve_tool(
        model=DBCM_MODEL,
        method=method,
        initial_guess="random",
        max_steps=MAX_STEPS,
        full_return=False,
        verbose=False,
    )
    runtime = time.perf_counter() - t0

    # --- error computation ---
    max_err: float | None = None
    mean_err: float | None = None

    exp_dseq = getattr(g, "expected_dseq", None)
    if exp_dseq is not None:
        kout_obs = np.asarray(A.sum(axis=1)).ravel()
        kin_obs = np.asarray(A.sum(axis=0)).ravel()
        exp_arr = np.asarray(exp_dseq).ravel()
        kout_exp = exp_arr[:n]
        kin_exp = exp_arr[n : 2 * n]
        max_out, mean_out = _rel_errors(kout_obs, kout_exp)
        max_in, mean_in = _rel_errors(kin_obs, kin_exp)
        max_err = max(max_out, max_in)
        mean_err = (mean_out + mean_in) / 2.0

    converged, n_steps = _extract_convergence(g)

    # --- optional sampling ---
    if sample_n > 0 and sample_outdir is not None:
        sample_outdir.mkdir(parents=True, exist_ok=True)
        try:
            g.ensemble_sampler(
                n=sample_n, cpu_n=1,
                output_dir=str(sample_outdir) + "/",
                seed=seed,
            )
        except Exception as exc:
            log.warning("ensemble_sampler failed: %s", exc)

    return RunResult(
        network="", model=DBCM_MODEL, method=method,
        n=n, m=int(A.sum()), run=0,
        runtime_s=runtime,
        max_rel_err=max_err, mean_rel_err=mean_err,
        converged=converged, n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# Synthetic network generators
# ---------------------------------------------------------------------------
def gen_undirected(n: int, p: float, seed: int) -> nx.Graph:
    """Generate an Erdos-Renyi undirected graph G(n, p)."""
    return nx.gnp_random_graph(n=n, p=p, seed=seed, directed=False)


def gen_directed(n: int, p: float, seed: int) -> nx.DiGraph:
    """Generate an Erdos-Renyi directed graph G(n, p)."""
    return nx.gnp_random_graph(n=n, p=p, seed=seed, directed=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_runtime_scaling(df: pd.DataFrame, figpath: Path) -> None:
    """Log-scale runtime vs n, separate panels for UBCM and DBCM.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark data (synthetic networks only).
    figpath : Path
        Output file path.
    """
    models = sorted(df["model"].unique())
    n_panels = len(models)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(7 * n_panels, 5), squeeze=False,
    )

    for ax, model in zip(axes[0], models):
        sub = df[df["model"] == model]
        agg = (
            sub.groupby(["method", "n"])["runtime_s"]
            .agg(["mean", "std"])
            .reset_index()
        )
        for method, msub in agg.groupby("method"):
            msub = msub.sort_values("n")
            ax.errorbar(
                msub["n"], msub["mean"], yerr=msub["std"],
                marker="o", linewidth=2, capsize=4, label=method,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n (number of nodes)")
        ax.set_ylabel("runtime (s)")
        label = model.upper().replace("_EXP", "")
        ax.set_title(f"{label} — runtime scaling")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", figpath)


def plot_karate_methods(df: pd.DataFrame, figpath: Path) -> None:
    """Bar plot of runtime for the three methods on Karate Club.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark data (Karate Club rows only).
    figpath : Path
        Output file path.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    agg = df.groupby("method")["runtime_s"].agg(["mean", "std"]).reset_index()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    ax.bar(
        agg["method"], agg["mean"], yerr=agg["std"],
        capsize=5, color=colors[: len(agg)],
        edgecolor="black", linewidth=0.5,
    )
    ax.set_ylabel("runtime (s)")
    ax.set_title("UBCM on Karate Club — solver comparison")
    fig.tight_layout()
    fig.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", figpath)


def plot_error_heatmap(df: pd.DataFrame, figpath: Path) -> None:
    """Heatmap of max relative error, separate panels for UBCM and DBCM.

    Parameters
    ----------
    df : pd.DataFrame
        Full benchmark data.
    figpath : Path
        Output file path.
    """
    sub = df.dropna(subset=["max_rel_err"])
    if sub.empty:
        return

    models = sorted(sub["model"].unique())
    n_panels = len(models)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(7 * n_panels, 4), squeeze=False,
    )

    for ax, model in zip(axes[0], models):
        msub = sub[sub["model"] == model]
        pivot = (
            msub.groupby(["n", "method"])["max_rel_err"]
            .mean()
            .unstack(fill_value=0)
        )
        try:
            import seaborn as _sns

            _sns.heatmap(
                pivot, annot=True, fmt=".2e", cmap="YlOrRd",
                ax=ax, linewidths=0.5,
            )
        except Exception:
            ax.imshow(pivot.values, aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
        label = model.upper().replace("_EXP", "")
        ax.set_title(f"Max relative error — {label}")

    fig.tight_layout()
    fig.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", figpath)


def plot_error_vs_n(df: pd.DataFrame, figpath: Path) -> None:
    """Log-scale plot of mean relative error vs n for each method.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark data (synthetic networks only).
    figpath : Path
        Output file path.
    """
    sub = df.dropna(subset=["mean_rel_err"])
    if sub.empty:
        return

    models = sorted(sub["model"].unique())
    n_panels = len(models)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(7 * n_panels, 5), squeeze=False,
    )

    for ax, model in zip(axes[0], models):
        msub = sub[sub["model"] == model]
        agg = (
            msub.groupby(["method", "n"])["mean_rel_err"]
            .agg(["mean", "std"])
            .reset_index()
        )
        for method, gsub in agg.groupby("method"):
            gsub = gsub.sort_values("n")
            ax.errorbar(
                gsub["n"], gsub["mean"], yerr=gsub["std"],
                marker="s", linewidth=2, capsize=4, label=method,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n (number of nodes)")
        ax.set_ylabel("mean relative error")
        label = model.upper().replace("_EXP", "")
        ax.set_title(f"Mean relative error — {label}")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", figpath)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "NEMtropy benchmark: ERGM solver comparison "
            "(Vallarano et al. 2021)"
        ),
    )
    ap.add_argument(
        "--outdir", type=str, default="results",
        help="Output directory (default: results)",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="Global RNG seed",
    )
    ap.add_argument(
        "--runs", type=int, default=5,
        help="Independent runs per (method, network) combination",
    )
    ap.add_argument(
        "--sample_n", type=int, default=0,
        help="Ensemble graphs to sample per solved model (0 = skip)",
    )
    ap.add_argument(
        "--sizes", type=str, default="50,100,200,500,1000,2000",
        help="Comma-separated network sizes for synthetic G(n,p)",
    )
    ap.add_argument(
        "--p", type=float, default=0.05,
        help="Edge probability for synthetic G(n,p) networks",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    dirs = ensure_dirs(outdir)
    rng = np.random.default_rng(args.seed)
    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]

    results: list[RunResult] = []

    # -------------------------------------------------------------------
    # 1) Karate Club (UBCM)
    # -------------------------------------------------------------------
    log.info("=" * 60)
    log.info("1) Karate Club — UBCM (%s)", UBCM_MODEL)
    log.info("=" * 60)

    Gk = nx.karate_club_graph()
    Ak = adjacency_from_graph(Gk)

    for method in METHODS:
        for r in range(args.runs):
            seed_i = int(rng.integers(0, 1_000_000))
            res = solve_ubcm(
                Ak, method=method, seed=seed_i,
                sample_n=args.sample_n if r == 0 else 0,
                sample_outdir=(
                    dirs["samples"] / f"karate_{UBCM_MODEL}_{method}"
                    if args.sample_n > 0 else None
                ),
            )
            res.network = "karate_club"
            res.run = r
            results.append(res)
            log.info(
                "  %-14s run=%d  t=%.4fs  max_err=%s  mean_err=%s",
                method, r, res.runtime_s, res.max_rel_err, res.mean_rel_err,
            )

    # -------------------------------------------------------------------
    # 1b) Les Miserables (UBCM) — second real network, n=77
    # -------------------------------------------------------------------
    log.info("=" * 60)
    log.info("1b) Les Miserables — UBCM (%s)", UBCM_MODEL)
    log.info("=" * 60)

    Glm = nx.les_miserables_graph()
    Glm = nx.convert_node_labels_to_integers(Glm)
    Alm = adjacency_from_graph(Glm)

    for method in METHODS:
        for r in range(args.runs):
            seed_i = int(rng.integers(0, 1_000_000))
            res = solve_ubcm(Alm, method=method, seed=seed_i)
            res.network = "les_miserables"
            res.run = r
            results.append(res)
            log.info(
                "  %-14s run=%d  t=%.4fs  max_err=%s  mean_err=%s",
                method, r, res.runtime_s, res.max_rel_err, res.mean_rel_err,
            )

    # -------------------------------------------------------------------
    # 2) Synthetic undirected — UBCM
    # -------------------------------------------------------------------
    log.info("=" * 60)
    log.info("2) Synthetic undirected — UBCM (%s)", UBCM_MODEL)
    log.info("=" * 60)

    for n in sizes:
        Gu = gen_undirected(n, args.p, int(rng.integers(0, 1_000_000)))
        Au = adjacency_from_graph(Gu)
        for method in METHODS:
            for r in range(args.runs):
                res = solve_ubcm(
                    Au, method=method,
                    seed=int(rng.integers(0, 1_000_000)),
                )
                res.network = f"gnp_undirected_p{args.p}"
                res.run = r
                results.append(res)
                log.info(
                    "  n=%-5d %-14s run=%d  t=%.4fs  max_err=%s",
                    n, method, r, res.runtime_s, res.max_rel_err,
                )

    # -------------------------------------------------------------------
    # 3) Synthetic directed — DBCM
    # -------------------------------------------------------------------
    log.info("=" * 60)
    log.info("3) Synthetic directed — DBCM (%s)", DBCM_MODEL)
    log.info("=" * 60)

    for n in sizes:
        Gd = gen_directed(n, args.p, int(rng.integers(0, 1_000_000)))
        Ad = adjacency_from_graph(Gd)
        for method in METHODS:
            for r in range(args.runs):
                res = solve_dbcm(
                    Ad, method=method,
                    seed=int(rng.integers(0, 1_000_000)),
                )
                res.network = f"gnp_directed_p{args.p}"
                res.run = r
                results.append(res)
                log.info(
                    "  n=%-5d %-14s run=%d  t=%.4fs  max_err=%s",
                    n, method, r, res.runtime_s, res.max_rel_err,
                )

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    df = pd.DataFrame([asdict(r) for r in results])
    csv_path = dirs["tab"] / "benchmark.csv"
    df.to_csv(csv_path, index=False)
    log.info("Table saved: %s (%d rows)", csv_path, len(df))

    # -------------------------------------------------------------------
    # Summary table (aggregated by network × model × method × n)
    # -------------------------------------------------------------------
    summary = (
        df.groupby(["network", "model", "method", "n"])
        .agg(
            runtime_mean=("runtime_s", "mean"),
            runtime_std=("runtime_s", "std"),
            max_rel_err_mean=("max_rel_err", "mean"),
            mean_rel_err_mean=("mean_rel_err", "mean"),
            n_runs=("run", "count"),
        )
        .reset_index()
    )
    summary_path = dirs["tab"] / "summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Summary saved: %s", summary_path)

    # -------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------
    log.info("Generating figures...")

    df_synth = df[df["network"].str.contains("gnp")]
    if not df_synth.empty:
        plot_runtime_scaling(df_synth, dirs["fig"] / "runtime_vs_n.png")
        plot_error_vs_n(df_synth, dirs["fig"] / "error_vs_n.png")

    df_karate = df[df["network"] == "karate_club"]
    if not df_karate.empty:
        plot_karate_methods(df_karate, dirs["fig"] / "karate_runtime_methods.png")

    plot_error_heatmap(df, dirs["fig"] / "error_heatmap.png")

    # -------------------------------------------------------------------
    # Reproducibility metadata
    # -------------------------------------------------------------------
    meta = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "networkx_version": nx.__version__,
        "nemtropy_version": getattr(
            __import__("NEMtropy"), "__version__", "unknown",
        ),
        "seed": args.seed,
        "runs": args.runs,
        "sizes": sizes,
        "p": args.p,
        "max_steps": MAX_STEPS,
    }
    meta_path = dirs["tab"] / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info("Metadata saved: %s", meta_path)

    log.info("Experiment completed — %d total runs.", len(results))


if __name__ == "__main__":
    main()
