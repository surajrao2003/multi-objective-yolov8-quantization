#!/usr/bin/env python3
"""
YOLOv8 configuration optimization pipeline (single command).

1. Runs csv_generator.py → configurations.csv (unless --skip-csv-gen).
2. Loads the CSV, scores configs, applies constraints, Pareto frontier.
3. Writes under output_csv_results/ (next to this script by default):
   full_results_with_utility.csv, optimization_report.md,
   and trade-off PNGs.

Utility: U = alpha * A_norm + beta * F_norm - gamma * S_norm (min-max over full CSV).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

REQUIRED_COLUMNS = ("model", "precision", "resolution", "hardware", "fps", "size_mb", "map50")
OPTIONAL_LATENCY_COL = "latency_ms"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _path_for_display(path: Path) -> str:
    """Paths relative to repo root for logs/reports (avoid embedding absolute paths)."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        pass
    try:
        return resolved.relative_to(Path.cwd()).as_posix()
    except ValueError:
        pass
    if path.is_absolute():
        return path.name
    return path.as_posix()


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=(
            "Regenerate configurations.csv (optional), rank YOLOv8 benchmarks by utility, "
            "emit CSV/Markdown/PNG under output_csv_results."
        )
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=here / "configurations.csv",
        help="Benchmark CSV (default: configurations.csv next to this script).",
    )
    p.add_argument(
        "--skip-csv-gen",
        action="store_true",
        help="Skip csv_generator.py; use existing --csv file.",
    )
    p.add_argument("--alpha", type=float, default=0.50, help="Weight on normalized accuracy (map50).")
    p.add_argument("--beta", type=float, default=0.30, help="Weight on normalized FPS.")
    p.add_argument("--gamma", type=float, default=0.20, help="Penalty weight on normalized model size.")
    p.add_argument("--amin", type=float, default=0.65, dest="a_min", help="Minimum map50 constraint.")
    p.add_argument("--lmax", type=float, default=100.0, dest="l_max", help="Maximum latency_ms constraint.")
    p.add_argument("--smax", type=float, default=50.0, dest="s_max", help="Maximum size_mb constraint.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output_csv_results",
        help="Directory for output CSVs (created if missing). Default: optimization/output_csv_results next to this script.",
    )
    p.add_argument(
        "--closest-k",
        type=int,
        default=10,
        help="When no rows are feasible, show this many closest configurations.",
    )
    p.add_argument(
        "--no-console-tables",
        action="store_true",
        help="Skip compact tabulate output to stdout (CSV/report/plot unchanged).",
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        help="Skip Markdown report (optimization_report.md).",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip FPS vs mAP trade-off figure.",
    )
    return p.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")


def ensure_latency_ms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if OPTIONAL_LATENCY_COL not in out.columns:
        out[OPTIONAL_LATENCY_COL] = np.nan
    need = out[OPTIONAL_LATENCY_COL].isna()
    if need.any():
        fps = pd.to_numeric(out.loc[need, "fps"], errors="coerce")
        invalid = fps.isna() | (fps <= 0)
        if invalid.any():
            bad_idx = out.index[need][invalid]
            raise ValueError(f"Cannot compute latency_ms: invalid fps at rows {list(bad_idx)}")
        out.loc[need, OPTIONAL_LATENCY_COL] = 1000.0 / fps
    out[OPTIONAL_LATENCY_COL] = pd.to_numeric(out[OPTIONAL_LATENCY_COL], errors="coerce")
    return out


def min_max_normalize(series: pd.Series, name: str) -> tuple[pd.Series, float, float]:
    s = pd.to_numeric(series, errors="coerce")
    vmin = float(s.min())
    vmax = float(s.max())
    if np.isnan(vmin) or np.isnan(vmax):
        raise ValueError(f"Cannot normalize '{name}': non-numeric or empty values.")
    span = vmax - vmin
    if span == 0:
        norm = pd.Series(0.0, index=s.index)
    else:
        norm = (s - vmin) / span
    return norm.clip(0.0, 1.0), vmin, vmax


def compute_utility(
    a_norm: pd.Series,
    f_norm: pd.Series,
    s_norm: pd.Series,
    alpha: float,
    beta: float,
    gamma: float,
) -> pd.Series:
    return alpha * a_norm + beta * f_norm - gamma * s_norm


def pareto_optimal_mask(df: pd.DataFrame) -> pd.Series:
    """
    True where row is Pareto-optimal w.r.t. maximize map50, maximize fps, minimize size_mb.
    """
    m = pd.to_numeric(df["map50"], errors="coerce").to_numpy(dtype=float)
    f = pd.to_numeric(df["fps"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(df["size_mb"], errors="coerce").to_numpy(dtype=float)
    n = len(df)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            better_or_equal = (m[j] >= m[i]) and (f[j] >= f[i]) and (z[j] <= z[i])
            strictly_better = (m[j] > m[i]) or (f[j] > f[i]) or (z[j] < z[i])
            if better_or_equal and strictly_better:
                dominated[i] = True
                break
    return pd.Series(~dominated, index=df.index)


def constraint_violation_score(
    df: pd.DataFrame,
    a_min: float,
    l_max: float,
    s_max: float,
) -> pd.Series:
    """Non-negative aggregate violation; 0 iff feasible."""
    v_a = np.maximum(0.0, a_min - pd.to_numeric(df["map50"], errors="coerce"))
    v_l = np.maximum(0.0, pd.to_numeric(df["latency_ms"], errors="coerce") - l_max)
    v_s = np.maximum(0.0, pd.to_numeric(df["size_mb"], errors="coerce") - s_max)
    # Scale latency and size roughly to [0,1] ballpark using constraints to keep terms comparable
    return v_a + (v_l / max(l_max, 1e-9)) + (v_s / max(s_max, 1e-9))


def format_config_row(row: pd.Series) -> str:
    return (
        f"{row['model']} | {row['precision']} | res={row['resolution']} | "
        f"{row['hardware']} | fps={row['fps']:.4f} | latency_ms={row['latency_ms']:.2f} | "
        f"size_mb={row['size_mb']:.2f} | map50={row['map50']:.4f} | utility={row['utility']:.4f}"
    )


DISPLAY_TABLE_COLS = (
    "selection_type",
    "group",
    "rank",
    "model",
    "precision",
    "resolution",
    "hardware",
    "fps",
    "latency_ms",
    "size_mb",
    "map50",
    "utility",
    "feasible",
    "constraint_violation",
)


def _subset_display_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DISPLAY_TABLE_COLS if c in df.columns]
    out = df[cols].copy()
    if "fps" in out.columns:
        out["fps"] = out["fps"].round(2)
    if "latency_ms" in out.columns:
        out["latency_ms"] = out["latency_ms"].round(2)
    if "size_mb" in out.columns:
        out["size_mb"] = out["size_mb"].round(2)
    if "map50" in out.columns:
        out["map50"] = out["map50"].round(4)
    if "utility" in out.columns:
        out["utility"] = out["utility"].round(4)
    if "constraint_violation" in out.columns:
        out["constraint_violation"] = out["constraint_violation"].round(4)
    return out


def print_console_tables(
    *,
    feasible_ranked: pd.DataFrame,
    feasible_any: bool,
    closest: pd.DataFrame | None,
    best_by_hardware: pd.DataFrame | None,
    best_by_precision: pd.DataFrame | None,
    skip: bool,
) -> None:
    if skip:
        return
    print()
    print("=== Compact tables (rounded for readability) ===")
    print()
    if feasible_any and len(feasible_ranked):
        sub = _subset_display_df(feasible_ranked.head(10))
        print(f"Top {len(sub)} feasible by utility")
        print(tabulate(sub, headers="keys", tablefmt="simple", showindex=False))
        print()
        if best_by_hardware is not None and len(best_by_hardware):
            print("Best feasible per hardware")
            print(tabulate(_subset_display_df(best_by_hardware), headers="keys", tablefmt="simple", showindex=False))
            print()
        if best_by_precision is not None and len(best_by_precision):
            print("Best feasible per precision")
            print(tabulate(_subset_display_df(best_by_precision), headers="keys", tablefmt="simple", showindex=False))
            print()
    elif closest is not None and len(closest):
        sub = _subset_display_df(closest)
        print(f"Closest configurations by constraint violation (no feasible set)")
        print(tabulate(sub, headers="keys", tablefmt="simple", showindex=False))
        print()


def df_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    sub = _subset_display_df(df)
    return tabulate(sub, headers="keys", tablefmt="github", showindex=False) + "\n"


def write_markdown_report(
    path: Path,
    *,
    csv_path: Path,
    args: argparse.Namespace,
    feasible_count: int,
    total: int,
    feasible_ranked: pd.DataFrame,
    pareto_sorted: pd.DataFrame,
    summary_df: pd.DataFrame,
    best_by_hardware: pd.DataFrame | None,
    best_by_precision: pd.DataFrame | None,
    closest: pd.DataFrame | None,
) -> None:
    lines = [
        "# YOLOv8 configuration optimization report\n",
        "## Inputs and settings\n",
        f"- **Benchmark CSV:** `{_path_for_display(csv_path)}`\n",
        f"- **Utility weights:** α={args.alpha}, β={args.beta}, γ={args.gamma}\n",
        f"- **Constraints:** map50 ≥ {args.a_min}, latency_ms ≤ {args.l_max}, size_mb ≤ {args.s_max}\n",
        f"- **Feasible configurations:** {feasible_count} / {total}\n",
        "\n",
        "## Best configuration summary\n",
        df_to_markdown_table(summary_df),
        "## Top feasible configurations (by utility)\n",
        df_to_markdown_table(feasible_ranked.head(15)),
        "## Pareto frontier (max map50 & fps, min size_mb)\n",
        df_to_markdown_table(pareto_sorted.head(40)),
    ]
    if best_by_hardware is not None and len(best_by_hardware):
        lines.extend(["## Best feasible per hardware\n", df_to_markdown_table(best_by_hardware)])
    if best_by_precision is not None and len(best_by_precision):
        lines.extend(["## Best feasible per precision\n", df_to_markdown_table(best_by_precision)])
    if closest is not None and len(closest):
        lines.extend(["## Closest configurations (infeasible case)\n", df_to_markdown_table(closest)])

    path.write_text("".join(lines), encoding="utf-8")


def write_tradeoff_plot(
    path: Path,
    df_plot: pd.DataFrame,
    *,
    title: str | None = None,
) -> bool:
    """Scatter: FPS vs map50; marker size ~ size_mb; fill color = feasible; Pareto = crimson edge."""
    d = df_plot.dropna(subset=["fps", "map50", "size_mb"])
    if len(d) == 0:
        return False
    s_min = float(d["size_mb"].min())
    s_max = float(d["size_mb"].max())
    span = max(s_max - s_min, 1e-9)
    sizes = (20.0 + 200.0 * (d["size_mb"].astype(float) - s_min) / span).to_numpy(dtype=float)
    fps = d["fps"].astype(float).to_numpy()
    map50 = d["map50"].astype(float).to_numpy()
    feas = d["feasible"].to_numpy(dtype=bool)
    pm = d["pareto_optimal"].to_numpy(dtype=bool)
    colors = np.where(feas, "#2ca02c", "#a6a6a6")
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    npm = ~pm
    if npm.any():
        ax.scatter(fps[npm], map50[npm], s=sizes[npm], c=colors[npm], alpha=0.72, edgecolors="none")
    if pm.any():
        ax.scatter(
            fps[pm],
            map50[pm],
            s=sizes[pm],
            c=colors[pm],
            alpha=0.85,
            linewidths=2.0,
            edgecolors="#c0392b",
        )
    ax.set_xlabel("FPS (throughput)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title(
        title
        if title is not None
        else "Throughput vs accuracy (marker area ∝ model size MB)"
    )
    ax.grid(True, alpha=0.3)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=10, label="Feasible"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#a6a6a6", markersize=10, label="Infeasible"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#888888",
            markeredgecolor="#c0392b",
            markersize=10,
            markeredgewidth=2,
            label="Pareto (ring)",
        ),
    ]
    ax.legend(handles=handles, loc="lower right")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    args = parse_args()
    here = Path(__file__).resolve().parent
    if not args.skip_csv_gen:
        gen = subprocess.run([sys.executable, str(here / "csv_generator.py")], cwd=here)
        if gen.returncode != 0:
            return int(gen.returncode)

    if not args.csv.is_file():
        print(f"Error: CSV not found: {_path_for_display(args.csv)}", file=sys.stderr)
        return 1

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.csv)
    validate_columns(df_raw)
    df = ensure_latency_ms(df_raw)

    for col in REQUIRED_COLUMNS:
        if col not in ("model", "precision", "hardware"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    a_norm, _, _ = min_max_normalize(df["map50"], "map50")
    f_norm, _, _ = min_max_normalize(df["fps"], "fps")
    s_norm, _, _ = min_max_normalize(df["size_mb"], "size_mb")

    df["A_norm"] = a_norm
    df["F_norm"] = f_norm
    df["S_norm"] = s_norm
    df["utility"] = compute_utility(a_norm, f_norm, s_norm, args.alpha, args.beta, args.gamma)

    feasible = (
        (df["map50"] >= args.a_min)
        & (df["latency_ms"] <= args.l_max)
        & (df["size_mb"] <= args.s_max)
    )
    df["feasible"] = feasible
    df["constraint_violation"] = constraint_violation_score(df, args.a_min, args.l_max, args.s_max)

    df_pareto = df.copy()
    df_pareto["pareto_optimal"] = pareto_optimal_mask(df_pareto)

    full_path = out_dir / "full_results_with_utility.csv"
    df_pareto.sort_values("utility", ascending=False).to_csv(full_path, index=False)
    print(f"Wrote {_path_for_display(full_path)}")

    pareto_only = df_pareto.loc[df_pareto["pareto_optimal"]].copy()

    print()
    print(
        f"Weights: alpha={args.alpha} beta={args.beta} gamma={args.gamma} | "
        f"Constraints: map50>={args.a_min}, latency_ms<={args.l_max}, size_mb<={args.s_max}"
    )
    print(f"Feasible configurations: {int(feasible.sum())} / {len(df)}")
    print()

    best_by_hardware: pd.DataFrame | None = None
    best_by_precision: pd.DataFrame | None = None
    closest_show: pd.DataFrame | None = None

    if not feasible.any():
        print("*** No configuration satisfies all constraints. ***")
        print("Consider relaxing --amin, --lmax, or --smax, or inspect closest runs below.")
        closest_show = df.nsmallest(args.closest_k, "constraint_violation")
        feasible_ranked = df.iloc[0:0].copy()
        best_row = None
    else:
        feasible_df = df.loc[feasible].copy()
        feasible_ranked = feasible_df.sort_values("utility", ascending=False).reset_index(drop=True)
        feasible_ranked.insert(0, "rank", range(1, len(feasible_ranked) + 1))
        best_row = feasible_ranked.iloc[0]

        print("Best overall (feasible):")
        print(f"  {format_config_row(best_row)}")
        print()

        best_by_hardware = feasible_ranked.groupby("hardware", sort=True).first().reset_index()
        best_by_precision = feasible_ranked.groupby("precision", sort=True).first().reset_index()

    summary_rows: list[dict] = []
    meta = {
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "A_min": args.a_min,
        "L_max": args.l_max,
        "S_max": args.s_max,
    }

    def add_summary(selection: str, group: str, row: pd.Series) -> None:
        summary_rows.append(
            {
                "selection_type": selection,
                "group": group,
                "utility": row["utility"],
                "feasible": bool(row["feasible"]),
                "model": row["model"],
                "precision": row["precision"],
                "resolution": row["resolution"],
                "hardware": row["hardware"],
                "fps": row["fps"],
                "latency_ms": row["latency_ms"],
                "size_mb": row["size_mb"],
                "map50": row["map50"],
                "A_norm": row["A_norm"],
                "F_norm": row["F_norm"],
                "S_norm": row["S_norm"],
                **{f"param_{k}": v for k, v in meta.items()},
            }
        )

    if best_row is not None:
        add_summary("best_overall", "", best_row)
        for hw in sorted(feasible_ranked["hardware"].unique()):
            g = feasible_ranked[feasible_ranked["hardware"] == hw]
            if len(g):
                add_summary("best_hardware", hw, g.iloc[0])
        for prec in sorted(feasible_ranked["precision"].unique()):
            g = feasible_ranked[feasible_ranked["precision"] == prec]
            if len(g):
                add_summary("best_precision", prec, g.iloc[0])
    else:
        closest = df.nsmallest(1, "constraint_violation").iloc[0]
        add_summary("no_feasible_closest", "", closest)

    summary_df = pd.DataFrame(summary_rows)

    print_console_tables(
        feasible_ranked=feasible_ranked,
        feasible_any=bool(feasible.any()),
        closest=closest_show,
        best_by_hardware=best_by_hardware,
        best_by_precision=best_by_precision,
        skip=args.no_console_tables,
    )

    if not args.no_plot:
        main_path = out_dir / "optimization_tradeoff.png"
        if write_tradeoff_plot(
            main_path,
            df_pareto,
            title="Throughput vs accuracy — all hardware (marker area ∝ model size MB)",
        ):
            print(f"Wrote {_path_for_display(main_path)}")
        for hw in sorted(df_pareto["hardware"].astype(str).unique()):
            sub = df_pareto[df_pareto["hardware"] == hw].copy()
            if len(sub) == 0:
                continue
            sub["pareto_optimal"] = pareto_optimal_mask(sub)
            hpath = out_dir / f"optimization_tradeoff_{hw}.png"
            if write_tradeoff_plot(
                hpath,
                sub,
                title=f"Throughput vs accuracy — {hw} (marker area ∝ model size MB)",
            ):
                print(f"Wrote {_path_for_display(hpath)}")

    pareto_sorted = pareto_only.sort_values(["map50", "fps", "size_mb"], ascending=[False, False, True])

    if not args.no_report:
        md_path = out_dir / "optimization_report.md"
        write_markdown_report(
            md_path,
            csv_path=args.csv.resolve(),
            args=args,
            feasible_count=int(feasible.sum()),
            total=len(df),
            feasible_ranked=feasible_ranked,
            pareto_sorted=pareto_sorted,
            summary_df=summary_df,
            best_by_hardware=best_by_hardware,
            best_by_precision=best_by_precision,
            closest=closest_show,
        )
        print(f"Wrote {_path_for_display(md_path)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
