"""
XplacePlacer — Xplace global placer + exact-proxy CD polish.

Pipeline per benchmark:
  1. Write the Benchmark to Bookshelf (UCLA ICCAD04) in a temp dir.
  2. Invoke Xplace as a subprocess (see ``--xplace_root``); Xplace runs
     ePlace-style global placement with electrostatic density and WA-HPWL
     and writes a placed .pl back.
  3. Parse the placed .pl, mapping Bookshelf node names → Benchmark macro
     indices.
  4. Run coordinate-descent on the exact (non-smoothed) HPWL, restricted
     to hard movable macros and overlap-free moves.
  5. Optionally re-run a soft-macro force-directed pass for wirelength.

This file is also the submission entry-point: ``XplacePlacer`` is the
first class with a ``place`` method and is instantiated with no args by
the evaluator.

For build & run instructions, including the Dockerfile, see the
``README.md`` next to this file.

Two known biases of the CD stage:
  * CD optimises HPWL only.  Tighter clustering is good for wirelength
    but can degrade the density / congestion components of the proxy.
    Bound the search via ``cd_radius_factor`` so CD only does local
    refinement on top of Xplace's globally-balanced layout.
  * CD does not repair pre-existing overlaps in its input; it only
    rejects moves that create new ones.  Xplace's output is legal, so
    this is only a concern when ``skip_xplace=True`` (a debugging mode).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from macro_place.benchmark import Benchmark

# Allow sibling-module imports when loaded by the evaluator's
# ``spec_from_file_location`` (which doesn't put this dir on sys.path).
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from bookshelf_io import write_bookshelf, read_placed_pl  # noqa: E402
from cd_polish import cd_polish                            # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Xplace invocation
# ──────────────────────────────────────────────────────────────────────────────

# Common locations the Dockerfile lays Xplace into; environment overrides win.
_XPLACE_DEFAULT_ROOTS = ["/opt/Xplace", "/workspace/Xplace", str(Path.home() / "Xplace")]


def _find_xplace_root() -> Optional[Path]:
    env = os.environ.get("XPLACE_ROOT")
    if env and Path(env).exists():
        return Path(env)
    for c in _XPLACE_DEFAULT_ROOTS:
        if Path(c).exists():
            return Path(c)
    return None


def _run_xplace(
    xplace_root: Path,
    aux_path: Path,
    out_dir: Path,
    target_density: float,
    gpu: bool,
    extra_args: Optional[list[str]] = None,
    timeout_s: float = 1800.0,
) -> Path:
    """Invoke Xplace on a Bookshelf design and return the path to its
    placed .pl. The exact flag set has shifted across Xplace versions —
    we pass the common subset and let ``extra_args`` carry per-version
    overrides.

    Expected Xplace produces an output .pl named ``<design>.gp.pl``
    (or similar) in ``out_dir``.  We tolerate any single ``*.pl`` that
    appears.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    design = aux_path.stem  # design name
    cmd = [
        sys.executable, "main.py",
        "--dataset_format", "bookshelf",
        "--aux", str(aux_path),
        "--output_dir", str(out_dir),
        "--target_density", f"{target_density:.4f}",
        "--use_cell_inflate", "false",
        "--deterministic", "true",
    ]
    if gpu:
        cmd += ["--device", "cuda"]
    else:
        cmd += ["--device", "cpu"]
    if extra_args:
        cmd += list(extra_args)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(xplace_root) + os.pathsep + env.get("PYTHONPATH", "")

    print(f"  [xplace] $ {' '.join(cmd)} (cwd={xplace_root})")
    proc = subprocess.run(
        cmd, cwd=str(xplace_root), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=timeout_s,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"Xplace failed with return code {proc.returncode}")
    # Search recursively in case Xplace nested an output dir.
    candidates = sorted(out_dir.rglob("*.pl"))
    if not candidates:
        print(proc.stdout)
        raise RuntimeError(f"Xplace produced no .pl files in {out_dir}")
    # Prefer the largest .pl (the final post-legalization placement).
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


# ──────────────────────────────────────────────────────────────────────────────
# Soft-macro repositioning (lightweight force-directed)
# ──────────────────────────────────────────────────────────────────────────────


def _force_directed_soft(
    benchmark: Benchmark,
    positions: torch.Tensor,
    iters: int = 60,
    lr: float = 0.5,
) -> torch.Tensor:
    """Pull each soft macro toward the centroid of its net neighbours.

    Soft macros may overlap, so we ignore density. This is a much faster
    drop-in for ``plc.optimize_stdcells`` (which is several minutes of
    pure Python per call).
    """
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.num_macros
    n_ports = int(benchmark.port_positions.shape[0])
    if n_total == n_hard:
        return positions

    pos = positions.clone().double()
    port_pos = benchmark.port_positions.double() if n_ports > 0 else torch.zeros(0, 2, dtype=torch.float64)
    cw = float(benchmark.canvas_width); ch = float(benchmark.canvas_height)

    use_pin_level = len(benchmark.net_pin_nodes) == benchmark.num_nets
    # Flatten nets to pin-row arrays.
    rows: list[tuple[torch.Tensor, int]] = []
    for nid in range(benchmark.num_nets):
        if use_pin_level:
            arr = benchmark.net_pin_nodes[nid]
            if arr.numel() == 0: continue
            owners = arr[:, 0].long()
        else:
            owners = benchmark.net_nodes[nid].long()
            if owners.numel() == 0: continue
        if owners.numel() < 2: continue
        rows.append((owners, nid))

    soft_lo, soft_hi = n_hard, n_total

    for _ in range(iters):
        # Net centroids
        force = torch.zeros_like(pos)
        weight = torch.zeros(n_total, dtype=torch.float64)
        for owners, _nid in rows:
            o = owners
            xs = torch.empty(o.numel(), dtype=torch.float64)
            ys = torch.empty(o.numel(), dtype=torch.float64)
            for k in range(o.numel()):
                oi = int(o[k].item())
                if oi < n_total:
                    xs[k] = pos[oi, 0]; ys[k] = pos[oi, 1]
                else:
                    xs[k] = port_pos[oi - n_total, 0]
                    ys[k] = port_pos[oi - n_total, 1]
            cx, cy = float(xs.mean()), float(ys.mean())
            for k in range(o.numel()):
                oi = int(o[k].item())
                if soft_lo <= oi < soft_hi:
                    force[oi, 0] += (cx - pos[oi, 0])
                    force[oi, 1] += (cy - pos[oi, 1])
                    weight[oi] += 1.0
        nz = weight > 0
        # Only step soft macros that are connected to ≥1 net.
        step = torch.zeros_like(pos)
        step[nz] = force[nz] / weight[nz].unsqueeze(-1)
        pos[soft_lo:soft_hi] += lr * step[soft_lo:soft_hi]
        pos[soft_lo:soft_hi, 0].clamp_(0, cw)
        pos[soft_lo:soft_hi, 1].clamp_(0, ch)

    return pos.float()


# ──────────────────────────────────────────────────────────────────────────────
# Top-level placer
# ──────────────────────────────────────────────────────────────────────────────


class XplacePlacer:
    """Xplace global placement + exact-proxy CD polish.

    Hyperparameters
    ---------------
    target_density
        Xplace target utilisation. Lower → more spreading; higher → tighter
        layouts. 0.55–0.75 is the usual sweet spot for macro-heavy designs.
    cd_grid_step, cd_radius_factor, cd_sweeps
        See ``cd_polish.cd_polish``.
    xplace_extra_args
        Forwarded to Xplace main.py. Used for hyper-parameter sweeps.
    use_gpu
        Pass ``--device cuda`` to Xplace.
    skip_xplace
        Skip the global placement and run CD on the benchmark's initial
        positions only.  Diagnostic.
    fd_soft_iters
        Iterations of soft-macro FD. 0 disables and leaves soft macros
        where Xplace put them.
    """

    def __init__(
        self,
        *,
        target_density: float = 0.65,
        cd_grid_step: float = 0.5,
        cd_radius_factor: float = 2.0,
        cd_sweeps: int = 4,
        cd_time_budget_s: float = 900.0,
        xplace_extra_args: tuple[str, ...] = (),
        use_gpu: bool = True,
        skip_xplace: bool = False,
        fd_soft_iters: int = 0,
        overlap_gap: float = 1e-3,
        verbose: bool = True,
    ) -> None:
        self.target_density = target_density
        self.cd_grid_step = cd_grid_step
        self.cd_radius_factor = cd_radius_factor
        self.cd_sweeps = cd_sweeps
        self.cd_time_budget_s = cd_time_budget_s
        self.xplace_extra_args = list(xplace_extra_args)
        self.use_gpu = use_gpu
        self.skip_xplace = skip_xplace
        self.fd_soft_iters = fd_soft_iters
        self.overlap_gap = overlap_gap
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        positions = benchmark.macro_positions.clone()

        # ── 1–3. Xplace global placement ────────────────────────────────
        if not self.skip_xplace:
            xplace_root = _find_xplace_root()
            if xplace_root is None:
                raise RuntimeError(
                    "Xplace not found. Set XPLACE_ROOT or install Xplace at "
                    "/opt/Xplace (see Dockerfile)."
                )
            with tempfile.TemporaryDirectory(prefix=f"xplace_{benchmark.name}_") as td:
                td_p = Path(td)
                in_dir = td_p / "in"; out_dir = td_p / "out"
                name_map, _port_map = write_bookshelf(
                    benchmark, in_dir, design_name=benchmark.name
                )
                if self.verbose:
                    print(f"  [xplace] wrote Bookshelf to {in_dir}")
                pl_path = _run_xplace(
                    xplace_root,
                    in_dir / f"{benchmark.name}.aux",
                    out_dir,
                    target_density=self.target_density,
                    gpu=self.use_gpu,
                    extra_args=self.xplace_extra_args,
                )
                name_to_idx = {v: k for k, v in name_map.items()}
                positions = read_placed_pl(pl_path, name_to_idx, benchmark)
                if self.verbose:
                    print(f"  [xplace] parsed {pl_path.name} ({time.time()-t0:.1f}s elapsed)")

        # ── 4. CD polish on exact HPWL ──────────────────────────────────
        positions = cd_polish(
            benchmark, positions,
            grid_step=self.cd_grid_step,
            radius_factor=self.cd_radius_factor,
            num_sweeps=self.cd_sweeps,
            time_budget_s=self.cd_time_budget_s,
            overlap_gap=self.overlap_gap,
            verbose=self.verbose,
        )

        # ── 5. Soft-macro pass (optional) ───────────────────────────────
        if self.fd_soft_iters > 0:
            positions = _force_directed_soft(
                benchmark, positions, iters=self.fd_soft_iters
            )

        if self.verbose:
            print(f"  [xplace_placer] total {time.time()-t0:.1f}s")
        return positions
