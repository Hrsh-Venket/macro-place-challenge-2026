"""
Coordinate-descent polish on the exact (non-smoothed) HPWL with hard-macro
zero-overlap constraint.

This is the cheap-but-essential last stage of an Xplace + polish pipeline.
Xplace optimizes a smoothed log-sum-exp HPWL + electrostatic density; the
discrete proxy cost the competition scores is non-smooth, and most of the
last 5–15% of quality lives in the gap between the two. CD here picks the
best legal lattice position per macro by exact incremental HPWL.

Algorithm (per macro, in random sweep order):
  1. Build the union of nets touching this macro.
  2. For each candidate position on a coarse grid around the current cx/cy,
     compute the marginal HPWL contribution from those nets (others are
     unchanged), reject candidates that create hard-macro overlap, and pick
     the minimum.  Move only if it improves.
  3. Repeat sweeps until improvement stalls or the wall-clock budget is up.

Notes
-----
* We do NOT include density/congestion in CD — they're hard to update
  incrementally and Xplace's global solve already handled them. CD's job is
  to close the HPWL surrogate gap. (The leaderboard's KLA MACH / JonaU /
  thinkorplace teams all run CD on the proxy with HPWL as the dominant
  term and rely on the analytical placer for density.)
* Overlap check is vectorised against all other hard macros in NumPy —
  O(N) per candidate position; fine for N ≤ 600.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark


def _build_net_index(benchmark: Benchmark) -> Tuple[
    List[np.ndarray],   # owner per pin, length = num_nets
    List[np.ndarray],   # pin offsets (x,y), length = num_nets
    List[List[int]],    # owner_to_nets[owner_idx] = [net_ids...]
    List[Dict[int, List[int]]],  # owner_to_pin_idxs_in_net[owner_idx][net_id] = [k1,k2,...]
]:
    """Flatten net connectivity into per-net arrays plus reverse indices.

    For each net we store the owners and (x,y) offsets of every pin endpoint
    (rows = pins), in the same order. ``owner_to_pin_idxs_in_net[o][nid]``
    lists the pin row-indices for owner ``o`` on net ``nid``, which is what
    we need to subtract/re-add a macro's pins during an incremental update.
    """
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.num_macros
    n_ports = int(benchmark.port_positions.shape[0])

    have_pin_level = len(benchmark.net_pin_nodes) == benchmark.num_nets
    have_node_level = len(benchmark.net_nodes) == benchmark.num_nets
    if not (have_pin_level or have_node_level):
        # No connectivity → CD has nothing to optimise. Caller should have
        # loaded via load_benchmark_from_dir(); fail loudly.
        raise RuntimeError(
            f"Benchmark {benchmark.name!r} has empty net data; load via "
            "macro_place.loader.load_benchmark_from_dir() first."
        )
    use_pin_level = have_pin_level
    pin_offsets_np = (
        [po.cpu().numpy().astype(np.float64) if po.numel() > 0 else None
         for po in benchmark.macro_pin_offsets]
    )

    net_owners: List[np.ndarray] = []
    net_offsets: List[np.ndarray] = []
    owner_to_nets: List[List[int]] = [[] for _ in range(n_total + n_ports)]
    owner_pin_rows: List[Dict[int, List[int]]] = [dict() for _ in range(n_total + n_ports)]

    for net_id in range(benchmark.num_nets):
        if use_pin_level:
            arr = benchmark.net_pin_nodes[net_id].cpu().numpy().astype(np.int64)
            if arr.shape[0] == 0:
                net_owners.append(np.zeros(0, dtype=np.int64))
                net_offsets.append(np.zeros((0, 2), dtype=np.float64))
                continue
            owners = arr[:, 0]
            pins = arr[:, 1]
            offsets = np.zeros((owners.shape[0], 2), dtype=np.float64)
            for k in range(owners.shape[0]):
                o = int(owners[k]); p = int(pins[k])
                if o < n_hard and pin_offsets_np[o] is not None and p < pin_offsets_np[o].shape[0]:
                    offsets[k] = pin_offsets_np[o][p]
                # soft macros & ports: offset (0,0)
        else:
            nodes = benchmark.net_nodes[net_id].cpu().numpy().astype(np.int64)
            owners = nodes
            offsets = np.zeros((owners.shape[0], 2), dtype=np.float64)

        net_owners.append(owners)
        net_offsets.append(offsets)
        for k, o in enumerate(owners):
            o = int(o)
            if 0 <= o < n_total + n_ports:
                if net_id not in owner_pin_rows[o]:
                    owner_to_nets[o].append(net_id)
                    owner_pin_rows[o][net_id] = []
                owner_pin_rows[o][net_id].append(k)

    return net_owners, net_offsets, owner_to_nets, owner_pin_rows


def _all_pin_xy(
    positions: np.ndarray,
    port_positions: np.ndarray,
    n_total: int,
    owners: np.ndarray,
    offsets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pin (x,y) for one net given current macro+port positions."""
    px = np.empty(owners.shape[0], dtype=np.float64)
    py = np.empty(owners.shape[0], dtype=np.float64)
    for k in range(owners.shape[0]):
        o = int(owners[k])
        if o < n_total:
            cx, cy = positions[o, 0], positions[o, 1]
        else:
            cx, cy = port_positions[o - n_total, 0], port_positions[o - n_total, 1]
        px[k] = cx + offsets[k, 0]
        py[k] = cy + offsets[k, 1]
    return px, py


def cd_polish(
    benchmark: Benchmark,
    positions_in: torch.Tensor,
    *,
    grid_step: float = 0.5,                # candidate spacing in microns
    radius_factor: float = 2.0,            # search radius = factor * macro_diag
    num_sweeps: int = 4,
    time_budget_s: float = 600.0,
    overlap_gap: float = 1e-3,
    rng_seed: int = 0,
    verbose: bool = True,
) -> torch.Tensor:
    """Run coordinate-descent HPWL polish on hard macros.

    Parameters
    ----------
    positions_in : [num_macros, 2] tensor — output of Xplace.
    grid_step    : candidate-position spacing in microns.
    radius_factor: candidates lie within radius_factor*macro_diag of current.
    num_sweeps   : maximum CD passes over all movable hard macros.
    time_budget_s: wall-clock cap.
    """
    rng = np.random.default_rng(rng_seed)
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.num_macros
    n_ports = int(benchmark.port_positions.shape[0])

    positions = positions_in.cpu().numpy().astype(np.float64).copy()
    port_positions = (
        benchmark.port_positions.cpu().numpy().astype(np.float64)
        if n_ports > 0 else np.zeros((0, 2), dtype=np.float64)
    )
    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed.cpu().numpy()
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    net_owners, net_offsets, owner_to_nets, owner_pin_rows = _build_net_index(benchmark)

    # ── per-net (px, py) cache; recomputed lazily on macro move ──────────
    n_nets = benchmark.num_nets
    net_px: List[np.ndarray] = [None] * n_nets
    net_py: List[np.ndarray] = [None] * n_nets
    for nid in range(n_nets):
        owners = net_owners[nid]
        if owners.size == 0:
            net_px[nid] = np.zeros(0); net_py[nid] = np.zeros(0); continue
        net_px[nid], net_py[nid] = _all_pin_xy(
            positions, port_positions, n_total, owners, net_offsets[nid]
        )

    def net_hpwl(nid: int) -> float:
        if net_px[nid].size < 2:
            return 0.0
        return float((net_px[nid].max() - net_px[nid].min())
                     + (net_py[nid].max() - net_py[nid].min()))

    # Hard-macro overlap check vs all *other* hard macros.
    # Cached arrays for vectorised distance.
    hard_sizes = sizes[:n_hard]
    hard_pos = positions[:n_hard].copy()

    def overlap_free(macro_idx: int, cx: float, cy: float) -> bool:
        if cx < hard_sizes[macro_idx, 0] / 2 - 1e-9: return False
        if cy < hard_sizes[macro_idx, 1] / 2 - 1e-9: return False
        if cx > cw - hard_sizes[macro_idx, 0] / 2 + 1e-9: return False
        if cy > ch - hard_sizes[macro_idx, 1] / 2 + 1e-9: return False
        dx = np.abs(hard_pos[:, 0] - cx)
        dy = np.abs(hard_pos[:, 1] - cy)
        tx = (hard_sizes[:, 0] + hard_sizes[macro_idx, 0]) / 2.0 + overlap_gap
        ty = (hard_sizes[:, 1] + hard_sizes[macro_idx, 1]) / 2.0 + overlap_gap
        bad = (dx < tx) & (dy < ty)
        bad[macro_idx] = False
        return not bool(bad.any())

    movable = [i for i in range(n_hard) if not bool(fixed[i])]
    if not movable:
        return positions_in.clone()

    t0 = time.time()
    total_improve = 0.0
    for sweep in range(num_sweeps):
        if time.time() - t0 > time_budget_s:
            break
        order = rng.permutation(movable)
        sweep_improve = 0.0
        for macro_idx in order:
            if time.time() - t0 > time_budget_s:
                break
            macro_idx = int(macro_idx)
            mw, mh = float(sizes[macro_idx, 0]), float(sizes[macro_idx, 1])
            half_w, half_h = mw / 2.0, mh / 2.0
            cx0, cy0 = float(positions[macro_idx, 0]), float(positions[macro_idx, 1])

            nets_touched = owner_to_nets[macro_idx]
            if not nets_touched:
                continue

            # Snapshot pre-move per-net pin arrays & HPWL contribution.
            base_hpwl = 0.0
            base_pin_arrays: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            base_others_bbox: Dict[int, Tuple[float, float, float, float]] = {}
            for nid in nets_touched:
                px = net_px[nid].copy()
                py = net_py[nid].copy()
                base_pin_arrays[nid] = (px, py)
                base_hpwl += net_hpwl(nid)
                # Compute bbox of pins NOT belonging to this macro for fast incremental scoring.
                rows = owner_pin_rows[macro_idx][nid]
                mask = np.ones(px.shape[0], dtype=bool)
                for r in rows:
                    mask[r] = False
                if mask.any():
                    base_others_bbox[nid] = (
                        float(px[mask].min()), float(px[mask].max()),
                        float(py[mask].min()), float(py[mask].max()),
                    )
                else:
                    base_others_bbox[nid] = None

            # Build candidate grid.
            macro_diag = math.hypot(mw, mh)
            radius = max(grid_step * 2, radius_factor * macro_diag)
            steps = max(2, int(radius / grid_step))
            xs = cx0 + grid_step * np.arange(-steps, steps + 1)
            ys = cy0 + grid_step * np.arange(-steps, steps + 1)
            # Always include the current position as the baseline.
            xs = np.clip(xs, half_w, cw - half_w)
            ys = np.clip(ys, half_h, ch - half_h)

            # Precompute per-net pin offsets *for this macro* so we can
            # generate moved-pin coordinates by addition.
            macro_offsets = {
                nid: net_offsets[nid][owner_pin_rows[macro_idx][nid], :]
                for nid in nets_touched
            }

            best_delta = 0.0
            best_xy: Tuple[float, float] | None = None
            for cx in xs:
                for cy in ys:
                    if cx == cx0 and cy == cy0:
                        continue
                    if not overlap_free(macro_idx, float(cx), float(cy)):
                        continue
                    delta = 0.0
                    for nid in nets_touched:
                        offs = macro_offsets[nid]
                        new_px = cx + offs[:, 0]
                        new_py = cy + offs[:, 1]
                        bb = base_others_bbox[nid]
                        if bb is None:
                            new_min_x, new_max_x = float(new_px.min()), float(new_px.max())
                            new_min_y, new_max_y = float(new_py.min()), float(new_py.max())
                        else:
                            ox_min, ox_max, oy_min, oy_max = bb
                            new_min_x = min(ox_min, float(new_px.min()))
                            new_max_x = max(ox_max, float(new_px.max()))
                            new_min_y = min(oy_min, float(new_py.min()))
                            new_max_y = max(oy_max, float(new_py.max()))
                        new_h = (new_max_x - new_min_x) + (new_max_y - new_min_y)
                        # subtract the old contribution for this net
                        px_b, py_b = base_pin_arrays[nid]
                        old_h = (
                            (float(px_b.max()) - float(px_b.min()))
                            + (float(py_b.max()) - float(py_b.min()))
                        )
                        delta += new_h - old_h
                        if delta >= best_delta:
                            # Pruning: any positive partial sum can't beat best.
                            break
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_xy = (float(cx), float(cy))

            if best_xy is None:
                continue

            cx_new, cy_new = best_xy
            positions[macro_idx, 0] = cx_new
            positions[macro_idx, 1] = cy_new
            hard_pos[macro_idx, 0] = cx_new
            hard_pos[macro_idx, 1] = cy_new
            # Apply: update per-net pin arrays in-place.
            for nid in nets_touched:
                offs = macro_offsets[nid]
                rows = owner_pin_rows[macro_idx][nid]
                for j, r in enumerate(rows):
                    net_px[nid][r] = cx_new + offs[j, 0]
                    net_py[nid][r] = cy_new + offs[j, 1]
            sweep_improve += -best_delta  # delta is negative when improving

        total_improve += sweep_improve
        if verbose:
            print(f"  [cd-polish] sweep {sweep+1}/{num_sweeps} improvement={sweep_improve:.3f}")
        if sweep_improve < 1e-6:
            break

    if verbose:
        print(f"  [cd-polish] total HPWL improvement={total_improve:.3f} "
              f"in {time.time()-t0:.1f}s")
    return torch.from_numpy(positions).float()
