"""
EfficientPlace — Greedy sequential macro placer based on the wire-mask
inference policy of Geng et al., ICML 2024
("Reinforcement Learning within Tree Search for Fast Macro Placement",
https://github.com/MIRALab-USTC/AI4EDA-EfficientPlace).

Method summary
--------------
EfficientPlace casts macro placement as a sequential decision problem on a
discretized canvas grid. For each macro in a fixed ordering, the agent
observes three channels (matching `src/environment.py` in the upstream
repo):

  * canvas        — which cells are occupied by previously-placed macros
  * position_mask — cells where placing the current macro would overlap
  * wire_mask     — marginal HPWL contribution at each candidate cell

The paper trains a CNN actor-critic via PPO inside a tree search over
partial solutions ("solution pool" with frontier expansion). At inference
time the greedy policy `act_greedy` in `src/agent.py` simply picks a
random cell among those with minimum wire_mask, restricted to feasible
positions. That greedy step is the EfficientPlace component reproduced
here — RL training is impractical inside the competition's 1-hour budget,
but the greedy wire-mask policy is itself a strong analytical placer.

What this placer does
---------------------
1. Rank hard macros using the orderings from `src/place_db.py::rank_macros`:
   - rank_mode 1: area descending, ties broken by net-area-sum descending
   - rank_mode 2: net-area-sum descending
   - plus a degree-based ordering (third "root" of the tree search).
2. For each ordering, run the EfficientPlace greedy inference step:
   - Compute position_mask vs. already-placed hard macros.
   - Compute wire_mask = sum over the macro's nets of marginal HPWL
     contribution if that pin sits at each grid cell.
   - Pick the feasible cell with minimum wire_mask.
3. Keep the placement with the lowest HPWL across orderings ("solution
   pool" reduced to picking the best frontier).

Soft macros (standard-cell clusters) stay at their initial positions and
act as fixed anchors for HPWL, matching the EfficientPlace environment
where standard cells form a background grid.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark


class EfficientPlacer:
    """Greedy wire-mask sequential placer (EfficientPlace inference policy)."""

    def __init__(
        self,
        grid_size: int = 128,
        num_starts: int = 3,
        seed: int = 0,
        overlap_gap: float = 1e-3,
    ) -> None:
        self.grid_size = grid_size
        self.num_starts = num_starts
        self.seed = seed
        self.overlap_gap = overlap_gap

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        n_hard = benchmark.num_hard_macros
        positions = benchmark.macro_positions.clone()
        if n_hard == 0:
            return positions

        orderings = self._build_orderings(benchmark, n_hard)
        net_meta = self._build_net_metadata(benchmark)

        best_positions = positions[:n_hard].numpy().astype(np.float64)
        best_cost = math.inf
        for order in orderings[: self.num_starts]:
            cand_positions, cand_cost = self._run_greedy(
                benchmark, order, net_meta, rng
            )
            if cand_cost < best_cost:
                best_cost = cand_cost
                best_positions = cand_positions

        positions[:n_hard] = torch.from_numpy(best_positions).float()
        return positions

    # ---------------------------------------------------------------- orderings

    def _build_orderings(self, benchmark: Benchmark, n_hard: int) -> List[List[int]]:
        sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        fixed = benchmark.macro_fixed[:n_hard].numpy()
        areas = sizes[:, 0] * sizes[:, 1]

        net_area_sum = np.zeros(n_hard, dtype=np.float64)
        degree = np.zeros(n_hard, dtype=np.int64)
        for net in benchmark.net_nodes:
            nodes = net.numpy()
            hards = nodes[nodes < n_hard]
            if hards.size <= 1:
                continue
            tot = float(areas[hards].sum())
            for nd in hards:
                net_area_sum[nd] += tot
                degree[nd] += 1

        movable = [i for i in range(n_hard) if not fixed[i]]
        # EfficientPlace rank_mode 1: -(area, area_sum)
        order_area = sorted(movable, key=lambda i: (-areas[i], -net_area_sum[i]))
        # EfficientPlace rank_mode 2: -area_sum
        order_areasum = sorted(movable, key=lambda i: -net_area_sum[i])
        # Connectivity-first ordering (additional tree-search root)
        order_degree = sorted(movable, key=lambda i: (-degree[i], -areas[i]))
        return [order_area, order_areasum, order_degree]

    # -------------------------------------------------------------- net metadata

    def _build_net_metadata(
        self, benchmark: Benchmark
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Per-net (owner_indices [E], pin_offsets [E, 2]) in microns.

        Owner index conventions match Benchmark.net_pin_nodes:
          - [0, num_hard)                            -> hard macro
          - [num_hard, num_macros)                   -> soft macro
          - [num_macros, num_macros + num_ports)     -> I/O port
        """
        nets: List[Tuple[np.ndarray, np.ndarray]] = []
        if len(benchmark.net_pin_nodes) > 0:
            n_hard = benchmark.num_hard_macros
            pin_offsets_np: List[np.ndarray] = [
                po.numpy().astype(np.float64) if po.numel() > 0 else np.zeros((0, 2))
                for po in benchmark.macro_pin_offsets
            ]
            for net_pn in benchmark.net_pin_nodes:
                if net_pn.numel() == 0:
                    nets.append(
                        (np.zeros(0, dtype=np.int64), np.zeros((0, 2), dtype=np.float64))
                    )
                    continue
                owners = net_pn[:, 0].numpy().astype(np.int64)
                pin_idx = net_pn[:, 1].numpy().astype(np.int64)
                offsets = np.zeros((owners.shape[0], 2), dtype=np.float64)
                for e in range(owners.shape[0]):
                    o = int(owners[e])
                    p = int(pin_idx[e])
                    if o < n_hard and p < pin_offsets_np[o].shape[0]:
                        offsets[e] = pin_offsets_np[o][p]
                nets.append((owners, offsets))
        else:
            for net in benchmark.net_nodes:
                owners = net.numpy().astype(np.int64)
                offsets = np.zeros((owners.shape[0], 2), dtype=np.float64)
                nets.append((owners, offsets))
        return nets

    # ------------------------------------------------------------------- greedy

    def _run_greedy(
        self,
        benchmark: Benchmark,
        order: List[int],
        net_meta: List[Tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float]:
        n_hard = benchmark.num_hard_macros
        n_total = benchmark.num_macros
        num_ports = int(benchmark.port_positions.shape[0])
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        G = self.grid_size

        sizes_np = benchmark.macro_sizes.numpy().astype(np.float64)
        positions_np = benchmark.macro_positions.numpy().astype(np.float64).copy()
        port_positions_np = (
            benchmark.port_positions.numpy().astype(np.float64)
            if num_ports > 0
            else np.zeros((0, 2), dtype=np.float64)
        )
        fixed_np = benchmark.macro_fixed.numpy()

        # Grid cell centers in microns. We place macro centers at grid cell centers
        # then clip to canvas — the discretization mirrors EfficientPlace's grid.
        gx_centers = (np.arange(G) + 0.5) * (cw / G)
        gy_centers = (np.arange(G) + 0.5) * (ch / G)

        # Anchors: soft macros, ports, and fixed hard macros contribute to nets
        # from the start. We grow this set as we place hard macros.
        anchored = np.zeros(n_total + num_ports, dtype=bool)
        anchored[n_hard:n_total] = True  # soft macros
        anchored[n_total:] = True        # ports
        for i in range(n_hard):
            if fixed_np[i]:
                anchored[i] = True

        n_nets = benchmark.num_nets
        net_min = np.full((n_nets, 2), np.inf, dtype=np.float64)
        net_max = np.full((n_nets, 2), -np.inf, dtype=np.float64)
        net_count = np.zeros(n_nets, dtype=np.int64)

        # owner -> nets it appears on
        owner_to_nets: List[List[int]] = [[] for _ in range(n_total + num_ports)]
        # (owner, net) -> list of pin offsets (most pairs have a single pin)
        owner_pins_on_net: Dict[Tuple[int, int], List[np.ndarray]] = {}
        for net_id, (owners, offsets) in enumerate(net_meta):
            for e in range(owners.shape[0]):
                o = int(owners[e])
                if 0 <= o < n_total + num_ports:
                    owner_to_nets[o].append(net_id)
                    owner_pins_on_net.setdefault((o, net_id), []).append(offsets[e])

        def owner_center(o: int) -> np.ndarray:
            return positions_np[o] if o < n_total else port_positions_np[o - n_total]

        # Seed net bboxes with all pre-anchored pins
        for net_id, (owners, offsets) in enumerate(net_meta):
            for e in range(owners.shape[0]):
                o = int(owners[e])
                if not anchored[o]:
                    continue
                p = owner_center(o) + offsets[e]
                if net_count[net_id] == 0:
                    net_min[net_id] = p
                    net_max[net_id] = p
                else:
                    if p[0] < net_min[net_id, 0]:
                        net_min[net_id, 0] = p[0]
                    if p[1] < net_min[net_id, 1]:
                        net_min[net_id, 1] = p[1]
                    if p[0] > net_max[net_id, 0]:
                        net_max[net_id, 0] = p[0]
                    if p[1] > net_max[net_id, 1]:
                        net_max[net_id, 1] = p[1]
                net_count[net_id] += 1

        gap = self.overlap_gap
        placed_hards: List[int] = [i for i in range(n_hard) if anchored[i]]

        for macro_idx in order:
            mw = sizes_np[macro_idx, 0]
            mh = sizes_np[macro_idx, 1]
            half_w = mw / 2.0
            half_h = mh / 2.0

            # Canvas-feasible cells: macro fully inside canvas when centered here.
            valid_x = (gx_centers >= half_w - 1e-9) & (
                gx_centers <= cw - half_w + 1e-9
            )
            valid_y = (gy_centers >= half_h - 1e-9) & (
                gy_centers <= ch - half_h + 1e-9
            )
            canvas_ok = valid_x[:, None] & valid_y[None, :]  # [G, G]

            # Position mask: overlap with previously-placed hard macros.
            if placed_hards:
                ph_pos = positions_np[placed_hards]   # [P, 2]
                ph_size = sizes_np[placed_hards]      # [P, 2]
                tx = (mw + ph_size[:, 0]) / 2.0 + gap  # [P]
                ty = (mh + ph_size[:, 1]) / 2.0 + gap  # [P]
                dx = np.abs(gx_centers[:, None] - ph_pos[None, :, 0])  # [G, P]
                dy = np.abs(gy_centers[:, None] - ph_pos[None, :, 1])  # [G, P]
                x_block = (dx < tx[None, :]).astype(np.int32)  # [G, P]
                y_block = (dy < ty[None, :]).astype(np.int32)  # [G, P]
                # pos_mask[gx, gy] = any p: x_block[gx, p] AND y_block[gy, p]
                # which equals (x_block @ y_block.T) > 0.
                pos_mask = (x_block @ y_block.T) > 0  # [G, G]
            else:
                pos_mask = np.zeros((G, G), dtype=bool)

            invalid = pos_mask | (~canvas_ok)

            # Wire mask: marginal HPWL contribution at each grid cell.
            wire_mask = np.zeros((G, G), dtype=np.float64)
            for net_id in owner_to_nets[macro_idx]:
                if net_count[net_id] == 0:
                    continue  # macro is the first pin on this net -> 0 marginal
                bmin0, bmin1 = net_min[net_id, 0], net_min[net_id, 1]
                bmax0, bmax1 = net_max[net_id, 0], net_max[net_id, 1]
                for off in owner_pins_on_net[(macro_idx, net_id)]:
                    px = gx_centers + off[0]
                    py = gy_centers + off[1]
                    dx = np.maximum(px - bmax0, 0.0) + np.maximum(bmin0 - px, 0.0)
                    dy = np.maximum(py - bmax1, 0.0) + np.maximum(bmin1 - py, 0.0)
                    wire_mask += dx[:, None] + dy[None, :]

            wm = np.where(invalid, np.inf, wire_mask)
            if not np.isfinite(wm).any():
                # Hard fallback: any non-overlapping canvas cell, else clamp to nearest
                # in-canvas point.
                fallback = (~pos_mask) & canvas_ok
                if fallback.any():
                    flat = np.flatnonzero(fallback.ravel())
                    pick = int(rng.choice(flat))
                    gx, gy = divmod(pick, G)
                    cx = float(gx_centers[gx])
                    cy = float(gy_centers[gy])
                else:
                    cx = float(np.clip(positions_np[macro_idx, 0], half_w, cw - half_w))
                    cy = float(np.clip(positions_np[macro_idx, 1], half_h, ch - half_h))
            else:
                min_val = wm.min()
                # Match EfficientPlace's act_greedy: random tie-break among argmin.
                cands = np.flatnonzero(wm.ravel() == min_val)
                pick = int(rng.choice(cands))
                gx, gy = divmod(pick, G)
                cx = float(np.clip(gx_centers[gx], half_w, cw - half_w))
                cy = float(np.clip(gy_centers[gy], half_h, ch - half_h))

            positions_np[macro_idx, 0] = cx
            positions_np[macro_idx, 1] = cy
            anchored[macro_idx] = True
            placed_hards.append(macro_idx)

            # Update net bboxes with this macro's pins.
            for net_id in owner_to_nets[macro_idx]:
                for off in owner_pins_on_net[(macro_idx, net_id)]:
                    p0 = cx + off[0]
                    p1 = cy + off[1]
                    if net_count[net_id] == 0:
                        net_min[net_id, 0] = p0
                        net_min[net_id, 1] = p1
                        net_max[net_id, 0] = p0
                        net_max[net_id, 1] = p1
                    else:
                        if p0 < net_min[net_id, 0]:
                            net_min[net_id, 0] = p0
                        if p1 < net_min[net_id, 1]:
                            net_min[net_id, 1] = p1
                        if p0 > net_max[net_id, 0]:
                            net_max[net_id, 0] = p0
                        if p1 > net_max[net_id, 1]:
                            net_max[net_id, 1] = p1
                    net_count[net_id] += 1

        # Total HPWL over multi-pin nets — used to pick the best ordering.
        multi = net_count > 1
        hpwl = float(
            (net_max[multi, 0] - net_min[multi, 0]).sum()
            + (net_max[multi, 1] - net_min[multi, 1]).sum()
        )
        return positions_np[:n_hard].copy(), hpwl
