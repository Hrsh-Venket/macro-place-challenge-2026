"""
Bookshelf (UCLA ICCAD04) I/O for the XplacePlacer wrapper.

Writes a Benchmark to Bookshelf .aux/.nodes/.nets/.pl/.scl/.wts files in a
temp directory so Xplace can ingest it, and parses Xplace's output .pl
back to a {macro_name: (cx, cy)} dict.

Conventions:
  - Hard macros (movable) and soft macros are emitted as normal nodes.
  - Fixed macros, ports, and explicitly-anchored macros are emitted with
    a `terminal` suffix and `/FIXED` in the .pl.
  - All coordinates in the Bookshelf files use micron units identical to
    Benchmark.canvas_*, so Xplace's output can be parsed back directly.
  - Names are sanitized to be alphanumeric+underscore (Bookshelf parsers
    are picky about colons / slashes that appear in protobuf names).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from macro_place.benchmark import Benchmark


_BAD_NAME_RE = re.compile(r"[^A-Za-z0-9_]")


def sanitize(name: str, idx: int, prefix: str) -> str:
    """Sanitize a node/net name. Collisions resolved by `_idx` suffix."""
    base = _BAD_NAME_RE.sub("_", name) if name else f"{prefix}{idx}"
    return f"{base}_{idx}"


# ──────────────────────────────────────────────────────────── Bookshelf writer


def write_bookshelf(
    benchmark: Benchmark,
    out_dir: str | Path,
    design_name: str = "design",
    row_height_microns: float = 0.2,
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Write Benchmark to Bookshelf files in *out_dir*.

    Returns
    -------
    macro_name_map :  {macro_index_in_benchmark -> bookshelf_node_name}
    port_name_map  :  {port_index -> bookshelf_terminal_name}
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = out / design_name

    sizes = benchmark.macro_sizes.cpu().numpy()
    positions = benchmark.macro_positions.cpu().numpy()
    fixed = benchmark.macro_fixed.cpu().numpy()
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.num_macros
    n_ports = int(benchmark.port_positions.shape[0])
    port_pos = benchmark.port_positions.cpu().numpy() if n_ports > 0 else None

    # ── node names ────────────────────────────────────────────────────────
    macro_name_map: Dict[int, str] = {}
    for i in range(n_total):
        raw = benchmark.macro_names[i] if i < len(benchmark.macro_names) else ""
        prefix = "M" if i < n_hard else "S"
        macro_name_map[i] = sanitize(raw, i, prefix)

    port_name_map: Dict[int, str] = {}
    for p in range(n_ports):
        port_name_map[p] = f"P_{p}"

    n_terminals = int(fixed.sum()) + n_ports
    n_nodes_total = n_total + n_ports

    # ── .nodes ────────────────────────────────────────────────────────────
    with open(f"{base}.nodes", "w") as f:
        f.write("UCLA nodes 1.0\n\n")
        f.write(f"NumNodes     :   {n_nodes_total}\n")
        f.write(f"NumTerminals :   {n_terminals}\n\n")
        for i in range(n_total):
            w, h = float(sizes[i, 0]), float(sizes[i, 1])
            line = f"\t{macro_name_map[i]}\t{w:.6f}\t{h:.6f}"
            if bool(fixed[i]):
                line += "\tterminal"
            f.write(line + "\n")
        for p in range(n_ports):
            # Ports have zero area in our format. Give a tiny size to keep
            # Xplace happy with degenerate terminals.
            f.write(f"\t{port_name_map[p]}\t0.001\t0.001\tterminal\n")

    # ── .pl ───────────────────────────────────────────────────────────────
    # Bookshelf positions are lower-left corners.
    with open(f"{base}.pl", "w") as f:
        f.write("UCLA pl 1.0\n\n")
        for i in range(n_total):
            cx, cy = float(positions[i, 0]), float(positions[i, 1])
            w, h = float(sizes[i, 0]), float(sizes[i, 1])
            x_ll, y_ll = cx - w / 2.0, cy - h / 2.0
            tag = " /FIXED" if bool(fixed[i]) else ""
            f.write(f"\t{macro_name_map[i]}\t{x_ll:.6f}\t{y_ll:.6f} : N{tag}\n")
        if n_ports > 0:
            for p in range(n_ports):
                px, py = float(port_pos[p, 0]), float(port_pos[p, 1])
                f.write(f"\t{port_name_map[p]}\t{px:.6f}\t{py:.6f} : N /FIXED\n")

    # ── .nets ─────────────────────────────────────────────────────────────
    # Pin-level if available, otherwise node-level.
    have_pin_level = len(benchmark.net_pin_nodes) == benchmark.num_nets
    have_node_level = len(benchmark.net_nodes) == benchmark.num_nets
    if not (have_pin_level or have_node_level):
        # Some cached .pt files (benchmarks/processed/public/*.pt) strip
        # connectivity; nets live only inside the PlacementCost object.
        # We cannot write a useful Bookshelf .nets without them.
        raise RuntimeError(
            f"Benchmark {benchmark.name!r} has no net connectivity "
            f"(net_nodes={len(benchmark.net_nodes)}, "
            f"net_pin_nodes={len(benchmark.net_pin_nodes)}, "
            f"num_nets={benchmark.num_nets}). "
            "Load via macro_place.loader.load_benchmark_from_dir() "
            "instead of Benchmark.load() so connectivity is populated."
        )
    use_pin_level = have_pin_level
    pin_offsets = (
        [po.cpu().numpy() if po.numel() > 0 else None for po in benchmark.macro_pin_offsets]
        if use_pin_level else []
    )

    with open(f"{base}.nets", "w") as f:
        f.write("UCLA nets 1.0\n\n")
        # First pass to count
        net_lines: List[Tuple[str, List[Tuple[str, float, float]]]] = []
        for net_id in range(benchmark.num_nets):
            pins: List[Tuple[str, float, float]] = []
            if use_pin_level:
                arr = benchmark.net_pin_nodes[net_id].cpu().numpy()
                for row in arr:
                    owner = int(row[0]); pin = int(row[1])
                    if owner < n_hard:
                        off = pin_offsets[owner]
                        ox, oy = (float(off[pin, 0]), float(off[pin, 1])) if off is not None and pin < off.shape[0] else (0.0, 0.0)
                        pins.append((macro_name_map[owner], ox, oy))
                    elif owner < n_total:
                        pins.append((macro_name_map[owner], 0.0, 0.0))
                    else:
                        p_idx = owner - n_total
                        if p_idx < n_ports:
                            pins.append((port_name_map[p_idx], 0.0, 0.0))
            else:
                nodes = benchmark.net_nodes[net_id].cpu().numpy()
                for owner in nodes:
                    owner = int(owner)
                    if owner < n_total:
                        pins.append((macro_name_map[owner], 0.0, 0.0))
                    elif owner - n_total < n_ports:
                        pins.append((port_name_map[owner - n_total], 0.0, 0.0))
            if len(pins) >= 2:
                net_lines.append((f"n_{net_id}", pins))

        total_pins = sum(len(pins) for _, pins in net_lines)
        f.write(f"NumNets : {len(net_lines)}\n")
        f.write(f"NumPins : {total_pins}\n\n")
        for net_name, pins in net_lines:
            f.write(f"NetDegree : {len(pins)}   {net_name}\n")
            for nname, ox, oy in pins:
                f.write(f"\t{nname} I : {ox:.6f}\t{oy:.6f}\n")

    # ── .wts ──────────────────────────────────────────────────────────────
    weights = benchmark.net_weights.cpu().numpy()
    with open(f"{base}.wts", "w") as f:
        f.write("UCLA wts 1.0\n\n")
        # Only emit weights for the nets we actually wrote.
        for i, (net_name, _) in enumerate(net_lines):
            # net_lines is in order of net_id, but indices may have been
            # skipped for degree<2 nets. Recover the original id from name.
            try:
                orig_id = int(net_name[2:])
                w = float(weights[orig_id])
            except (ValueError, IndexError):
                w = 1.0
            f.write(f"{net_name}\t{w:.6f}\n")

    # ── .scl  (single core row spanning the canvas) ───────────────────────
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    rh = float(row_height_microns)
    n_rows = max(1, int(ch / rh))
    sitewidth = 0.05  # μm; small enough that Xplace's row-snap is benign
    n_sites = max(1, int(cw / sitewidth))
    with open(f"{base}.scl", "w") as f:
        f.write("UCLA scl 1.0\n\n")
        f.write(f"NumRows : {n_rows}\n\n")
        for r in range(n_rows):
            y0 = r * rh
            f.write("CoreRow Horizontal\n")
            f.write(f"  Coordinate   :   {y0:.6f}\n")
            f.write(f"  Height       :   {rh:.6f}\n")
            f.write(f"  Sitewidth    :   {sitewidth:.6f}\n")
            f.write(f"  Sitespacing  :   {sitewidth:.6f}\n")
            f.write(f"  Siteorient   :   N\n")
            f.write(f"  Sitesymmetry :   Y\n")
            f.write(f"  SubrowOrigin :   0.000   NumSites : {n_sites}\n")
            f.write("End\n")

    # ── .aux ──────────────────────────────────────────────────────────────
    with open(f"{base}.aux", "w") as f:
        f.write(
            f"RowBasedPlacement : {design_name}.nodes  {design_name}.nets  "
            f"{design_name}.wts  {design_name}.pl  {design_name}.scl\n"
        )

    return macro_name_map, port_name_map


# ──────────────────────────────────────────────────────────── Bookshelf reader


def read_placed_pl(
    pl_path: str | Path,
    name_to_idx: Dict[str, int],
    benchmark: Benchmark,
) -> torch.Tensor:
    """Parse a Bookshelf .pl file and return a placement tensor in Benchmark
    index order. Unmatched entries (ports, anything not in name_to_idx) are
    skipped; unmatched benchmark macros retain their initial positions.
    """
    positions = benchmark.macro_positions.clone()
    sizes = benchmark.macro_sizes
    pl = Path(pl_path)
    if not pl.exists():
        raise FileNotFoundError(f"Xplace did not produce {pl}")

    with open(pl, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("UCLA"):
                continue
            # "name  x  y : orientation [/FIXED]"
            head, _, _ = line.partition(":")
            tokens = head.split()
            if len(tokens) < 3:
                continue
            name, sx, sy = tokens[0], tokens[1], tokens[2]
            if name not in name_to_idx:
                continue
            try:
                x_ll = float(sx); y_ll = float(sy)
            except ValueError:
                continue
            i = name_to_idx[name]
            w, h = float(sizes[i, 0]), float(sizes[i, 1])
            # Convert Bookshelf lower-left back to center.
            positions[i, 0] = x_ll + w / 2.0
            positions[i, 1] = y_ll + h / 2.0
    return positions
