# XplacePlacer

Xplace (CUHK, DAC 2022 / TCAD 2024) global placement + exact-proxy
coordinate-descent polish.

## What it does

1. **Bookshelf export** — writes the in-memory `Benchmark` to a UCLA
   Bookshelf (ICCAD04) dataset in a temp dir.
2. **Xplace** — invoked as a subprocess. Runs ePlace-style global placement
   (electrostatic density via FFT-Poisson + weighted-average HPWL,
   Nesterov AG, λ continuation). Writes a placed `.pl`.
3. **CD polish** — coordinate descent on **exact** HPWL with hard-macro
   zero-overlap constraint. This closes the surrogate gap between Xplace's
   smoothed objective and the competition's discrete proxy cost.
4. **Soft-macro FD** *(optional)* — quick force-directed pass for soft
   macros that follows the new hard-macro layout.

## Build (Docker — required for judges)

```bash
docker build -t xplace-placer -f submissions/xplace_placer/Dockerfile .
```

The image installs Xplace at `/opt/Xplace` and exposes
`XPLACE_ROOT=/opt/Xplace`. The base image is `nvidia/cuda:11.8.0-devel`;
the RTX 6000 Ada eval GPU is supported.

## Run a single benchmark

```bash
docker run --gpus all --rm -v $PWD:/workspace xplace-placer \
    uv run evaluate submissions/xplace_placer/placer.py -b ibm01
```

Or all 17:

```bash
docker run --gpus all --rm -v $PWD:/workspace xplace-placer \
    uv run evaluate submissions/xplace_placer/placer.py --all
```

## Hyperparameters worth sweeping

Edit defaults in `XplacePlacer.__init__` or pass a sweep wrapper.

| Knob | Default | Notes |
|---|---|---|
| `target_density` | 0.65 | Xplace utilisation target. Lower spreads more; 0.55–0.75 is reasonable for these benchmarks. |
| `cd_grid_step` | 0.5 μm | CD candidate spacing. Smaller → finer (slower). |
| `cd_radius_factor` | 2.0 | Search radius = factor × macro diagonal. |
| `cd_sweeps` | 4 | CD passes; early-stops on stall. |
| `cd_time_budget_s` | 900 | Wall-clock cap for the CD stage. |

## Local dev without GPU

If you're iterating on the wrapper itself (Bookshelf writer, CD logic) on
a Mac, you can bypass Xplace and run CD on the benchmark's initial
positions:

```python
from submissions.xplace_placer.placer import XplacePlacer
XplacePlacer(skip_xplace=True).place(benchmark)
```

This is for plumbing-level debugging only — the proxy cost will be
terrible without the global placement step.

## File layout

```
submissions/xplace_placer/
├── Dockerfile           # CUDA + Boost + Cairo + Xplace
├── placer.py            # XplacePlacer.place(benchmark) — entry point
├── bookshelf_io.py      # Benchmark → Bookshelf; .pl → positions
├── cd_polish.py         # exact-HPWL coordinate descent
└── README.md            # you are here
```

## Known unknowns

* **Xplace CLI flags** in `placer._run_xplace` are written against the
  conventional `main.py --dataset_format bookshelf --aux ... --output_dir
  ...` form. Versions diverge — verify against `python main.py --help` in
  the image and pass overrides via `XplacePlacer(xplace_extra_args=...)`.
* **`.pl` output filename** varies (`<design>.gp.pl`, `<design>.lg.pl`,
  etc.). The reader scans for any `*.pl` in the output dir; if Xplace
  produces multiple stages we take the largest one (post-legalization).
* **Pin offsets for soft macros and ports** are zero in our Bookshelf
  export (Benchmark doesn't carry them). This matches the existing
  `efficient_placer.py` and `compute_proxy_cost`.
