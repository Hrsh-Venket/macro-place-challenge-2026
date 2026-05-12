#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# SLURM job: evaluate the EfficientPlacer (with PPO training) on a GPU node.
# Target: HKUST SuperPod (normal partition, H800 80GB,
# 14 cores / 224GB RAM per GPU, 3-day max walltime).
#
# Submit (from anywhere — the script locates the repo from its own path):
#   sbatch /path/to/macro-place-challenge-2026/scripts/train_efficient_placer.sbatch
#
# All artefacts land under  <repo>/runs/<jobid>/ :
#   stdout.log     — combined stdout+stderr from the evaluate run
#   env.txt        — captured GPU / torch / CUDA info
#   slurm.out      — symlink to SLURM's raw stdout
#   slurm.err      — symlink to SLURM's raw stderr
#   vis/           — placement visualisations (from --vis)
#
# SLURM's own log files land beside this script's submission cwd as
# slurm-<jobid>.{out,err} (these are symlinked into runs/<jobid>/ for
# convenience — SLURM holds the live file handle so they cannot be moved).
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=eff_placer
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus=1                          # 1 H800
#SBATCH --cpus-per-task=14                # per-GPU CPU allotment
#SBATCH --mem=224G                        # per-GPU RAM allotment
#SBATCH --time=24:00:00                   # bump to 3-00:00:00 if needed
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# Uncomment and set if your SuperPod account spans multiple projects:
##SBATCH --account=<your_project>

set -euo pipefail

# ── Locate the repo ──────────────────────────────────────────────────────────
# SLURM copies the script into its own spool dir before running, so we can't
# rely on $BASH_SOURCE / $0 to point at the original file. Use SLURM_SUBMIT_DIR
# (the dir where `sbatch` was invoked) and walk up until we find pyproject.toml.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "${REPO_ROOT}" != "/" ] && [ ! -f "${REPO_ROOT}/pyproject.toml" ]; do
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
if [ ! -f "${REPO_ROOT}/pyproject.toml" ]; then
    echo "ERROR: could not find repo root (no pyproject.toml at or above ${SLURM_SUBMIT_DIR})" >&2
    exit 1
fi
cd "${REPO_ROOT}"

# ── Submodules (plc_client_os lives in external/MacroPlacement) ──────────────
if [ ! -f "${REPO_ROOT}/external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py" ]; then
    git -C "${REPO_ROOT}" submodule update --init --recursive
fi

# ── Per-job output directory ─────────────────────────────────────────────────
RUN_DIR="${REPO_ROOT}/runs/${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}/vis"

# Symlink SLURM's raw logs into the run dir. They live in SLURM_SUBMIT_DIR
# (where `sbatch` was invoked) since SLURM opens them before the script runs.
SLURM_STDOUT="${SLURM_SUBMIT_DIR}/slurm-${SLURM_JOB_ID}.out"
SLURM_STDERR="${SLURM_SUBMIT_DIR}/slurm-${SLURM_JOB_ID}.err"
[ -e "${SLURM_STDOUT}" ] && ln -sf "${SLURM_STDOUT}" "${RUN_DIR}/slurm.out"
[ -e "${SLURM_STDERR}" ] && ln -sf "${SLURM_STDERR}" "${RUN_DIR}/slurm.err"

# ── Modules ──────────────────────────────────────────────────────────────────
module purge
module load Anaconda3/2023.09-0 2>/dev/null || module load Anaconda3 2>/dev/null || true
# Uncomment if PyTorch needs a system CUDA module on this cluster:
# module load cuda/12.1

# ── uv (idempotent install if missing) ───────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

# Sync deps from pyproject.toml / uv.lock into a local .venv.
uv sync --frozen

# ── Diagnostics ──────────────────────────────────────────────────────────────
{
    echo "================================ Job info ================================"
    echo "Host           : $(hostname)"
    echo "Job ID         : ${SLURM_JOB_ID}"
    echo "Submit dir     : ${SLURM_SUBMIT_DIR}"
    echo "Repo root      : ${REPO_ROOT}"
    echo "Run dir        : ${RUN_DIR}"
    echo "Allocated GPUs : ${SLURM_GPUS:-?}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    echo
    nvidia-smi || echo "(nvidia-smi unavailable)"
    echo
    uv run python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" \
        || echo "(torch import failed)"
    echo "=========================================================================="
} > "${RUN_DIR}/env.txt" 2>&1
cat "${RUN_DIR}/env.txt"

# ── Run evaluation ───────────────────────────────────────────────────────────
# `--all` runs the 17 IBM benchmarks; swap in `--ng45` or `-b ibm03` as needed.
# Combined stdout+stderr is tee'd to RUN_DIR/stdout.log so the job log mirrors
# the live progress.
uv run evaluate submissions/examples/efficient_placer.py --all --vis 2>&1 \
    | tee "${RUN_DIR}/stdout.log"

# Move any visualisations emitted by --vis into the run dir.
if [ -d "${REPO_ROOT}/vis" ]; then
    mv "${REPO_ROOT}/vis"/*.png "${RUN_DIR}/vis/" 2>/dev/null || true
    rmdir "${REPO_ROOT}/vis" 2>/dev/null || true
fi

echo "Done. Outputs: ${RUN_DIR}/"
