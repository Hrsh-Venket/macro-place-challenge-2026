#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# SLURM job: evaluate the XplacePlacer on a GPU node.
#
# Pipeline:
#   1. Locate repo root from the submit dir
#   2. Initialise the TILOS submodule (plc_client_os) if missing
#   3. Install / refresh uv venv from pyproject.toml
#   4. Build Xplace at $XPLACE_ROOT (default: $HOME/Xplace) if not already built
#   5. Run `uv run evaluate submissions/xplace_placer/placer.py …`
#   6. Stash everything (stdout, env, vis/, slurm raw logs) under runs/<jobid>/
#
# Submit:
#   sbatch /path/to/repo/scripts/run_xplace_placer.sbatch
#
# Override defaults at submit time:
#   sbatch --export=ALL,XPLACE_BENCH="-b ibm03" ./run_xplace_placer.sbatch
#   sbatch --export=ALL,XPLACE_BENCH="--ng45" ./run_xplace_placer.sbatch
#   sbatch --export=ALL,XPLACE_BENCH="--all --vis" ./run_xplace_placer.sbatch
#   sbatch --export=ALL,XPLACE_ROOT=/scratch/$USER/Xplace ./run_xplace_placer.sbatch
#
# Skip the Xplace stage entirely (CD only — useful for plumbing tests):
#   sbatch --export=ALL,SKIP_XPLACE=1 ./run_xplace_placer.sbatch
#
# Artefacts:
#   runs/<jobid>/stdout.log       full evaluate stdout+stderr
#   runs/<jobid>/env.txt          host / GPU / torch / Xplace versions
#   runs/<jobid>/xplace_build.log Xplace build log (only first install)
#   runs/<jobid>/slurm.out        symlink to SLURM's raw stdout
#   runs/<jobid>/slurm.err        symlink to SLURM's raw stderr
#   runs/<jobid>/vis/             placement images (if --vis was passed)
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=xplace_placer
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=224G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --account=mscaisuperpod

set -euo pipefail

# ── Configurable knobs (override via --export=ALL,KEY=VAL) ──────────────────
: "${XPLACE_BENCH:=--all}"          # default: all 17 IBM benchmarks
: "${XPLACE_REPO:=https://github.com/cuhk-eda/Xplace.git}"
: "${XPLACE_REF:=main}"
: "${XPLACE_ROOT:=$HOME/Xplace}"
: "${SKIP_XPLACE:=0}"               # set 1 to skip Xplace build/run (CD-only)

export XPLACE_ROOT

# ── Locate the repo root ────────────────────────────────────────────────────
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "${REPO_ROOT}" != "/" ] && [ ! -f "${REPO_ROOT}/pyproject.toml" ]; do
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
if [ ! -f "${REPO_ROOT}/pyproject.toml" ]; then
    echo "ERROR: no pyproject.toml at or above ${SLURM_SUBMIT_DIR}" >&2
    exit 1
fi
cd "${REPO_ROOT}"

# ── Submodule init (TILOS MacroPlacement / plc_client_os) ───────────────────
if [ ! -f "${REPO_ROOT}/external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py" ]; then
    git -C "${REPO_ROOT}" submodule update --init --recursive
fi

# ── Per-job output dir ──────────────────────────────────────────────────────
RUN_DIR="${REPO_ROOT}/runs/${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}/vis"

SLURM_STDOUT="${SLURM_SUBMIT_DIR}/slurm-${SLURM_JOB_ID}.out"
SLURM_STDERR="${SLURM_SUBMIT_DIR}/slurm-${SLURM_JOB_ID}.err"
[ -e "${SLURM_STDOUT}" ] && ln -sf "${SLURM_STDOUT}" "${RUN_DIR}/slurm.out"
[ -e "${SLURM_STDERR}" ] && ln -sf "${SLURM_STDERR}" "${RUN_DIR}/slurm.err"

# ── Modules ─────────────────────────────────────────────────────────────────
# HKUST SuperPod uses Bright Cluster Manager-style module names (verified
# against https://itso.hkust.edu.hk/.../superpod/software-module-system):
#   cuda11.8/toolkit/11.8.0    cuda12.2/toolkit/12.2.2
#   gcc/13.1.0                 Anaconda3/2023.09-0      boost/1.81.0
# There is NO cmake module; we pip-install it into the venv below.
#
# Defaults: CUDA 12.2 (full sm_90 / H800 support) + system gcc (CUDA 12.2's
# nvcc rejects gcc ≥ 14 but accepts the cluster's default 11.x; loading
# gcc/13.1.0 explicitly previously broke nvcc's host-compiler probe and
# caused cmake's "_CMAKE_CUDA_WHOLE_FLAG not set" error).
#
# Override at submit time:
#   sbatch --export=ALL,CUDA_MODULE=cuda11.8/toolkit/11.8.0 ...
#   sbatch --export=ALL,GCC_MODULE=gcc/13.1.0 ...   # force a specific gcc
: "${CUDA_MODULE:=cuda12.2/toolkit/12.2.2}"
: "${GCC_MODULE:=}"   # empty = don't override system gcc
: "${BOOST_MODULE:=boost/1.81.0}"
: "${ANACONDA_MODULE:=Anaconda3/2023.09-0}"

module purge 2>/dev/null || true
module load "${ANACONDA_MODULE}" 2>&1 || echo "[modules] WARN: ${ANACONDA_MODULE} not loaded"
module load "${CUDA_MODULE}"     2>&1 || echo "[modules] WARN: ${CUDA_MODULE} not loaded"
if [ -n "${GCC_MODULE}" ]; then
    module load "${GCC_MODULE}"  2>&1 || echo "[modules] WARN: ${GCC_MODULE} not loaded"
fi
module load "${BOOST_MODULE}"    2>&1 || echo "[modules] WARN: ${BOOST_MODULE} not loaded"

# Pin CUDA_HOME so cmake's FindCUDA / FindCUDAToolkit picks up the right one
# even if the module didn't set it.
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi

echo "[modules] nvcc: $(command -v nvcc || echo NOT-FOUND)"
echo "[modules] gcc:  $(command -v gcc  || echo NOT-FOUND) ($(gcc --version 2>/dev/null | head -1))"
echo "[modules] CUDA_HOME=${CUDA_HOME:-unset}"

# ── Python environment: prefer uv, fall back to pip+venv ────────────────────
# HKUST SuperPod compute nodes don't ship uv. The login-node-installed copy
# at ~/.local/bin is usually on the compute-node $PATH after `module load
# Anaconda3`, but not guaranteed. If uv isn't available and we can't fetch
# it (compute nodes often have no outbound internet), we fall back to
# `python -m venv .venv` + `pip install -e .` from pyproject.toml.
#
# After this block:
#   PYRUN     — wrapper to run a console entry point (e.g., "$PYRUN evaluate ...")
#   PYBIN     — direct path to the project venv's python

setup_with_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        for cand in "${HOME}/.local/bin/uv" "${HOME}/.cargo/bin/uv" /opt/uv/bin/uv; do
            if [ -x "${cand}" ]; then
                export PATH="$(dirname "${cand}"):${PATH}"
                break
            fi
        done
    fi
    if ! command -v uv >/dev/null 2>&1; then
        # Bounded install attempt — fail fast if the compute node has no internet.
        echo "[env] uv not found, attempting install (10s timeout)..."
        if timeout 30 curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh 2>/dev/null \
           && timeout 60 sh /tmp/uv-install.sh >/dev/null 2>&1; then
            export PATH="${HOME}/.local/bin:${PATH}"
        else
            echo "[env] uv install failed (no internet?), will fall back to pip"
            return 1
        fi
    fi
    uv sync --frozen
    PYRUN="uv run"
    PYBIN="${REPO_ROOT}/.venv/bin/python"
    USED_UV=1
    return 0
}

setup_with_pip() {
    # Resolve a working python. Prefer the Anaconda module loaded above.
    PYTHON_BIN="$(command -v python3 || command -v python)"
    if [ -z "${PYTHON_BIN}" ]; then
        echo "ERROR: no python on PATH after module load" >&2
        return 1
    fi
    echo "[env] using ${PYTHON_BIN} ($(${PYTHON_BIN} --version 2>&1))"

    VENV="${REPO_ROOT}/.venv"
    if [ ! -x "${VENV}/bin/python" ]; then
        "${PYTHON_BIN}" -m venv "${VENV}"
    fi
    # Quote-trick to bypass `set -u` while activating.
    # shellcheck disable=SC1091
    . "${VENV}/bin/activate"
    python -m pip install --upgrade pip wheel setuptools

    # Editable install of macro_place + all runtime deps from pyproject.toml.
    # If the cluster blocks PyPI, the user must pre-stage a wheelhouse and add
    # `--no-index --find-links /path/to/wheels` here.
    python -m pip install -e "${REPO_ROOT}"

    PYRUN="${VENV}/bin/python -m"
    PYBIN="${VENV}/bin/python"
    USED_UV=0
    return 0
}

USED_UV=0
if ! setup_with_uv; then
    setup_with_pip
fi

# ── pip helper: `uv sync` creates pip-less venvs, so route installs via
# `uv pip install` when we're on the uv path. ────────────────────────────────
pip_install() {
    if [ "${USED_UV}" = "1" ]; then
        uv pip install "$@"
    else
        "${PYBIN}" -m pip install "$@"
    fi
}

# Verify the evaluate entry-point is reachable, regardless of how we got here.
if ! ${PYBIN} -c "from macro_place import evaluate" >/dev/null 2>&1; then
    echo "ERROR: macro_place.evaluate not importable from ${PYBIN}" >&2
    exit 1
fi

# Resolve the command that actually runs `evaluate`. With uv we call
# `uv run evaluate …`; with the pip fallback we call the console script
# the editable install dropped into the venv's bin/.
if [ "${PYRUN}" = "uv run" ]; then
    EVAL_CMD=(uv run evaluate)
else
    EVAL_CMD=("${REPO_ROOT}/.venv/bin/evaluate")
fi

# ── Build Xplace (idempotent — skipped if cpp_to_py/*.so already installed) ─
build_xplace() {
    if [ ! -d "${XPLACE_ROOT}/.git" ]; then
        echo "[xplace-build] cloning ${XPLACE_REPO} (ref ${XPLACE_REF}) → ${XPLACE_ROOT}"
        git clone --recursive --branch "${XPLACE_REF}" "${XPLACE_REPO}" "${XPLACE_ROOT}"
    else
        # Make sure submodules are present even on a re-run.
        git -C "${XPLACE_ROOT}" submodule update --init --recursive
    fi

    # Detect a working build: `make install` lands cpp_to_py/cpybin*.so.
    if find "${XPLACE_ROOT}/cpp_to_py" -maxdepth 2 -name 'cpybin*.so' 2>/dev/null | grep -q .; then
        echo "[xplace-build] cpp_to_py/cpybin*.so already present, skipping rebuild"
        return
    fi

    # Xplace needs CMake ≥3.24, Boost ≥1.56, Cairo, CUDA ≥11.3, Torch ≥1.12.
    # Torch comes from our venv; the rest must be module-loaded above.
    # HKUST SuperPod ships cmake 3.22.1, which is too old — pip-install a
    # modern one into the venv and call it by absolute path so the system
    # cmake on $PATH doesn't shadow it.
    #
    # Pin cmake 3.27.x: cmake 3.28 had a regression in CUDA whole-archive
    # detection that surfaces as "_CMAKE_CUDA_WHOLE_FLAG not set".
    pip_install --upgrade pybind11 "cmake>=3.27,<3.28" || true

    # Locate the venv's cmake (pip wheel drops it next to PYBIN).
    VENV_BIN="$(dirname "${PYBIN}")"
    if [ -x "${VENV_BIN}/cmake" ]; then
        CMAKE_BIN="${VENV_BIN}/cmake"
    else
        CMAKE_BIN="$(command -v cmake || true)"
    fi
    echo "[xplace-build] cmake: ${CMAKE_BIN} ($("${CMAKE_BIN}" --version | head -1))"

    # H800 = sm_90. Hard-code instead of `native` since the build sometimes
    # runs on a CPU partition or in a container where the GPU probe fails.
    : "${CMAKE_CUDA_ARCH:=90}"
    # Force the CUDA host compiler to whichever gcc is on $PATH so cmake's
    # internal probe doesn't pick up an unsupported one and silently leave
    # _CMAKE_CUDA_WHOLE_FLAG unset.
    CUDA_HOST_GCC="$(command -v gcc)"

    mkdir -p "${XPLACE_ROOT}/build"
    cd "${XPLACE_ROOT}/build"
    "${CMAKE_BIN}" .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCH}" \
        -DCMAKE_CUDA_HOST_COMPILER="${CUDA_HOST_GCC}" \
        -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" \
        -DPYTHON_EXECUTABLE="${PYBIN}"
    "${CMAKE_BIN}" --build . -- -j"${SLURM_CPUS_PER_TASK:-8}"
    # CRITICAL: `make install` is what copies the built shared libs into
    # cpp_to_py/ where Xplace's Python imports expect them.
    "${CMAKE_BIN}" --install .
    cd "${REPO_ROOT}"

    # Verify the import works before we commit to running anything.
    if ! PYTHONPATH="${XPLACE_ROOT}" "${PYBIN}" -c "import cpp_to_py.cpybin" 2>&1; then
        echo "[xplace-build] ERROR: cpp_to_py.cpybin still not importable after make install" >&2
        find "${XPLACE_ROOT}/cpp_to_py" -name '*.so' 2>/dev/null || true
        exit 1
    fi
    echo "[xplace-build] cpp_to_py.cpybin imports cleanly"
}

if [ "${SKIP_XPLACE}" = "0" ]; then
    build_xplace 2>&1 | tee -a "${RUN_DIR}/xplace_build.log"
fi

# ── Diagnostics ────────────────────────────────────────────────────────────
{
    echo "================================ Job info ================================"
    echo "Host           : $(hostname)"
    echo "Job ID         : ${SLURM_JOB_ID}"
    echo "Submit dir     : ${SLURM_SUBMIT_DIR}"
    echo "Repo root      : ${REPO_ROOT}"
    echo "Run dir        : ${RUN_DIR}"
    echo "XPLACE_ROOT    : ${XPLACE_ROOT}"
    echo "XPLACE_BENCH   : ${XPLACE_BENCH}"
    echo "SKIP_XPLACE    : ${SKIP_XPLACE}"
    echo "Allocated GPUs : ${SLURM_GPUS:-?}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    echo
    nvidia-smi || echo "(nvidia-smi unavailable)"
    echo
    echo "Python runner : ${PYRUN}"
    echo "Python venv   : ${PYBIN}"
    echo "Used uv       : ${USED_UV}"
    echo
    echo "── module list (currently loaded) ─────────────────────────────────"
    module list 2>&1 || echo "(module command unavailable)"
    echo
    echo "── module avail (first 80 lines) ──────────────────────────────────"
    module avail 2>&1 | head -80 || true
    echo
    echo "── toolchain ──────────────────────────────────────────────────────"
    echo "cmake: $(command -v cmake) $(cmake --version 2>/dev/null | head -1)"
    echo "nvcc:  $(command -v nvcc)  $(nvcc --version 2>/dev/null | tail -1)"
    echo "gcc:   $(command -v gcc)   $(gcc --version 2>/dev/null | head -1)"
    echo "CUDA_HOME=${CUDA_HOME:-unset}"
    echo
    "${PYBIN}" -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" \
        || echo "(torch import failed)"
    echo
    if [ "${SKIP_XPLACE}" = "0" ] && [ -d "${XPLACE_ROOT}/.git" ]; then
        echo "Xplace HEAD    : $(git -C "${XPLACE_ROOT}" rev-parse --short HEAD)"
        echo "Xplace commit  : $(git -C "${XPLACE_ROOT}" log -1 --format='%s' HEAD)"
    fi
    echo "=========================================================================="
} > "${RUN_DIR}/env.txt" 2>&1
cat "${RUN_DIR}/env.txt"

# ── Run evaluation ─────────────────────────────────────────────────────────
# Tee combined stdout+stderr to stdout.log so SLURM's slurm-*.out and the
# run-dir log both contain the live progress.
set -o pipefail
"${EVAL_CMD[@]}" submissions/xplace_placer/placer.py ${XPLACE_BENCH} 2>&1 \
    | tee "${RUN_DIR}/stdout.log"

# Collect visualisations if --vis was requested.
if [ -d "${REPO_ROOT}/vis" ]; then
    mv "${REPO_ROOT}/vis"/*.png "${RUN_DIR}/vis/" 2>/dev/null || true
    rmdir "${REPO_ROOT}/vis" 2>/dev/null || true
fi

echo "Done. Outputs: ${RUN_DIR}/"
