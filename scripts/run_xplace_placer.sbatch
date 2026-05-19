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
# HKUST SuperPod uses Bright Cluster Manager module names (verified against
# https://itso.hkust.edu.hk/.../superpod/software-module-system).
# Modules that are actually NEEDED by Xplace:
#   - Anaconda3/2023.09-0           (Python 3 + libstdc++)
#   - cuda12.2/toolkit/12.2.2       (nvcc + libcudart for sm_90 / H800)
# Modules we deliberately DO NOT load:
#   - boost/1.81.0   : Xplace vendors Boost headers; loading the cluster's
#                      Boost only mutates LD_LIBRARY_PATH for no benefit.
#   - gcc/13.1.0     : CUDA 12.2 officially supports gcc up to 12.x. The
#                      cluster's default system gcc is 11.4 — compatible.
#                      Loading gcc/13.1.0 has caused nvcc host-compiler
#                      probe failures in past runs.
# Modules that don't exist on this cluster (we install via pip into venv):
#   - cmake          : system is 3.22.1 (< Xplace's 3.24 minimum)
#   - ninja, cairo, pkg-config
#
# Override at submit time:
#   sbatch --export=ALL,CUDA_MODULE=cuda11.8/toolkit/11.8.0 ...
: "${CUDA_MODULE:=cuda12.2/toolkit/12.2.2}"
: "${ANACONDA_MODULE:=Anaconda3/2023.09-0}"

module purge 2>/dev/null || true
module load "${ANACONDA_MODULE}" 2>&1 || echo "[modules] WARN: ${ANACONDA_MODULE} not loaded"
module load "${CUDA_MODULE}"     2>&1 || echo "[modules] WARN: ${CUDA_MODULE} not loaded"

# Pin CUDA_HOME so cmake's FindCUDAToolkit picks up the right toolkit even
# if the module didn't export it.
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi

# Make sure CUDA libs (libcudart, etc.) precede any other lib dirs in
# LD_LIBRARY_PATH — Anaconda3 ships its own libstdc++ which can shadow the
# system one and break nvcc's probe.
if [ -n "${CUDA_HOME:-}" ]; then
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

echo "[modules] nvcc: $(command -v nvcc || echo NOT-FOUND)"
echo "[modules] gcc:  $(command -v gcc  || echo NOT-FOUND) ($(gcc --version 2>/dev/null | head -1))"
echo "[modules] CUDA_HOME=${CUDA_HOME:-unset}"
echo "[modules] LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"

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
        git -C "${XPLACE_ROOT}" submodule update --init --recursive
    fi

    # ── Xplace runtime Python deps ──────────────────────────────────────
    # Xplace's main.py imports pandas/scipy/networkx/pyyaml at runtime, none
    # of which are macro-place deps. We install them into the same venv so
    # the `_run_xplace` subprocess (which invokes ${PYBIN} main.py) can find
    # them. Done BEFORE the build-idempotency check so deps are refreshed
    # every job, not just on rebuild — `uv sync --frozen` prunes anything
    # outside the lockfile on each invocation, so this must re-run.
    #
    # On HKUST SuperPod, compute nodes may have no outbound internet; this
    # works anyway because uv's per-user cache (~/.cache/uv) is shared
    # across login and compute nodes, so wheels downloaded once on the
    # login node satisfy later compute-node installs offline.
    install_xplace_runtime_deps() {
        local req="${XPLACE_ROOT}/requirements.txt"
        if [ -f "${req}" ]; then
            echo "[xplace-deps] installing from ${req}"
            pip_install -r "${req}"
        else
            # Fallback: known runtime imports in upstream Xplace main.py +
            # cpp_to_py helpers. Keep this list tight — every entry here is
            # a wheel uv has to fetch the first time a user runs this job.
            echo "[xplace-deps] no requirements.txt in Xplace, installing known set"
            pip_install pandas pyyaml networkx scipy
        fi
    }
    install_xplace_runtime_deps

    # Idempotency: don't just check for .so files — verify the package
    # actually imports under the current venv. This catches ABI drift
    # (e.g., torch was upgraded and now needs the new C++11 ABI).
    if PYTHONPATH="${XPLACE_ROOT}" "${PYBIN}" -c "import cpp_to_py.cpybin" 2>/dev/null; then
        echo "[xplace-build] cpp_to_py.cpybin imports cleanly — skipping rebuild"
        return
    fi
    # If the import fails but .so files exist, they're from a previous bad
    # build (wrong ABI, etc.). Wipe them so cmake/make redo a clean build.
    if find "${XPLACE_ROOT}/cpp_to_py/cpybin" -maxdepth 1 -name '*.so' 2>/dev/null | grep -q .; then
        echo "[xplace-build] cleaning stale .so files from previous build"
        rm -rf "${XPLACE_ROOT}/build" "${XPLACE_ROOT}/cpp_to_py/cpybin"/*.so
    fi

    # Patch out draw_placement: HKUST SuperPod has no Cairo (no module, no
    # system headers), and we don't need the --vis PNG renderer to score
    # benchmarks. Without this patch, cmake configure fails with
    # "CAIRO_INCLUDE_DIRS set to NOTFOUND".
    CPP_CMAKE="${XPLACE_ROOT}/cpp_to_py/CMakeLists.txt"
    if grep -q '^add_subdirectory(draw_placement)' "${CPP_CMAKE}"; then
        sed -i 's|^add_subdirectory(draw_placement)|# add_subdirectory(draw_placement)  # disabled: no Cairo on HKUST SuperPod|' "${CPP_CMAKE}"
        echo "[xplace-build] patched out draw_placement (no Cairo on cluster)"
    fi

    # cpp_to_py/__init__.py unconditionally does `from .cpybin import draw_placement`.
    # Since we skipped the subdir above, that .so was never built — the post-install
    # `import cpp_to_py.cpybin` check fails with:
    #   ImportError: cannot import name 'draw_placement' from 'cpp_to_py.cpybin'
    CPP_INIT="${XPLACE_ROOT}/cpp_to_py/__init__.py"
    if grep -q 'draw_placement' "${CPP_INIT}"; then
        # Drop `draw_placement` line inside a multi-line `from .cpybin import (...)`
        sed -i '/^[[:space:]]*draw_placement,\?[[:space:]]*$/d' "${CPP_INIT}"
        # Drop `, draw_placement` / `draw_placement,` on single-line imports
        sed -i 's/,[[:space:]]*draw_placement\b//g; s/\bdraw_placement[[:space:]]*,[[:space:]]*//g' "${CPP_INIT}"
        # Drop any bare `draw_placement` reference (e.g. __all__ entries)
        sed -i 's/^[[:space:]]*draw_placement[[:space:]]*$//; s/"draw_placement",\?//g; s/'\''draw_placement'\'',\?//g' "${CPP_INIT}"
        echo "[xplace-build] patched out draw_placement import in cpp_to_py/__init__.py"
    fi

    # cmake on this cluster: system is 3.22.1 (< Xplace's 3.24 floor) and
    # there is no cmake module → pip-install into the venv and call by
    # absolute path. Pin cmake≥3.28: cmake 3.27 had a regression where the
    # CUDA identification probe silently fails when CMAKE_CUDA_HOST_COMPILER
    # is set, surfacing as "_CMAKE_CUDA_WHOLE_FLAG not set". Fixed in 3.28+.
    pip_install --upgrade pybind11 "cmake>=3.28" || true

    VENV_BIN="$(dirname "${PYBIN}")"
    if [ -x "${VENV_BIN}/cmake" ]; then
        CMAKE_BIN="${VENV_BIN}/cmake"
    else
        CMAKE_BIN="$(command -v cmake || true)"
    fi

    CUDA_NVCC="${CUDA_HOME:+${CUDA_HOME}/bin/nvcc}"
    [ -x "${CUDA_NVCC}" ] || CUDA_NVCC="$(command -v nvcc)"

    # H800 = sm_90.
    : "${CMAKE_CUDA_ARCH:=90}"

    # Match torch's C++ ABI. PyTorch ≥2.4 CUDA wheels are built with the NEW
    # C++11 ABI (_GLIBCXX_USE_CXX11_ABI=1); Xplace's CMakeLists defaults to
    # the OLD ABI (=0). Mismatch surfaces at import time as:
    #   undefined symbol: _ZN3c106detail14torchCheckFailEPKcS2_jRKSs
    # ("Ss" is the old-ABI mangling for std::string; new-ABI torch doesn't
    # export it.)
    CXX11_ABI="$("${PYBIN}" -c "import torch; print(int(torch.compiled_with_cxx11_abi()))" 2>/dev/null || echo 0)"

    echo "==== xplace build sanity ===="
    echo "cmake     : ${CMAKE_BIN} ($("${CMAKE_BIN}" --version | head -1))"
    echo "nvcc      : ${CUDA_NVCC} ($("${CUDA_NVCC}" --version | tail -1))"
    echo "gcc       : $(command -v gcc) ($(gcc --version | head -1))"
    echo "g++       : $(command -v g++) ($(g++ --version | head -1))"
    echo "CUDA_HOME : ${CUDA_HOME:-unset}"
    echo "PYBIN     : ${PYBIN}"
    echo "CXX11_ABI : ${CXX11_ABI} (torch.compiled_with_cxx11_abi())"
    "${PYBIN}" -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available(), '| sm:', torch.cuda.get_arch_list() if torch.cuda.is_available() else [])" 2>&1 || true
    echo "============================="

    # ── nvcc smoke test ──────────────────────────────────────────────────
    # Compile a trivial CUDA program with nvcc directly. If THIS fails,
    # there's no point invoking cmake — the toolchain is broken at a
    # lower level than cmake's probe. The error here is far easier to
    # diagnose than cmake's cryptic "_CMAKE_CUDA_WHOLE_FLAG not set".
    SMOKE_DIR="$(mktemp -d)"
    cat > "${SMOKE_DIR}/probe.cu" <<'CUEOF'
#include <cuda_runtime.h>
#include <cstdio>
__global__ void k() {}
int main() {
    int n = 0;
    cudaGetDeviceCount(&n);
    printf("nvcc-smoke: devices=%d\n", n);
    return 0;
}
CUEOF
    echo "[smoke] compiling probe.cu with: ${CUDA_NVCC} -arch=sm_${CMAKE_CUDA_ARCH} -std=c++17"
    if ! "${CUDA_NVCC}" -arch="sm_${CMAKE_CUDA_ARCH}" -std=c++17 \
            "${SMOKE_DIR}/probe.cu" -o "${SMOKE_DIR}/probe" 2>&1; then
        echo "[smoke] FAIL — nvcc cannot compile a trivial .cu file."
        echo "        This is a toolchain problem (nvcc/host gcc/libstdc++),"
        echo "        not a cmake problem. Re-run the nvcc command above by"
        echo "        hand on a compute node to see the raw error."
        rm -rf "${SMOKE_DIR}"
        exit 1
    fi
    echo "[smoke] OK — nvcc compiled probe.cu cleanly"
    rm -rf "${SMOKE_DIR}"

    # ── cmake configure ──────────────────────────────────────────────────
    # Note: we deliberately do NOT pass -DCMAKE_CUDA_HOST_COMPILER. On
    # cmake 3.27 that flag broke the CUDA identification probe; in cmake
    # 3.28+ it still risks picking the wrong gcc out of an Anaconda env.
    # nvcc's default host compiler (resolved via PATH) is gcc 11.4 on this
    # cluster, which is supported by both CUDA 11.8 and 12.2.
    mkdir -p "${XPLACE_ROOT}/build"
    cd "${XPLACE_ROOT}/build"
    rm -f CMakeCache.txt
    if ! "${CMAKE_BIN}" .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCH}" \
        -DCMAKE_CUDA_COMPILER="${CUDA_NVCC}" \
        -DCMAKE_CXX_ABI="${CXX11_ABI}" \
        -DPYTHON_EXECUTABLE="${PYBIN}"; then
        echo "==== CMakeError.log (last 300 lines) ===="
        tail -300 "${XPLACE_ROOT}/build/CMakeFiles/CMakeError.log" 2>/dev/null \
            || echo "(no CMakeError.log)"
        echo "==== CMakeOutput.log (last 150 lines) ===="
        tail -150 "${XPLACE_ROOT}/build/CMakeFiles/CMakeOutput.log" 2>/dev/null \
            || echo "(no CMakeOutput.log)"
        echo "==== CMakeCache.txt (CUDA/TORCH/PYTHON vars) ===="
        grep -E '^(CMAKE_CUDA|CUDA|TORCH|PYTHON|Boost|CAIRO)' \
            "${XPLACE_ROOT}/build/CMakeCache.txt" 2>/dev/null \
            || echo "(no CMakeCache.txt)"
        exit 1
    fi
    "${CMAKE_BIN}" --build . -- -j"${SLURM_CPUS_PER_TASK:-8}"
    # `make install` is what copies the built shared libs into cpp_to_py/
    # where Xplace's Python imports expect them.
    "${CMAKE_BIN}" --install .
    cd "${REPO_ROOT}"

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
