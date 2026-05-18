# Xplace + macro-place submission image.
#
# Build:
#   docker build -t xplace-placer -f submissions/xplace_placer/Dockerfile .
# Run on a single benchmark:
#   docker run --gpus all --rm -v $PWD:/workspace xplace-placer \
#       uv run evaluate submissions/xplace_placer/placer.py -b ibm01
#
# The image installs Xplace at /opt/Xplace and exposes XPLACE_ROOT so the
# placer's _find_xplace_root() picks it up automatically.

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    XPLACE_ROOT=/opt/Xplace

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl wget \
        build-essential cmake ninja-build \
        libboost-all-dev libcairo2-dev \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        flex bison \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# uv (matches the host project setup so we can reuse pyproject.toml).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv

# ── Build Xplace ──────────────────────────────────────────────────────────
# Pin to a known-good tag if upstream HEAD breaks; default to main for now.
ARG XPLACE_REPO=https://github.com/cuhk-eda/Xplace.git
ARG XPLACE_REF=main
RUN git clone --depth 1 --branch ${XPLACE_REF} ${XPLACE_REPO} ${XPLACE_ROOT}

WORKDIR ${XPLACE_ROOT}
RUN python -m pip install --no-cache-dir -U pip setuptools wheel \
    && if [ -f requirements.txt ]; then python -m pip install --no-cache-dir -r requirements.txt; fi
# Build C++/CUDA ops (Xplace's build is invoked from setup.py / cmake).
RUN if [ -f setup.py ]; then python setup.py build_ext --inplace; \
    elif [ -f CMakeLists.txt ]; then mkdir -p build && cd build && cmake .. && make -j$(nproc); \
    fi

# ── Project code ──────────────────────────────────────────────────────────
WORKDIR /workspace
COPY pyproject.toml uv.lock* /workspace/
RUN uv sync --frozen || uv sync
COPY . /workspace

# Smoke test: import the placer module so build failures surface at image build time.
RUN python -c "import sys; sys.path.insert(0, 'submissions/xplace_placer'); import placer"

CMD ["bash"]
