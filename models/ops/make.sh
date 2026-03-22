#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------


# TORCH_CUDA_ARCH_LIST="8.0" CUDA_HOME='/path/to/your/cuda/dir'
set -eu

# Docker builds do not expose a live GPU, so PyTorch cannot infer a CUDA arch
# list automatically. Default to a broad set of common datacenter archs unless
# the caller provides a more specific value.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
fi

# Use the uv-managed interpreter when available so the extension is installed
# into the same environment as the rest of the project dependencies.
if command -v uv >/dev/null 2>&1; then
  uv run --no-sync python setup.py build install
else
  python setup.py build install
fi
