# Installation

Our codebase is built upon **Python 3.12, PyTorch 2.4.0 (recommended)**. 

:warning: As far as I know, due to the use of some new language features in our code, Python version 3.10 or higher is required. For PyTorch, because there have been changes in the type requirements for attention masks, PyTorch version 2.0 or higher is needed.

:construction: We plan to support lower versions of PyTorch in the future, but the exact timeline is yet to be determined. Currently, we do not have sufficient manpower to address this issue.

## Setup scripts

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv sync --python 3.12
# Compile the Deformable Attention:
cd models/ops/
sh make.sh
# [Optional] After compiled, you can use following script to test it:
uv run --no-sync python test.py
```

The `uv` environment matches the original install as closely as possible:
- Python `3.12`
- PyTorch `2.4.0`, `torchvision` `0.19.0`, `torchaudio` `2.4.0`
- CUDA 12.1 wheels from the official PyTorch index configured in `pyproject.toml`
- Original MOTIP runtime dependencies plus the extra packages this fork imports

The deformable attention build still requires a CUDA-enabled environment with a
working toolkit, matching the original `setup.py` behavior.

