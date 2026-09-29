# Installation

All tasks use this Linux x86_64 environment: Python 3.10.18, PyTorch 2.7.1,
torchvision 0.22.1 and CUDA 12.6. A compatible NVIDIA driver is required.

Run from the repository root:

```bash
export PYTHONNOUSERSITE=1
conda create --yes --name l2t --override-channels --channel defaults --file environments/cu126-conda-spec.txt
conda activate l2t
python -m pip install --no-deps -r environments/cu126-pip-requirements.txt
python -m pip install --no-deps -e .
python -m pip check
```

Keep `PYTHONNOUSERSITE=1` set when running experiments so user-level packages do
not override this environment. To check the installed versions:

```bash
python -m training.environment --require-cuda
```
