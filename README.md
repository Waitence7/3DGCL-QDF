# 3D Graph Contrastive Learning for Molecular Property Prediction
Official Code Repository for the paper "3D Graph Contrastive Learning for Molecular Property Prediction": [link](https://academic.oup.com/bioinformatics/article/39/6/btad371/7192173?utm_source=advanceaccess&utm_campaign=bioinformatics&utm_medium=email&login=true)

## Abstract
Self-supervised learning (SSL) is a method that learns the data representation by utilizing supervision inherent in the data. This learning method is in the spotlight in the drug field, lacking annotated data due to time-consuming and expensive experiments. SSL using enormous unlabeled data has shown excellent performance for molecular property prediction, but a few issues exist. (1) Existing SSL models are large-scale; there is a limitation to implementing SSL where the computing resource is insufficient. (2) In most cases, they do not utilize 3D structural information for molecular representation learning. The activity of a drug is closely related to the structure of the drug molecule. Nevertheless, most current models do not use 3D information or use it partially. (3) Previous models that apply contrastive learning to molecules use the augmentation of permuting atoms and bonds. Therefore, molecules having different characteristics can be in the same positive samples. We propose a novel contrastive learning framework, small-scale 3D Graph Contrastive Learning (3DGCL) for molecular property prediction, to solve the above problems. 3DGCL learns the molecular representation by reflecting the molecule’s structure through the pre-training process that does not change the semantics of the drug. Using only 1,128 samples for pre-train data and 0.5 million model parameters, we achieved state-of-the-art or comparable performance in six benchmark datasets. Extensive experiments demonstrate that 3D structural information based on chemical knowledge is essential to molecular representation learning for property prediction.


## Overview
<p align="center">
<img src=figures/3DGCL.png width=900px>
<img src=figures/methods_3D.png width=700px>
</p>

### Contribution
- We develop a compact self-supervised learning approach that can be run even in environments with low computational resources, using the small-scale pre-train samples and parameters. We also achieve the state-of-the-art or comparable performance in four regression benchmarks.
- To the best of our knowledge, we propose 3D-3D view contrastive learning that can take full advantage of 3D information for the first time. We actively utilize 3D positional information inherent in molecules through the pre-train scheme using the conformer pool.
- Extensive experiments demonstrate that our method, which can utilize structural information abundantly while maintaining semantics, is more suitable for molecular property prediction than conventional methods that can significantly change the structure or properties of molecules.

## Dependencies

Original paper code targeted:

- Python 3.6.9
- PyTorch 1.7.1
- PyTorch Geometric 2.0.3
- RDKit 2021.3.4

### Modern setup with uv (recommended)

This repository includes `pyproject.toml` and `uv.lock` so you can install CPU PyTorch + PyTorch Geometric + RDKit and the bundled `dig` package in one environment (Python 3.10–3.12):

```powershell
uv sync
# Register a Jupyter kernel that always uses this project's .venv (avoids conda/base mixing).
uv run python -m ipykernel install --user --name 3dgcl --display-name "Python (3dgcl uv)"

uv run jupyter notebook examples/sslgraph/pretrain.ipynb
```

After opening the notebook, choose kernel **Python (3dgcl uv)**. If PyTorch raises `_cuda` / `_C` errors, you are almost certainly on the wrong interpreter (conda `base` + another Python can break torch DLL init on Windows).

The environment setup cell asserts that `sys.executable` lives under `.venv`, **only clears `torch*` from `sys.modules` when the cached module is broken** (missing `torch._C` after a failed import — never after a healthy import, which would cause `RuntimeError: ... '_has_torch_function' already has a docstring`), and on Windows registers extra DLL directories when conda `Library\\bin` is present.

If you still hit `_C` / `_cuda` errors, use **Kernel → Restart**, then run the first cell once before others.

`numpy` is capped below 2 for compatibility with RDKit binaries. CPU PyTorch is used by default to avoid missing CUDA runtime DLL errors on Windows. The same env also satisfies [`QuantumDeepField_molecule`](https://github.com/masashitsubaki/QuantumDeepField_molecule) training scripts (`torch`, `numpy`, `scipy` only).

Optional visualization for QDF: `uv sync --extra mayavi`. Mayavi often builds from source and may print **Setuptools deprecation warnings about “License classifiers”**—that comes from upstream Mayavi packaging, not this repo. On **Windows**, the Mayavi/VTK wheel build commonly **fails** (e.g. access violation during `tvtk` codegen); omit `--extra mayavi` or install Mayavi/Vtk via **conda**/OS packages instead.

### Accelerators (GPU, Intel XPU, optional DirectML / NPU-oriented stacks)

The helper `dig.sslgraph.utils.pick_torch_device()` resolves the runtime device:

1. `TORCH_DEVICE` if set (examples: `cuda:0`, `xpu:0`, `directml`, `cpu`)
2. NVIDIA **CUDA**, if drivers + CUDA build are installed (`CUDA_DEVICE_INDEX` optional)
3. **Apple MPS** when available  
4. **Intel GPU (XPU)** when `torch.xpu` is registered (install the **`-xpu`** PyTorch stack; this repo locks it as a uv **dependency group** alongside Intel SYCL/TRiton wheels from [pytorch.org/whl/xpu](https://download.pytorch.org/whl/xpu))
5. **Huawei Ascend** `torch.npu` after the above unless **`TORCH_SKIP_NPU`** is truthy (`1` / `true` / `yes` / `on`); optionally `NPU_DEVICE_INDEX`
6. **Microsoft DirectML** (Windows, DX12 GPUs — often Intel/AMD/NVIDIA integrated or discrete GPUs are visible here): install [`torch-directml`](https://github.com/microsoft/DirectML) in the same env, then set **`USE_DIRECTML=1`** or **`TORCH_DEVICE=dml`**, optionally `DIRECTML_DEVICE_INDEX`.

Default `uv.lock` still defaults to **CPU** PyTorch (`[tool.uv] default-groups = ["cpu"]`, so plain `uv sync` matches the previous layout). For **Intel XPU**, sync the separate fork: `uv sync --no-default-groups --group xpu` (needs current Intel GPU drivers / runtime per [PyTorch Intel GPU](https://pytorch.org/get-started/locally/)). Switch back to CPU: run `uv sync` again. Optional [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) is not required for upstream `torch+xpu` builds but remains a harmless optional import in `pick_torch_device`. For NVIDIA training, reinstall a CUDA wheel from PyPI / [pytorch.org](https://pytorch.org/get-started/locally/), then rerun `uv pip install`/`uv lock` for matching **torch-geometric** wheels.

Dedicated **Intel AI Boost / Core Ultra tile NPUs** are usually not surfaced as plain `torch.device` backends in stock PyTorch; use vendor OpenVINO/ONNX NPU workflows if you mean that hardware strictly. DirectML above targets the **GPU** DX12 stack, which is separate from those NPUs.

## Run
### 1. Pre-train
```shell script
run examples/sslgraph/pretrain.ipynb
```

### 2. Fine-tune
```shell script
run examples/sslgraph/finetune.ipynb
```

### 3. Supervised learning (No pre-train)
```shell script
run examples/sslgraph/downstream.ipynb
```


## Acknowledgment
Our implementation is mainly based on the following work.

[DIG: A Turnkey Library for Diving into Graph Deep Learning Research](https://github.com/divelab/DIG)

We are thankful for the great work.
