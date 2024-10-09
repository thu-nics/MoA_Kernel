# MoA Kernel

This is the CUDA kernel implementation for `MoA: Mixture of Sparse Attention for Automatic Large Language Model Compression`, or [MoA](https://github.com/thu-nics/MoA).

# Installation

> We test our kernel with `CUDA 12.4` and `PyTorch 2.4`. Install the required environments for MoA before installing the kernel.

```bash
cd python
FLASHINFER_LOGITS_POST_HOOKS=0 FLASHINFER_HEAD_DIMS=64,128 FLASHINFER_POS_ENCODING_MODES=0 python setup.py install
```

# Quick Test

```python
python accuracy_test.py
```

# Acknowledgement

Our kernel is build upon [FlashInfer](https://github.com/flashinfer-ai/flashinfer) project.

# TODO
- [x] support batch size > 1
- [x] support multi-GPU inference
- [] support GQA
