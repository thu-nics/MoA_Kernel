<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/flashinfer-ai/web-data/blob/main/logo/FlashInfer-black-background.png?raw=true">
    <img alt="FlashInfer" src="https://github.com/flashinfer-ai/web-data/blob/main/logo/FlashInfer-white-background.png?raw=true" width=55%>
  </picture>
</p>
<h1 align="center">
Kernel Library for LLM Serving
</h1>

## Getting Started

Using our PyTorch API is the easiest way to get started:

### Installation

Tested on **CUDA 12.4** and **torch 2.4**.

Build from source for moa kernel:

```bash
git checkout moa_kernel
cd flashinfer/python
FLASHINFER_LOGITS_POST_HOOKS=0 FLASHINFER_HEAD_DIMS=128 FLASHINFER_POS_ENCODING_MODES=0 python setup.py install
```

### Trying it out

```python
python accuracy_test.py
```

### TODO
- [x] support batch size > 1
- [] acceleration of prefill kernel
- [] support GQA

## Acknowledgement

FlashInfer is inspired by [FlashAttention 1&2](https://github.com/dao-AILab/flash-attention/), [vLLM](https://github.com/vllm-project/vllm), [stream-K](https://arxiv.org/abs/2301.03598) and [cutlass](https://github.com/nvidia/cutlass) projects.
