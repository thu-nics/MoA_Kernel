# modified from flashinfer v0.1.5
# https://github.com/flashinfer-ai/flashinfer/blob/main/python/generate_single_prefill_inst.py

import sys
import re
from literal_map import (
    dtype_literal,
    mask_mode_literal,
)
from pathlib import Path


def get_cu_file_str(
    head_dim,
    allow_fp16_qk_reduction,
    mask_mode,
    dtype_q,
    dtype_kv,
    dtype_out,
):

    content = """#include <flashinfer/attention_impl.cuh>

namespace flashinfer {{

template cudaError_t PrefillMoADispatched<{head_dim}, {allow_fp16_qk_reduction}, {mask_mode}, {dtype_q}, {dtype_kv}, {dtype_out}>(
    {dtype_q}* q, {dtype_kv}* k, {dtype_kv}* v, long* num_global_blocks, long* num_band_blocks, long* left_padding_lengths,
    uint8_t* custom_mask, {dtype_out}* o,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len, uint32_t bz,
    uint32_t q_stride_bz, uint32_t q_stride_n, uint32_t q_stride_h, uint32_t kv_stride_bz, uint32_t kv_stride_n, uint32_t kv_stride_h, int32_t window_left,
    float sm_scale, cudaStream_t stream);

}}
    """.format(
        head_dim=head_dim,
        allow_fp16_qk_reduction=allow_fp16_qk_reduction,
        mask_mode=mask_mode_literal[int(mask_mode)],
        dtype_q=dtype_literal[dtype_q],
        dtype_kv=dtype_literal[dtype_kv],
        dtype_out=dtype_literal[dtype_out],
    )
    return content


if __name__ == "__main__":
    pattern = (
        r"moa_prefill_head_([0-9]+)_"
        r"fp16qkred_([a-z]+)_mask_([0-9]+)_dtypeq_([a-z0-9]+)_dtypekv_([a-z0-9]+)_dtypeout_([a-z0-9]+)\.cu"
    )

    compiled_pattern = re.compile(pattern)
    path = Path(sys.argv[1])
    fname = path.name
    match = compiled_pattern.match(fname)
    with open(path, "w") as f:
        f.write(get_cu_file_str(*match.groups()))
