"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
import re
from literal_map import (
    pos_encoding_mode_literal,
    dtype_literal,
    logits_hook_literal,
)
from pathlib import Path


def get_cu_file_str(
    head_dim,
    dtype_q,
    dtype_kv,
    dtype_out,
):
    content = """#include <flashinfer/attention_impl.cuh>

namespace flashinfer {{

template cudaError_t SingleDecodeWithKVCacheDispatchedMoA<{head_dim}, {dtype_q}, {dtype_kv}, {dtype_out}>(
    {dtype_q}* q, {dtype_kv}* k, {dtype_kv}* v, {dtype_out}* o,
    uint32_t batch_size, uint32_t num_heads,
    uint32_t q_stride_bz, uint32_t q_stride_h,
    uint32_t kv_stride_bz, uint32_t kv_stride_n,
    uint32_t o_stride_bz, uint32_t o_stride_h,
    long* Start, long* Length,
    float sm_scale,
    cudaStream_t stream);

}}
    """.format(
        head_dim=head_dim,
        dtype_q=dtype_literal[dtype_q],
        dtype_kv=dtype_literal[dtype_kv],
        dtype_out=dtype_literal[dtype_out],
    )
    return content


if __name__ == "__main__":
    pattern = (
        r"moa_decode_head_([0-9]+)_"
        r"dtypeq_([a-z0-9]+)_dtypekv_([a-z0-9]+)_dtypeout_([a-z0-9]+)\.cu"
    )

    compiled_pattern = re.compile(pattern)
    path = Path(sys.argv[1])
    fname = path.name
    match = compiled_pattern.match(fname)

    with open(path, "w") as f:
        f.write(get_cu_file_str(*match.groups()))
