/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <flashinfer/prefill_attention_decl.cuh>

#include "flashinfer_ops_prefill.h"
#include "pytorch_extension_utils.h"
#include <c10/cuda/CUDAGuard.h>

using namespace flashinfer;

std::vector<torch::Tensor> single_prefill_with_kv_cache(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor tmp, bool causal,
    unsigned int layout, unsigned int pos_encoding_mode, bool allow_fp16_qk_reduction,
    int32_t window_left, float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta,
    bool return_lse) {
  CHECK_CUDA(q);
  CHECK_CUDA(k);
  CHECK_CUDA(v);
  CHECK_INPUT(tmp);
  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_EQ(v.device(), device);
  CHECK_EQ(tmp.device(), device);
  CHECK_DIM(3, q);
  CHECK_DIM(3, k);
  CHECK_DIM(3, v);
  CHECK_SHAPE(k, v);
  CHECK_EQ(q.stride(2), 1);
  CHECK_EQ(k.stride(2), 1);
  CHECK_EQ(v.stride(2), 1);
  CHECK_EQ(q.size(2), k.size(2));
  CHECK_EQ(q.scalar_type(), k.scalar_type());
  CHECK_EQ(q.scalar_type(), v.scalar_type());
  unsigned int head_dim = q.size(2);
  unsigned int kv_len, qo_len, num_kv_heads, num_qo_heads;
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  qo_len = q.size(0);
  num_qo_heads = q.size(1);
  uint32_t q_stride_n = q.stride(0), q_stride_h = q.stride(1), kv_stride_n, kv_stride_h;
  if (kv_layout == QKVLayout::kNHD) {
    kv_len = k.size(0);
    num_kv_heads = k.size(1);
    kv_stride_n = k.stride(0);
    kv_stride_h = k.stride(1);
  } else {  // QKVLayout::kHND
    kv_len = k.size(1);
    num_kv_heads = k.size(0);
    kv_stride_h = k.stride(0);
    kv_stride_n = k.stride(1);
  }
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto o = torch::empty_like(q, q.options());
  torch::Tensor lse = torch::empty({0});
  if (return_lse) {
    lse = torch::empty({qo_len, num_qo_heads}, q.options().dtype(torch::kFloat32));
  }

  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");
  const LogitsPostHook logits_post_hook =
      logits_soft_cap > 0.f ? LogitsPostHook::kSoftCap : LogitsPostHook::kNone;

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = k.scalar_type();
  TORCH_CHECK(q_scalar_type == kv_scalar_type,
              "q and k must have the same scalar type, but got q: ", q_scalar_type,
              " and k: ", kv_scalar_type);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q_scalar_type, c_type, [&] {
    return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
      return DISPATCH_mask_mode(mask_mode, MASK_MODE, [&] {
        return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
          return DISPATCH_allow_fp16_qk_reduction(
              allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, [&] {
                return DISPATCH_pos_encoding_mode(
                    PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                      cudaError_t status =
                          SinglePrefillWithKVCacheDispatched<HEAD_DIM, LOGITS_POST_HOOK,
                                                             POS_ENCODING_MODE,
                                                             ALLOW_FP16_QK_REDUCTION, MASK_MODE>(
                              static_cast<c_type*>(q.data_ptr()),
                              static_cast<c_type*>(k.data_ptr()),
                              static_cast<c_type*>(v.data_ptr()),
                              /*custom_mask=*/nullptr, static_cast<c_type*>(o.data_ptr()),
                              static_cast<c_type*>(tmp.data_ptr()),
                              /*lse=*/return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr,
                              num_qo_heads, num_kv_heads, qo_len, kv_len, q_stride_n, q_stride_h,
                              kv_stride_n, kv_stride_h, window_left, logits_soft_cap, sm_scale,
                              rope_scale, rope_theta, torch_current_stream);
                      TORCH_CHECK(status == cudaSuccess,
                                  "SinglePrefillWithKVCache kernel launch failed, error: " +
                                      std::string(cudaGetErrorString(status)));
                      return true;
                    });
              });
        });
      });
    });
  });

  if (return_lse) {
    return {o, lse};
  } else {
    return {o};
  }
}

torch::Tensor moa_prefill(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor num_global_blocks, torch::Tensor num_band_blocks, 
    bool causal,
    unsigned int layout, bool allow_fp16_qk_reduction,
    int32_t window_left, float sm_scale) {
  CHECK_CUDA(q);
  CHECK_CUDA(k);
  CHECK_CUDA(v);
  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_EQ(v.device(), device);
  CHECK_EQ(num_global_blocks.device(), device);
  CHECK_EQ(num_band_blocks.device(), device);
  CHECK_DIM(4, q);
  CHECK_DIM(4, k);
  CHECK_DIM(4, v);
  CHECK_DIM(1, num_global_blocks);
  CHECK_DIM(1, num_band_blocks);
  CHECK_SHAPE(k, v);
  // CHECK_EQ(q.stride(3), 1);
  CHECK_CONTIGUOUS(q)
  CHECK_EQ(k.stride(3), 1);
  CHECK_EQ(v.stride(3), 1);
  CHECK_EQ(q.size(3), k.size(3));
  CHECK_EQ(q.scalar_type(), k.scalar_type());
  CHECK_EQ(q.scalar_type(), v.scalar_type());
  CHECK_EQ(num_global_blocks.scalar_type(), torch::kLong);
  CHECK_EQ(num_band_blocks.scalar_type(), torch::kLong);
  unsigned int head_dim = q.size(3);
  unsigned int bz = q.size(0);
  unsigned int kv_len, qo_len, num_kv_heads, num_qo_heads;
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  qo_len = q.size(1);
  num_qo_heads = q.size(2);
  CHECK_EQ(num_global_blocks.size(0), num_qo_heads);
  CHECK_EQ(num_band_blocks.size(0), num_qo_heads);
  uint32_t q_stride_bz = q.stride(0), kv_stride_bz = k.stride(0);
  uint32_t q_stride_n = q.stride(1), q_stride_h = q.stride(2), kv_stride_n, kv_stride_h;
  if (kv_layout == QKVLayout::kNHD) {
    kv_len = k.size(1);
    num_kv_heads = k.size(2);
    kv_stride_n = k.stride(1);
    kv_stride_h = k.stride(2);
  } else {  // QKVLayout::kHND
    kv_len = k.size(2);
    num_kv_heads = k.size(1);
    kv_stride_h = k.stride(1);
    kv_stride_n = k.stride(2);
  }

  const at::cuda::OptionalCUDAGuard device_guard(device);

  
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  cudaStream_t torch_current_stream = nullptr;
  auto o = torch::empty_like(q, q.options());

  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = k.scalar_type();
  TORCH_CHECK(q_scalar_type == kv_scalar_type,
              "q and k must have the same scalar type, but got q: ", q_scalar_type,
              " and k: ", kv_scalar_type);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q_scalar_type, c_type, [&] {
    return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
      return DISPATCH_mask_mode(mask_mode, MASK_MODE, [&] {
          return DISPATCH_allow_fp16_qk_reduction(
              allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, [&] {
                  cudaError_t status = 
                          PrefillMoADispatched<HEAD_DIM, 
                                                ALLOW_FP16_QK_REDUCTION, MASK_MODE>(
                              static_cast<c_type*>(q.data_ptr()),
                              static_cast<c_type*>(k.data_ptr()),
                              static_cast<c_type*>(v.data_ptr()),
                              static_cast<long*>(num_global_blocks.data_ptr()),
                              static_cast<long*>(num_band_blocks.data_ptr()),
                              /*custom_mask*/nullptr, static_cast<c_type*>(o.data_ptr()),
                              num_qo_heads, num_kv_heads, qo_len, kv_len, bz,
                              q_stride_bz, q_stride_n, q_stride_h,
                              kv_stride_bz, kv_stride_n, kv_stride_h, window_left, sm_scale, torch_current_stream);
                  TORCH_CHECK(status == cudaSuccess,
                              "PrefillMoA kernel launch failed, error: " +
                                  std::string(cudaGetErrorString(status)));
                  return true;
          });
      });
    });
  });

  return o;

}

std::vector<torch::Tensor> single_prefill_with_kv_cache_custom_mask(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor packed_custom_mask,
    torch::Tensor tmp, unsigned int layout, unsigned int pos_encoding_mode,
    bool allow_fp16_qk_reduction, int32_t window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, bool return_lse) {
  CHECK_CUDA(q);
  CHECK_CUDA(k);
  CHECK_CUDA(v);
  CHECK_INPUT(packed_custom_mask);
  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_EQ(v.device(), device);
  CHECK_EQ(packed_custom_mask.device(), device);
  CHECK_DIM(3, q);
  CHECK_DIM(3, k);
  CHECK_DIM(3, v);
  CHECK_DIM(1, packed_custom_mask);
  CHECK_SHAPE(k, v);
  CHECK_EQ(q.stride(2), 1);
  CHECK_EQ(k.stride(2), 1);
  CHECK_EQ(v.stride(2), 1);
  CHECK_EQ(q.size(2), k.size(2));
  // packed_custom_mask must be uint8
  TORCH_CHECK(packed_custom_mask.scalar_type() == torch::kUInt8,
              "packed_custom_mask must be uint8");
  unsigned int head_dim = q.size(2);
  unsigned int kv_len, qo_len, num_kv_heads, num_qo_heads;
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  qo_len = q.size(0);
  num_qo_heads = q.size(1);
  uint32_t q_stride_n = q.stride(0), q_stride_h = q.stride(1), kv_stride_n, kv_stride_h;
  if (kv_layout == QKVLayout::kNHD) {
    kv_len = k.size(0);
    num_kv_heads = k.size(1);
    kv_stride_n = k.stride(0);
    kv_stride_h = k.stride(1);
  } else {
    kv_len = k.size(1);
    num_kv_heads = k.size(0);
    kv_stride_h = k.stride(0);
    kv_stride_n = k.stride(1);
  }
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto o = torch::empty_like(q, q.options());
  torch::Tensor lse = torch::empty({0});
  if (return_lse) {
    lse = torch::empty({qo_len, num_qo_heads}, q.options().dtype(torch::kFloat32));
  }

  constexpr MaskMode MASK_MODE = MaskMode::kCustom;
  TORCH_CHECK(logits_soft_cap >= 0.f, "logits_soft_cap must be non-negative");
  const LogitsPostHook logits_post_hook =
      logits_soft_cap > 0.f ? LogitsPostHook::kSoftCap : LogitsPostHook::kNone;

  auto q_scalar_type = q.scalar_type();
  auto kv_scalar_type = k.scalar_type();
  TORCH_CHECK(q_scalar_type == kv_scalar_type,
              "q and k must have the same scalar type, but got q: ", q_scalar_type,
              " and k: ", kv_scalar_type);

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q.scalar_type(), c_type, [&] {
    return DISPATCH_head_dim(head_dim, HEAD_DIM, [&] {
      return DISPATCH_logits_post_hook(logits_post_hook, LOGITS_POST_HOOK, [&] {
        return DISPATCH_allow_fp16_qk_reduction(
            allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, [&] {
              return DISPATCH_pos_encoding_mode(
                  PosEncodingMode(pos_encoding_mode), POS_ENCODING_MODE, [&] {
                    cudaError_t status =
                        SinglePrefillWithKVCacheDispatched<HEAD_DIM, LOGITS_POST_HOOK,
                                                           POS_ENCODING_MODE,
                                                           ALLOW_FP16_QK_REDUCTION, MASK_MODE>(
                            static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
                            static_cast<c_type*>(v.data_ptr()),
                            static_cast<uint8_t*>(packed_custom_mask.data_ptr()),
                            static_cast<c_type*>(o.data_ptr()),
                            static_cast<c_type*>(tmp.data_ptr()),
                            /*lse=*/return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr,
                            num_qo_heads, num_kv_heads, qo_len, kv_len, q_stride_n, q_stride_h,
                            kv_stride_n, kv_stride_h, window_left, logits_soft_cap, sm_scale,
                            rope_scale, rope_theta, torch_current_stream);
                    TORCH_CHECK(status == cudaSuccess,
                                "SinglePrefillWithKVCache kernel launch failed, error: " +
                                    std::string(cudaGetErrorString(status)));
                    return true;
                  });
            });
      });
    });
  });

  if (return_lse) {
    return {o, lse};
  } else {
    return {o};
  }
}
