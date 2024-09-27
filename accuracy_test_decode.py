import torch
import random
import icecream
import flashinfer

from flash_attn import flash_attn_func

num_heads = 32
seq_len = 16384 * 4

q = torch.randn((1, 1, num_heads, 128), device="cuda:0", dtype=torch.float16)

k_cache = torch.randn((1, num_heads * seq_len, 128), device="cuda:0", dtype=torch.float16)
v_cache = torch.randn((1, num_heads * seq_len, 128), device="cuda:0", dtype=torch.float16)

start = torch.tensor([[i * seq_len for i in range(num_heads)]], dtype=torch.long, device="cuda:0")
length = torch.zeros(1, num_heads, device="cuda:0", dtype=torch.long)

for i in range(num_heads):
    length[0][i] = random.randint(64, seq_len)

# make a copy of k_cache and v_cache
original_k_cache = k_cache.clone().reshape(1, num_heads, seq_len, 128).transpose(1, 2).contiguous()
original_v_cache = v_cache.clone().reshape(1, num_heads, seq_len, 128).transpose(1, 2).contiguous()

h = flashinfer.moa_decode(q, k_cache, v_cache, start, length)

original_h_list = []
for i in range(num_heads):
    original_h = flash_attn_func(q[:, :, i:i+1, :].contiguous(), original_k_cache[:, :length[0][i], i:i+1, :].contiguous(), original_v_cache[:, :length[0][i], i:i+1, :].contiguous())
    original_h_list.append(original_h)

original_h = torch.cat(original_h_list, dim=2)

icecream.ic((h - original_h).abs().max())
