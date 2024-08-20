import torch
import flashinfer
import argparse

parser = argparse.ArgumentParser(description='Test the performance of the flashinfer')
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--token_len", type=int, default=8192, help="token length")
parser.add_argument("--num_heads", type=int, default=32, help="number of heads")
parser.add_argument("--head_dim", type=int, default=128, help="head dimension")
parser.add_argument("--num_warm_up_iters", type=int, default=5, help="number of warm up iterations")
parser.add_argument("--num_iters", type=int, default=10, help="number of iterations")
args = parser.parse_args()

num_iters = args.num_iters
batch_size = args.batch_size
token_len = args.token_len
head_dim = args.head_dim
num_heads = args.num_heads

q_list = [torch.randn(batch_size, token_len, num_heads, head_dim, dtype=torch.float16, device='cuda:0') for _ in range(args.num_iters + 1)]
k_list = [torch.randn(batch_size, token_len, num_heads, head_dim, dtype=torch.float16, device='cuda:0') for _ in range(args.num_iters + 1)]
v_list = [torch.randn(batch_size, token_len, num_heads, head_dim, dtype=torch.float16, device='cuda:0') for _ in range(args.num_iters + 1)]

q_trans_list = [torch.randn(batch_size, num_heads, token_len, head_dim, dtype=torch.float16, device='cuda:0') for _ in range(args.num_iters + 1)]
k_trans_list = [torch.randn(batch_size, num_heads, token_len, head_dim, dtype=torch.float16, device='cuda:0') for _ in range(args.num_iters + 1)]
v_trans_list = [torch.randn(batch_size, num_heads, token_len, head_dim, dtype=torch.float16, device='cuda:0') for _ in range(args.num_iters + 1)]

for _ in range(args.num_warm_up_iters):
    torch.nn.functional.scaled_dot_product_attention(
        q_trans_list[-1],
        k_trans_list[-1],
        v_trans_list[-1],
        is_causal=True
    )

num_global_blocks = torch.tensor([1 for _ in range(num_heads)], dtype=torch.long).to('cuda:0')
num_band_blocks = torch.tensor([1000 for _ in range(num_heads)], dtype=torch.long).to('cuda:0')

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()

for i in range(args.num_iters):
    # torch.nn.functional.scaled_dot_product_attention(
    #     q_trans_list[i],
    #     k_trans_list[i],
    #     v_trans_list[i],
    #     is_causal=True,
    # )
    flashinfer.moa_prefill(q_list[i], k_list[i], v_list[i], causal=True, kv_layout="NHD", num_global_blocks=num_global_blocks, num_band_blocks=num_band_blocks)

    # flashinfer.single_prefill_with_kv_cache(q_list[i].squeeze(0), k_list[i].squeeze(0), v_list[i].squeeze(0), causal=True, kv_layout="NHD")

end_event.record()
torch.cuda.synchronize()
elapsed_time_ms = start_event.elapsed_time(end_event)
print(f"Elapsed time: {elapsed_time_ms} ms")
print(f"per iter: {elapsed_time_ms / num_iters} ms")