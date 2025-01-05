# Args ref from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/configs/config_671B.json

class Args:
    def __init__(self):
        self.vocab_size = 129280
        self.dim = 7168
        self.inter_dim = 18432
        self.moe_inter_dim = 2048
        self.n_layers = 61
        self.n_dense_layers = 3
        self.n_heads = 128
        self.n_routed_experts = 256
        self.n_shared_experts = 1
        self.n_activated_experts = 8
        self.n_expert_groups = 8
        self.n_limited_groups = 4
        self.route_scale = 2.5
        self.score_func = "sigmoid"
        self.q_lora_rank = 1536
        self.kv_lora_rank = 512
        self.qk_nope_head_dim = 128
        self.qk_rope_head_dim = 64
        self.v_head_dim = 128
        self.dtype = "fp8"

args = Args()

def cal_embed_fwd_flops(bs: int, seq_len: int):
    return 2 * bs * seq_len * args.dim

def cal_head_fwd_flops(bs: int, seq_len: int):
    return 2 * bs * seq_len * args.dim * args.vocab_size

def cal_mla_fwd_flops(bs: int, seq_len: int):
    flops = 0

    # 192
    args.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim

    # Q down + up
    # q = self.wq_b(self.q_norm(self.wq_a(x))) -> [bs, seq_len, (args.n_heads * args.qk_head_dim)]
    flops += 2 * bs * seq_len * args.dim * args.q_lora_rank
    flops += 2 * bs * seq_len * args.q_lora_rank * args.n_heads * args.qk_head_dim

    # KV down
    # kv = self.wkv_a(x) -> [bs, seq_len, (args.kv_lora_rank + args.qk_rope_head_dim)]
    flops += 2 * bs * seq_len * args.dim * (args.kv_lora_rank + args.qk_rope_head_dim)

    # KV up
    # kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    # kv = self.wkv_b(self.kv_norm(kv)) -> [bs, seq_len, (args.n_heads * args.qk_head_dim)]
    flops += 2 * bs * seq_len * args.kv_lora_rank * args.n_heads * (args.qk_nope_head_dim + args.v_head_dim)

    # score = Q x K^T /2 double to causal
    # scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale -> [bs, seq_len, n_heads, (args.qk_nope_head_dim + args.qk_rope_head_dim)]
    flops += 2 * bs * seq_len * args.n_heads * args.qk_head_dim * seq_len / 2

    # score x V
    # x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
    flops += 2 * bs * seq_len * args.n_heads * args.qk_head_dim * args.v_head_dim

    # x = self.wo(x.flatten(2))
    flops += 2 * bs * seq_len * args.n_heads * args.v_head_dim * args.dim

    return flops

def cal_moe_fwd_flops(bs: int, seq_len: int):

    flops = 0

    flops += 2 * bs * seq_len * args.dim * args.moe_inter_dim * 3
    flops += 2 * bs * seq_len * args.moe_inter_dim

    return flops

def cal_mlp_fwd_flops(bs: int, seq_len: int):
    flops = 2 * bs * seq_len * args.dim * args.inter_dim * 3
    flops += 2 * bs * seq_len * args.inter_dim
    return flops

def cal_fwd_flops(bs: int, seq_len: int):
    """
        flops (TFLOPS) per token
    """

    shard_expert_num = 1
    routed_expert_num = 8

    flops_mla = cal_mla_fwd_flops(bs, seq_len) / seq_len / bs / (1024*1024*1024) * args.n_layers
    flops_moe = (shard_expert_num + routed_expert_num) * cal_moe_fwd_flops(bs, seq_len) / seq_len / bs / (1024*1024*1024) * (args.n_layers - args.n_dense_layers)
    flops_mlp = cal_mlp_fwd_flops(bs, seq_len) / seq_len / bs / (1024*1024*1024) * args.n_dense_layers


    flops_embed = cal_embed_fwd_flops(bs, seq_len) / seq_len / bs / (1024*1024*1024)
    flops_head = cal_head_fwd_flops(bs, seq_len) / seq_len / bs / (1024*1024*1024)

    print(f"flops_mla: {flops_mla} TFLOPS, flops_moe: {flops_moe} TFLOPS")

    # calculate MoE FLOPS from parameter numbers
    # MOE_act_param = args.dim * args.moe_inter_dim * 3 * args.n_layers * (shard_expert_num + routed_expert_num)
    # MOE_ref_flops = 2 * MOE_act_param / (1024*1024*1024)

    # print(f"MOE_ref_flops: {MOE_ref_flops} TFLOPS, MOE_act_param: {MOE_act_param / (1024*1024*1024)} B")

    flops = flops_mla + flops_moe + flops_mlp + flops_embed + flops_head
    
    print(f"flops: {flops} TFLOPS")
    return flops


bsz = 32

# pre-training context length 4K
seq_len = 1024 * 4
H100_peak_bf16_flops = 989.5

fwd_flops = cal_fwd_flops(bsz, seq_len)
bwd_flops = fwd_flops * 2

MFU = (fwd_flops + bwd_flops) * 14.8 / (2.788 * 3600 / 1024 * H100_peak_bf16_flops)

print(f"MFU: {MFU}")

# estimate MFU from parameter numbers
attn_flosp = 6 * (args.n_heads * args.qk_head_dim * seq_len / 2 + args.n_heads * args.qk_head_dim * args.v_head_dim) / (1024**3)
MFU_ref = (37*6 + attn_flosp) * 14.8 / (2.788 * 3600 / 1024 * H100_peak_bf16_flops)
print(f"ref MFU: {MFU_ref}")

