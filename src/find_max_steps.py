from omegaconf import OmegaConf

# === Load the YAML config ===
cfg = OmegaConf.load("best_config.yaml")

# === Extract values ===
n_embd = cfg.model_config.n_embd
n_layer = cfg.model_config.n_layer
n_head = cfg.model_config.n_head
seq_len = cfg.seq_len
batch_size = cfg.batch_size
grad_accum = cfg.grad_accumulation_steps

# === FLOP estimation function ===
def estimate_flops(n_embd, n_layer, n_head, seq_len, batch_size, steps, grad_accum):
    # Simplified proxy for transformer training FLOPs per step
    flops_per_token = 6 * n_layer * n_embd ** 2
    return flops_per_token * seq_len * batch_size * grad_accum * steps

# === Binary search to find max steps under a FLOP budget ===
def find_max_steps(max_flops, n_embd, n_layer, n_head, seq_len, batch_size, grad_accum):
    low, high = 100, 20000  # reasonable search bounds
    while low < high:
        mid = (low + high + 1) // 2
        flops = estimate_flops(n_embd, n_layer, n_head, seq_len, batch_size, mid, grad_accum)
        if flops <= max_flops:
            low = mid  # can afford more steps
        else:
            high = mid - 1  # too expensive
    return low

# === Set FLOP budget and find max steps ===
MAX_FLOPS = 1e17
max_steps = find_max_steps(MAX_FLOPS, n_embd, n_layer, n_head, seq_len, batch_size, grad_accum)

# === Output results ===
print(f"Model: embd={n_embd}, layer={n_layer}, head={n_head}, seq={seq_len}, bs={batch_size}, acc={grad_accum}")
print(f"Max steps under {MAX_FLOPS:.1e} FLOPs ≈ {max_steps}")

# Optionally compute FLOPs explicitly
final_flops = estimate_flops(n_embd, n_layer, n_head, seq_len, batch_size, max_steps, grad_accum)
print(f"Estimated FLOPs at {max_steps} steps: {final_flops:.2e}")