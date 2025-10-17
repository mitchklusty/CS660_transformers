import os
import subprocess
import json
from omegaconf import OmegaConf
from itertools import product

# Base configuration template
base_config = {
    "output_dir": "outputs",
    "device": "auto",
    "tokenizer_encoding": "gpt2",
    "seq_len": 128,
    "batch_size": 32,
    "grad_accumulation_steps": 1,
    "num_training_steps": 2000,
    "min_lr": 1e-4,
    "max_lr": 5e-3,
    "num_warmup_steps": 10,
    "weight_decay": 1e-2,
    "model_config": {
        "n_embd": 32,
        "n_head": 2,
        "n_layer": 4,
        "n_positions": 128
    }
}

# FLOP budget
MAX_FLOPS = 1e15

# Define reasonable hyperparameter ranges
n_embd_values = [32, 48, 64, 80]
n_layer_values = [2, 3, 4]
n_head_values = [2, 3, 4]
seq_len_values = [64, 96, 128]
batch_size_values = [8, 16, 32]
grad_accum_values = [1, 2]
num_training_steps_values = [500, 1000, 1500, 2000]

# Directory setup
os.makedirs("configs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

results = []

# Approximate FLOPs per token 
# FLOPs = model.flops_per_token * num_training_steps * grad_accum_steps * batch_size * seq_len
# model.flops_per_token depends on (n_layer, n_embd, n_head), roughly proportional to 6 * n_layer * n_embd^2
def estimate_flops(n_embd, n_layer, n_head, seq_len, batch_size, steps, grad_accum):
    flops_per_token = 6 * n_layer * n_embd**2  # very rough proxy
    return flops_per_token * steps * grad_accum * batch_size * seq_len

for n_embd, n_layer, n_head, seq_len, batch, grad_accum, steps in product(
    n_embd_values, n_layer_values, n_head_values,
    seq_len_values, batch_size_values, grad_accum_values, num_training_steps_values
):
    est_flops = estimate_flops(n_embd, n_layer, n_head, seq_len, batch, steps, grad_accum)
    if est_flops > MAX_FLOPS:
        print(f"Skipping n_embd={n_embd}, n_layer={n_layer}, n_head={n_head}, bs={batch}, seq={seq_len}, steps={steps} "
              f"({est_flops:.2e} FLOPs > {MAX_FLOPS:.2e})")
        continue

    cfg = base_config.copy()
    cfg["model_config"] = cfg["model_config"].copy()

    cfg["model_config"]["n_embd"] = n_embd
    cfg["model_config"]["n_layer"] = n_layer
    cfg["model_config"]["n_head"] = n_head
    cfg["model_config"]["n_positions"] = max(seq_len, cfg["model_config"]["n_positions"])
    cfg["seq_len"] = seq_len
    cfg["batch_size"] = batch
    cfg["grad_accumulation_steps"] = grad_accum
    cfg["num_training_steps"] = steps

    run_name = f"embd{n_embd}_layer{n_layer}_head{n_head}_bs{batch}_seq{seq_len}_acc{grad_accum}_steps{steps}"
    cfg["output_dir"] = f"outputs/{run_name}"
    os.makedirs(cfg["output_dir"], exist_ok=True)

    cfg_path = f"configs/{run_name}.yaml"
    OmegaConf.save(cfg, cfg_path)

    print(f"\n=== Running config: {cfg_path} (Est. FLOPs={est_flops:.2e}) ===")
    cmd = ["python", "-m" "lm.train", cfg_path]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"⚠️ Skipping {cfg_path} due to error")
        continue

    eval_path = os.path.join(cfg["output_dir"], "eval.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_results = json.load(f)
        val_ppl = eval_results.get("val-perplexity", None)
    else:
        val_ppl = None

    results.append({
        "config": cfg_path,
        "n_embd": n_embd,
        "n_layer": n_layer,
        "n_head": n_head,
        "seq_len": seq_len,
        "batch_size": batch,
        "grad_accum": grad_accum,
        "steps": steps,
        "val_perplexity": val_ppl,
        "FLOPs": est_flops,
    })

# Sort by best (lowest) perplexity
results = sorted(results, key=lambda x: x["val_perplexity"] if x["val_perplexity"] is not None else 9999)

# Keep only top 5
top_results = results[:5]

print("\n=== Summary of Top 5 Results ===")
for r in top_results:
    print(
        f"{r['config']}: PPL={r['val_perplexity']}, FLOPs={r['FLOPs']:.2e}, "
        f"n_embd={r['n_embd']}, n_layer={r['n_layer']}, n_head={r['n_head']}, "
        f"seq_len={r['seq_len']}, bs={r['batch_size']}, acc={r['grad_accum']}, steps={r['steps']}"
    )

# Save results
with open("hparam_results.json", "w") as f:
    json.dump(top_results, f, indent=2)

# Save best config YAML
if top_results:
    best_cfg_path = top_results[0]["config"]
    best_cfg = OmegaConf.load(best_cfg_path)
    OmegaConf.save(best_cfg, "best_config.yaml")
    print(f"\nBest config saved to best_config.yaml")

print("\nDone. Results saved to hparam_results.json")
