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
    "batch_size": 8,
    "grad_accumulation_steps": 1,
    "num_training_steps": 1000,
    "min_lr": 1e-4,
    "max_lr": 1e-3,
    "num_warmup_steps": 100,
    "weight_decay": 0.01,
    "model_config": {
        "n_embd": 128,
        "n_head": 4,
        "n_layer": 4,
        "n_positions": 128
    }
}

# Define the hyperparameter grid
n_embd_values = [128, 192, 256]
n_layer_values = [3, 4]
n_head_values = [4, 6, 8]
num_training_steps_values = [800, 1000]
batch_size_values = [8, 16]

# Directory for configs and results
os.makedirs("configs", exist_ok=True)
os.makedirs("output", exist_ok=True)

results = []

for n_embd, n_layer, n_head, steps, batch in product(
    n_embd_values, n_layer_values, n_head_values, num_training_steps_values, batch_size_values
):
    cfg = base_config.copy()
    cfg["model_config"] = cfg["model_config"].copy()

    cfg["model_config"]["n_embd"] = n_embd
    cfg["model_config"]["n_layer"] = n_layer
    cfg["model_config"]["n_head"] = n_head
    cfg["num_training_steps"] = steps
    cfg["batch_size"] = batch
    cfg["output_dir"] = f"outputs/embd{n_embd}_layer{n_layer}_head{n_head}_bs{batch}_steps{steps}"

    os.makedirs(cfg["output_dir"], exist_ok=True)

    cfg_path = f"configs/embd{n_embd}_layer{n_layer}_head{n_head}_bs{batch}_steps{steps}.yaml"
    OmegaConf.save(cfg, cfg_path)

    print(f"\n=== Running config: {cfg_path} ===")
    cmd = ["python", "train.py", cfg_path]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"⚠️ Skipping {cfg_path} due to error")
        continue

    # Read evaluation results
    eval_path = os.path.join(cfg["output_dir"], "eval.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_results = json.load(f)
        val_ppl = eval_results.get("val-perplexity", None)
    else:
        val_ppl = None

    # Parse FLOPs from printed output (if you modify train.py to log it to file)
    flops = None

    results.append({
        "config": cfg_path,
        "n_embd": n_embd,
        "n_layer": n_layer,
        "n_head": n_head,
        "batch_size": batch,
        "steps": steps,
        "val_perplexity": val_ppl,
        "FLOPs": flops,
    })

# Sort by best (lowest) perplexity
results = sorted(results, key=lambda x: x["val_perplexity"] or 9999)

# Print summary
print("\n=== Summary of Results ===")
for r in results:
    print(
        f"{r['config']}: "
        f"PPL={r['val_perplexity']}, "
        f"n_embd={r['n_embd']}, n_layer={r['n_layer']}, "
        f"n_head={r['n_head']}, bs={r['batch_size']}, steps={r['steps']}"
    )

# Save results to JSON
with open("hparam_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDone. Results saved to hparam_results.json")
