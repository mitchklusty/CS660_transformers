import torch
import io
from datasets import load_dataset
from transformers import AutoTokenizer

def determine_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def estimate_model_disk_size(model: torch.nn.Module) -> int:
    with io.BytesIO() as byte_stream:
        torch.save(model.state_dict(), byte_stream)
        return byte_stream.tell()


# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def enable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True




def get_text_column(ds):
    # try common names, else first string column
    for cand in ["text", "content", "body", "article", "abstract"]:
        if cand in ds["train"].column_names:
            return cand
    # fallback: pick the first column with string values
    for name in ds["train"].column_names:
        if isinstance(ds["train"][0][name], str):
            return name
    raise ValueError("No text column found.")

def tokenize_dataset(name="jamescalam/ai-arxiv", tokenizer_name="gpt2"):
    ds = load_dataset(name)
    text_col = get_text_column(ds)
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    # avoid the 1024 warning; we’ll chunk ourselves
    tok.model_max_length = 1_000_000
    if tok.eos_token is None:
        tok.add_special_tokens({"eos_token": ""})  # rarely needed

    def tok_fn(batch):
        return tok(batch[text_col], add_special_tokens=False, return_attention_mask=False)

    ds = ds.map(tok_fn, batched=True, remove_columns=ds["train"].column_names)
    return ds, tok

def concat_with_eos(list_of_lists, eos_id):
    # flatten docs with EOS separators
    total = []
    for ids in list_of_lists:
        total.extend(ids)
        total.append(eos_id)
    return torch.tensor(total, dtype=torch.long)

def dataset2tokens(name="jamescalam/ai-arxiv", tokenizer_name="gpt2", seq_len=128):
    ds, tok = tokenize_dataset(name, tokenizer_name)

    # Some splits are named "validation", others "val"; handle both.
    val_key = "validation" if "validation" in ds else ("val" if "val" in ds else None)
    if val_key is None:
        # create a split if only train exists
        ds = ds["train"].train_test_split(test_size=0.1, seed=42)
        train_ids_list = ds["train"]["input_ids"]
        val_ids_list   = ds["test"]["input_ids"]
    else:
        train_ids_list = ds["train"]["input_ids"]
        val_ids_list   = ds[val_key]["input_ids"]

    eos_id = tok.eos_token_id if tok.eos_token_id is not None else 50256  # GPT-2 EOS

    train_tokens = concat_with_eos(train_ids_list, eos_id)
    val_tokens   = concat_with_eos(val_ids_list,   eos_id)

    '''
    train_tokens = make_blocks(train_flat, seq_len)  # shape: [N_train_blocks, 128]
    val_tokens   = make_blocks(val_flat,   seq_len)  # shape: [N_val_blocks, 128]

    print("train_tokens:", train_tokens.shape, train_tokens.dtype)
    print("val_tokens:",   val_tokens.shape,   val_tokens.dtype)
    '''

    # If your training loop expects attention_mask, we can use all ones:
    train_masks = torch.ones_like(train_tokens, dtype=torch.float32)
    val_masks   = torch.ones_like(val_tokens,   dtype=torch.float32)

    return train_tokens, val_tokens, train_masks, val_masks
