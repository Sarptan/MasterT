"""
load_llm_backbone.py
--------------------
Loads a pretrained Gemma LLM to initialise the STARFlow deep top block.

This is the OPTIONAL alternative to using a frozen FLAN-T5-XL text encoder.
Instead of a separate encoder, the Gemma backbone IS the top block —
its weights are loaded, then finetuned end-to-end as part of the flow.

Gu et al. (2025, §3.2 / §4.1):
    "We also train a variant where the deep block is initialized from a
     pretrained LLM (Gemma2 in this case), without additional text encoder.
     Our image generator can be directly integrated into any LLM's
     semantic space, eliminating the need for a separate text encoder."

Source in model_setup.py:
    local_path = pathlib.Path(args.logdir) / model_name / 'gemma_meta_block.pth'
    model.blocks[-1].load_state_dict(torch.load(local_path), strict=False)

Supported backbones (from transformer_starflow.py configs):
    "gemma2_2b"   → google/gemma-2-2b-it   (channels=2304, heads=8, kv_heads=4)
    "gemma3_1b"   → google/gemma-3-1b-it   (channels=1152, heads=4, kv_heads=1)
    "gemma3_4b"   → google/gemma-3-4b-it   (channels=2560, heads=8, kv_heads=4)

Requirements:
    pip install transformers accelerate
    huggingface-cli login     ← Gemma requires accepting licence on HuggingFace

Note: Gemma models are ~5–9 GB. Ensure you have enough disk space and RAM.
For CPU-only machines, loading in bfloat16 is essential to avoid OOM.
"""

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Maps the STARFlow config name to the HuggingFace repo ID
GEMMA_REPOS = {
    "gemma2_2b": "google/gemma-2-2b-it",
    "gemma3_1b": "google/gemma-3-1b-it",
    "gemma3_4b": "google/gemma-3-4b-it",
}

# Architecture kwargs that must match the MetaBlock constructor in
# transformer_starflow.py (see gemma2_2b_kwargs etc. at bottom of that file)
GEMMA_METABLOCK_KWARGS = {
    "gemma2_2b": dict(
        channels=2304, num_heads=8, num_kv_heads=4, head_dim=256,
        num_layers=26, use_rope=True, hf_style_rope=True,
        use_swiglu=True, use_qk_norm=False, use_post_norm=True,
        use_final_norm=True, use_bias=False, norm_type="rms_norm",
    ),
    "gemma3_1b": dict(
        channels=1152, num_heads=4, num_kv_heads=1, head_dim=256,
        num_layers=26, expansion=6, use_rope=True, hf_style_rope=True,
        use_swiglu=True, use_qk_norm=True, use_post_norm=True,
        use_final_norm=True, use_bias=False, norm_type="rms_norm",
    ),
    "gemma3_4b": dict(
        channels=2560, num_heads=8, num_kv_heads=4, head_dim=256,
        num_layers=34, use_rope=True, hf_style_rope=True,
        use_swiglu=True, use_qk_norm=True, use_post_norm=True,
        use_final_norm=True, use_bias=False, norm_type="rms_norm",
    ),
}


def load_gemma_config(model_name: str = "gemma2_2b"):
    """
    Load only the config and tokenizer — no weights downloaded yet.
    Useful for inspecting architecture details before committing to the
    full download.

    Returns
    -------
    config    : HuggingFace model config
    tokenizer : text tokenizer
    """
    repo_id = GEMMA_REPOS[model_name]
    print(f"Loading config for {repo_id} ...")
    config    = AutoConfig.from_pretrained(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    print(f"  Hidden size  : {config.hidden_size}")
    print(f"  Num layers   : {config.num_hidden_layers}")
    return config, tokenizer


def load_gemma_backbone(
    model_name: str = "gemma2_2b",
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Load the full Gemma model for use as the STARFlow deep top block.

    The weights are loaded in bfloat16 by default to reduce memory usage.
    During finetuning as part of the flow, the model is updated end-to-end
    with the MLE objective — unlike the text encoder, this is NOT frozen.

    Parameters
    ----------
    model_name : one of "gemma2_2b", "gemma3_1b", "gemma3_4b"
    device     : 'cuda' or 'cpu'
    dtype      : torch.bfloat16 recommended for memory efficiency

    Returns
    -------
    model     : loaded Gemma model (to be used as top block initialisation)
    tokenizer : corresponding tokenizer
    kwargs    : MetaBlock constructor kwargs matching this backbone
    """
    repo_id = GEMMA_REPOS[model_name]
    print(f"Loading {model_name} backbone from {repo_id} ...")
    print(f"  (First run will download ~5-9 GB)")

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    # Starts trainable — will be finetuned as part of the flow
    model.requires_grad_(True)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters     : {params/1e9:.2f}B")
    print(f"  dtype          : {dtype}")
    print(f"  Status         : trainable (finetuned as top block)")

    kwargs = GEMMA_METABLOCK_KWARGS[model_name]
    print(f"\n  MetaBlock kwargs to pass to Model(use_pretrained_lm='{model_name}', ...):")
    for k, v in kwargs.items():
        print(f"    {k} = {v}")

    return model, tokenizer, kwargs


def print_backbone_options():
    """Print a summary of all available Gemma backbone options."""
    print("\nAvailable Gemma backbones for STARFlow top block:")
    print(f"  {'Name':<12}  {'HuggingFace ID':<28}  {'channels':>9}  {'layers':>7}  {'Size':>6}")
    print("  " + "-" * 68)
    specs = [
        ("gemma2_2b",  "google/gemma-2-2b-it",  2304, 26, "~5.4GB"),
        ("gemma3_1b",  "google/gemma-3-1b-it",  1152, 26, "~2.4GB"),
        ("gemma3_4b",  "google/gemma-3-4b-it",  2560, 34, "~9.0GB"),
    ]
    for name, repo, ch, layers, size in specs:
        print(f"  {name:<12}  {repo:<28}  {ch:>9}  {layers:>7}  {size:>6}")


if __name__ == "__main__":
    print_backbone_options()

    # Load just the config first (no large download)
    config, tokenizer = load_gemma_config("gemma2_2b")
    print(f"\nConfig loaded. Hidden size: {config.hidden_size}")
    print("To load full weights, call: load_gemma_backbone('gemma2_2b')")
    print("Note: requires huggingface-cli login and ~5.4 GB disk space.")
