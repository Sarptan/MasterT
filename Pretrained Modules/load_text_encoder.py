"""
load_text_encoder.py
--------------------
Loads the FLAN-T5-XL text encoder used in STARFlow text-to-image.

Status: FROZEN — never updated during any training stage.

Source in model_setup.py (args.text == 't5xl'):
    tokenizer    = AutoTokenizer.from_pretrained(args.text)
    text_encoder = AutoModel.from_pretrained(
                       args.text, add_cross_attention=False).encoder

Gu et al. (2025, Appendix B.1):
    "Text captions are encoded by a frozen FLAN-T5-XL encoder,
     truncated to 128 tokens."

Key number: txt_dim = text_encoder.config.hidden_size = 2048
This value is passed as txt_dim=2048 to the deep MetaBlock constructor,
which uses it to build the proj_txt linear layer.

Note on download size:
    google/flan-t5-xl   ~9.4 GB   (full model with both encoder + decoder)
    For testing, use    google/flan-t5-base  (~990 MB, hidden_size=768)
    If using base, remember to set txt_dim=768 in your Model constructor.
"""

import torch
from transformers import AutoTokenizer, AutoModel

T5_MODEL_ID  = "google/flan-t5-xl"    # Change to "google/flan-t5-base" for lighter testing
MAX_TXT_LEN  = 128                     # STARFlow truncates to 128 tokens (Appendix B.1)


def load_text_encoder(model_id: str = T5_MODEL_ID, device: str = "cpu"):
    """
    Load the T5 encoder and its tokenizer.

    Returns
    -------
    tokenizer    : HuggingFace tokenizer
    text_encoder : T5 encoder (encoder-only, frozen, eval mode)
    txt_dim      : hidden size (2048 for XL, 768 for base)
    """
    print(f"Loading text encoder from {model_id} ...")
    print(f"  (This may download several GB on first run)")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # We only need the encoder stack — no T5 decoder
    # add_cross_attention=False keeps it clean for encoder-only use
    text_encoder = AutoModel.from_pretrained(
        model_id,
        add_cross_attention=False,
    ).encoder.to(device)

    # Completely frozen — no gradients at any point
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    txt_dim = text_encoder.config.hidden_size
    params  = sum(p.numel() for p in text_encoder.parameters())

    print(f"  txt_dim (hidden_size): {txt_dim}  ← pass this to Model(txt_dim=...)")
    print(f"  Encoder params       : {params/1e6:.0f}M  → frozen")
    print(f"  Max token length     : {MAX_TXT_LEN}")

    return tokenizer, text_encoder, txt_dim


def encode_text(
    tokenizer,
    text_encoder,
    captions: list[str],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Tokenize and encode a list of text captions.

    This replicates what STARFlow does before each training step:
    the caption is tokenized, padded/truncated to 128 tokens, and
    passed through the frozen T5 encoder to get dense embeddings.

    Parameters
    ----------
    captions : list of strings, e.g. ["a dog on a beach", "a red car"]

    Returns
    -------
    embeddings : (B, 128, txt_dim) — one embedding vector per token
    """
    tokens = tokenizer(
        captions,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_TXT_LEN,
    ).to(device)

    with torch.no_grad():
        embeddings = text_encoder(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        ).last_hidden_state

    return embeddings   # (B, 128, txt_dim)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, text_encoder, txt_dim = load_text_encoder(device=device)

    test_captions = [
        "a corgi sitting on a beach at sunset",
        "a futuristic cityscape at night with neon lights",
    ]

    embeddings = encode_text(tokenizer, text_encoder, test_captions, device)

    print(f"\nEncoding test:")
    for i, cap in enumerate(test_captions):
        print(f"  [{i}] '{cap}'")
    print(f"  Output shape: {tuple(embeddings.shape)}")
    print(f"  → (batch={len(test_captions)}, seq=128, txt_dim={txt_dim})")
    print("Text encoder OK.")
