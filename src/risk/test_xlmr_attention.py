import torch
from transformers import AutoTokenizer

from xlmr_attention_model import LegalXLMRAttentionRiskModel


MODEL_NAME = "joelniklaus/legal-xlm-roberta-base"


def main():

    print("=" * 70)
    print("TESTING LEGAL XLM-R + MULTI-HEAD ATTENTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("Loading model...")

    model = LegalXLMRAttentionRiskModel(
        model_name=MODEL_NAME,
        num_labels=2,
        num_attention_heads=8,
        dropout=0.2,
    )

    model.to(device)
    model.train()

    # --------------------------------------------------------
    # Dummy legal clauses
    # --------------------------------------------------------

    texts = [
        "The company may terminate this agreement at any time without providing notice.",
        "The parties shall maintain commercially reasonable insurance coverage.",
    ]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    print("\nRunning forward pass...")

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_attention=True,
    )

    logits = outputs["logits"]
    attention_weights = outputs["attention_weights"]
    mha_weights = outputs["mha_weights"]

    print("\nShapes:")
    print(f"  Input IDs:          {input_ids.shape}")
    print(f"  Logits:             {logits.shape}")
    print(f"  Attention weights:  {attention_weights.shape}")
    print(f"  MHA weights:        {mha_weights.shape}")

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    assert logits.shape == (
        len(texts),
        2,
    )

    assert attention_weights.shape[0] == len(texts)

    assert mha_weights.shape[0] == len(texts)

    assert mha_weights.shape[1] == 8

    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    print("\nRunning backward pass...")

    labels = torch.tensor(
        [0, 1],
        dtype=torch.long,
        device=device,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    loss = loss_fn(
        logits,
        labels,
    )

    loss.backward()

    print(f"Loss: {loss.item():.4f}")

    # --------------------------------------------------------
    # Check gradients
    # --------------------------------------------------------

    print("\nChecking gradients...")

    gradient_found = False

    for name, parameter in model.named_parameters():

        if parameter.grad is not None:
            gradient_found = True
            break

    assert gradient_found

    print("✓ Gradients successfully propagated.")

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SANITY TEST PASSED")
    print("=" * 70)

    print("\nModel is ready for training.")


if __name__ == "__main__":
    main()