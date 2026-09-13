import torch
import torch.nn as nn

from transformers import XLMRobertaModel


class LegalXLMRAttentionRiskModel(nn.Module):
    """
    Legal XLM-R + Multi-Head Self-Attention risk classifier.

    Architecture:

        Input text
            ↓
        Legal XLM-R encoder
            ↓
        Multi-Head Self-Attention
            ↓
        Residual Connection + LayerNorm
            ↓
        Learned Attention Pooling
            ↓
        Dropout
            ↓
        Linear Classifier
            ↓
        FAIR / UNFAIR
    """

    def __init__(
        self,
        model_name="joelniklaus/legal-xlm-roberta-base",
        num_labels=2,
        num_attention_heads=8,
        dropout=0.2,
    ):
        super().__init__()

        # ====================================================
        # Legal XLM-R Encoder
        # ====================================================

        self.xlmr = XLMRobertaModel.from_pretrained(
            model_name
        )

        hidden_size = self.xlmr.config.hidden_size

        # ====================================================
        # Task-Specific Multi-Head Self-Attention
        #
        # Legal XLM-R already contains Transformer attention.
        # This is an additional attention layer specifically
        # trained for the unfairness/risk classification task.
        # ====================================================

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ====================================================
        # Residual + LayerNorm
        # ====================================================

        self.attention_norm = nn.LayerNorm(
            hidden_size
        )

        # ====================================================
        # Learned Attention Pooling
        #
        # Learns which token representations are most useful
        # for determining whether the clause is unfair.
        # ====================================================

        self.attention_pool = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_size,
                1
            ),
        )

        # ====================================================
        # Classification Head
        # ====================================================

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_labels
        )

    # ========================================================
    # Forward Pass
    # ========================================================

    def forward(
        self,
        input_ids,
        attention_mask,
        return_attention=False,
    ):

        # ====================================================
        # 1. Legal XLM-R
        # ====================================================

        outputs = self.xlmr(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state

        # Shape:
        # [batch_size, sequence_length, hidden_size]

        # ====================================================
        # 2. Padding Mask
        #
        # MultiheadAttention expects:
        # True  -> ignore token
        # False -> valid token
        # ====================================================

        key_padding_mask = (
            attention_mask == 0
        )

        # ====================================================
        # 3. Multi-Head Self-Attention
        # ====================================================

        attended_states, mha_weights = (
            self.multihead_attention(
                query=hidden_states,
                key=hidden_states,
                value=hidden_states,
                key_padding_mask=key_padding_mask,
                need_weights=return_attention,
                average_attn_weights=False,
            )
        )

        # Shape:
        # attended_states:
        # [batch_size, sequence_length, hidden_size]

        # If requested:
        # mha_weights:
        # [batch_size, num_heads,
        #  sequence_length, sequence_length]

        # ====================================================
        # 4. Residual Connection + LayerNorm
        # ====================================================

        attended_states = self.attention_norm(
            hidden_states + attended_states
        )

        # ====================================================
        # 5. Learned Attention Pooling
        # ====================================================

        attention_scores = self.attention_pool(
            attended_states
        ).squeeze(-1)

        # Shape:
        # [batch_size, sequence_length]

        # ----------------------------------------------------
        # Ignore padding tokens
        # ----------------------------------------------------

        attention_scores = attention_scores.masked_fill(
            attention_mask == 0,
            -1e9
        )

        # ----------------------------------------------------
        # Convert scores into normalized attention weights
        # ----------------------------------------------------

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        # ====================================================
        # 6. Attention-Weighted Clause Representation
        # ====================================================

        pooled_output = torch.sum(
            attended_states
            * attention_weights.unsqueeze(-1),
            dim=1,
        )

        # Shape:
        # [batch_size, hidden_size]

        # ====================================================
        # 7. Dropout
        # ====================================================

        pooled_output = self.dropout(
            pooled_output
        )

        # ====================================================
        # 8. Risk Classification
        # ====================================================

        logits = self.classifier(
            pooled_output
        )

        # Shape:
        # [batch_size, 2]
        #
        # Class 0 = FAIR
        # Class 1 = UNFAIR

        # ====================================================
        # 9. Return
        # ====================================================

        if return_attention:

            return {
                "logits": logits,
                "attention_weights": attention_weights,
                "mha_weights": mha_weights,
            }

        return {
            "logits": logits,
        }