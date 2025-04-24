"""
Basic Transformer model in PyTorch for sequence classification.
"""

import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=128, num_heads=4, num_classes=2, max_len=128):
        super(SimpleTransformer, self).__init__()

        # Token embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        # Multi-head self-attention layer
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Final classifier head
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        # Get batch size and sequence length
        batch_size, seq_len = input_ids.size()

        # Create position indices (0 to seq_len - 1)
        position_ids = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        # Combine token and position embeddings
        x = self.embedding(input_ids) + self.position_embedding(position_ids)

        # Apply multi-head self-attention
        attn_output, _ = self.attention(x, x, x)

        # Pass through feed-forward layers
        x = self.ffn(attn_output)

        # Pooling: mean over the sequence length
        x = x.mean(dim=1)

        # Classification output
        return self.classifier(x)
