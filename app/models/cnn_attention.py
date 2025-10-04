# models/cnn_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Simple Conv1D -> BatchNorm -> ReLU -> (optional) MaxPool1d"""
    def __init__(self, in_ch, out_ch, kernel_size=5, pool=True):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool1d(kernel_size=2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, L]
        return self.net(x)


class CNN_Attention(nn.Module):
    """
    CNN backbone -> project to embedding dim -> MultiHeadAttention -> classifier
    Input: x [B, C_in, L]  (C_in usually 1 or 3)
    Output: logits [B]  (use BCEWithLogitsLoss)
    """
    def __init__(
        self,
        in_channels: int = 1,
        seq_len: int = 2000,
        conv_channels: tuple[int, ...] = (16, 32, 64),
        kernel_size: int = 7,
        attn_embed_dim: int = 64,
        num_heads: int = 4,
        attn_dropout: float = 0.0,
        fc_dropout: float = 0.3
    ):
        super().__init__()
        # --- CNN backbone ---
        blocks = []
        ch_in = in_channels
        for i, ch_out in enumerate(conv_channels):
            # pool at every block except maybe last: we'll pool to reduce seq length
            pool = True
            blocks.append(ConvBlock(ch_in, ch_out, kernel_size=kernel_size, pool=pool))
            ch_in = ch_out
        self.cnn = nn.Sequential(*blocks)

        # compute downsampled sequence length after pools (each pool halves length)
        n_pools = len(conv_channels)  # because we used pool=True every block
        down_len = seq_len // (2 ** n_pools)
        if down_len < 1:
            raise ValueError(f"seq_len {seq_len} too short for {n_pools} pooling layers")

        # project CNN channels to attention embedding size if needed
        last_cnn_ch = conv_channels[-1]
        if last_cnn_ch != attn_embed_dim:
            self.project = nn.Conv1d(last_cnn_ch, attn_embed_dim, kernel_size=1)
        else:
            self.project = nn.Identity()

        # --- Attention block (uses batch_first=True) ---
        # MultiheadAttention expects [B, L, E] when batch_first=True
        self.attn = nn.MultiheadAttention(embed_dim=attn_embed_dim, num_heads=num_heads,
                                          dropout=attn_dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(attn_embed_dim)
        # small feed-forward in attention block (like transformer FFN)
        self.ffn = nn.Sequential(
            nn.Linear(attn_embed_dim, attn_embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(attn_embed_dim * 2, attn_embed_dim)
        )
        self.ffn_norm = nn.LayerNorm(attn_embed_dim)

        # final classifier
        self.pool = nn.AdaptiveAvgPool1d(1)  # pool over temporal dim after attention (if we permute back)
        self.classifier = nn.Sequential(
            nn.Linear(attn_embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(64, 1)  # output single logit per sample
        )

        # init weights (good default)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B, C_in, L]
        returns logits: [B]
        """
        B = x.shape[0]
        # 1) CNN backbone
        x = self.cnn(x)                      # [B, C_last, L_down]
        x = self.project(x)                  # [B, attn_embed_dim, L_down]
        # prepare for attention: [B, L_down, E]
        x = x.permute(0, 2, 1).contiguous()

        # 2) Self-attention with residual and norm
        attn_out, attn_weights = self.attn(x, x, x)   # attn_out: [B, L_down, E]
        x = self.attn_norm(x + attn_out)             # residual + norm

        # 3) FFN block (transformer-style) with residual
        ffn_out = self.ffn(x)                        # [B, L_down, E]
        x = self.ffn_norm(x + ffn_out)

        # 4) Pool across sequence dimension -> [B, E]
        # use mean pooling across L_down
        x = x.mean(dim=1)                            # [B, E]

        # 5) Classifier -> [B, 1] then flatten to [B]
        logits = self.classifier(x).squeeze(1)
        return logits  # raw logits (use BCEWithLogitsLoss)
    
    def parameters(self, recurse = True):
        return super().parameters(recurse)

    def print_layer_summary(self):
        print(f"{'Layer':30} {'Shape':25} {'Param #':10}")
        print("-" * 70)
        total_params = 0
        for name, param in self.named_parameters():
            param_count = param.numel()
            total_params += param_count
            print(f"{name:30} {str(list(param.shape)):25} {param_count:10}")
        print("-" * 70)
        print(f"Total parameters: {total_params}")