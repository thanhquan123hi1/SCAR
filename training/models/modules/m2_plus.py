"""M2-Plus: Decoupled Cross-Attention Fusion for SCAR.
Separates scar and edema attention streams without feature averaging.
"""
from __future__ import annotations
import torch
from torch import nn

class M2Plus_Fusion(nn.Module):
    """Decoupled Cross-Attention:
    - Branch 1: CINE queries attend to PSIR/LGE (Scar specialist)
    - Branch 2: CINE queries attend to T2W (Edema specialist)
    Combines both streams with CINE to avoid pathology signal dilution.
    """
    def __init__(self, in_channels: int = 1024, out_channels: int = 512, num_heads: int = 8):
        super().__init__()
        if in_channels % num_heads:
            raise ValueError("Channels must be divisible by num_heads")
        
        # 2 bộ Multihead Attention độc lập
        self.mha_scar = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.mha_edema = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        
        # Chiếu gộp 3 nguồn: CINE + Attn_Scar + Attn_Edema -> out_channels (512)
        self.proj = nn.Sequential(
            nn.Conv2d(3 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = cine.shape
        query = cine.flatten(2).transpose(1, 2)
        key_scar = psir.flatten(2).transpose(1, 2)
        key_edema = t2w.flatten(2).transpose(1, 2)

        # Chú ý riêng biệt: không cộng trung bình, không làm suy giảm tín hiệu Sẹo
        attn_scar, _ = self.mha_scar(query, key_scar, key_scar, need_weights=False)
        attn_edema, _ = self.mha_edema(query, key_edema, key_edema, need_weights=False)

        # Đưa về kích thước ban đầu (B, C, H, W)
        attn_scar = attn_scar.transpose(1, 2).reshape(batch, channels, height, width)
        attn_edema = attn_edema.transpose(1, 2).reshape(batch, channels, height, width)

        # Gộp cả 3 đặc trưng đưa vào Decoder
        return self.proj(torch.cat((cine, attn_scar, attn_edema), dim=1))