"""M2-Max-Pro: Cooperative Bi-Pathology Co-Attention with Strip Anatomy Envelope Gating.
Resolves zero-sum suppression between Scar and Edema by:
1. Strip Anatomy Envelope Gating (CINE structural prior) locking pathology within myocardium.
2. Mutual Pathology Co-Attention (LGE <-> T2w) capturing infarct core vs ischemic penumbra.
3. Cooperative (non-zero-sum) dual-gated pathology refinement.
"""
from __future__ import annotations

import torch
from torch import nn


class M2MaxPro_Fusion(nn.Module):
    """Cooperative Bi-Pathology Co-Attention Fusion:
    Overcomes the zero-sum competitive limitation of M2-Max.
    """

    def __init__(self, in_channels: int = 1024, out_channels: int = 512, num_heads: int = 8):
        super().__init__()
        if in_channels % num_heads:
            raise ValueError("in_channels must be divisible by num_heads")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        # 1. Strip Anatomy Envelope Gating from CINE
        hidden = max(in_channels // 4, 16)
        self.strip_anatomy = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # 2. Multi-Head Cross-Attention modules
        # CINE queries Scar (LGE) and Edema (T2w)
        self.mha_cine_scar = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.mha_cine_edema = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)

        # Mutual Pathology Co-Attention:
        # Edema attends to Scar (anchoring penumbra around the necrotic core)
        self.mha_edema_to_scar = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        # Scar attends to Edema (verifying acute edema context around the scar)
        self.mha_scar_to_edema = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)

        # Fusion of direct attention and co-attention for each pathology
        self.fuse_scar = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse_edema = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # 3. Cooperative (non-zero-sum) pathology refinement
        self.joint_patho = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        # Independent cooperative gates (NOT 1 - gate!)
        self.gate_scar = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.gate_edema = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # 4. Final projection into decoder dimension
        self.proj = nn.Sequential(
            nn.Conv2d(3 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cine: torch.Tensor, psir: torch.Tensor, t2w: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = cine.shape

        # Step 1: Strip anatomy envelope from CINE (B, 1, H, W)
        strip_feat = cine.mean(dim=3, keepdim=True) + cine.mean(dim=2, keepdim=True)
        anatomy_gate = self.strip_anatomy(strip_feat)

        # Step 2: Multi-head cross-attention tokens
        q_cine = cine.flatten(2).transpose(1, 2)
        k_scar = psir.flatten(2).transpose(1, 2)
        k_edema = t2w.flatten(2).transpose(1, 2)

        # (a) CINE-guided attention
        attn_c_scar, _ = self.mha_cine_scar(q_cine, k_scar, k_scar, need_weights=False)
        attn_c_edema, _ = self.mha_cine_edema(q_cine, k_edema, k_edema, need_weights=False)

        # (b) Mutual pathology co-attention (Edema <-> Scar)
        attn_e_to_s, _ = self.mha_edema_to_scar(k_edema, k_scar, k_scar, need_weights=False)
        attn_s_to_e, _ = self.mha_scar_to_edema(k_scar, k_edema, k_edema, need_weights=False)

        # Reshape to spatial
        attn_c_scar = attn_c_scar.transpose(1, 2).reshape(batch, channels, height, width)
        attn_c_edema = attn_c_edema.transpose(1, 2).reshape(batch, channels, height, width)
        attn_e_to_s = attn_e_to_s.transpose(1, 2).reshape(batch, channels, height, width)
        attn_s_to_e = attn_s_to_e.transpose(1, 2).reshape(batch, channels, height, width)

        # Fuse direct + mutual co-attention
        scar_feat = self.fuse_scar(torch.cat([attn_c_scar, attn_s_to_e], dim=1))
        edema_feat = self.fuse_edema(torch.cat([attn_c_edema, attn_e_to_s], dim=1))

        # Step 3: Cooperative joint pathology refinement
        joint = self.joint_patho(torch.cat([scar_feat, edema_feat], dim=1))
        g_scar = self.gate_scar(joint)
        g_edema = self.gate_edema(joint)

        refined_scar = scar_feat + joint * g_scar
        refined_edema = edema_feat + joint * g_edema

        # Step 4: Anatomical envelope confinement
        # Residual gating ensures gradient flow while dampening extracardiac noise
        gated_scar = refined_scar * (0.5 + 0.5 * anatomy_gate)
        gated_edema = refined_edema * (0.5 + 0.5 * anatomy_gate)

        # Step 5: Final projection to decoder channel dimension
        return self.proj(torch.cat([cine, gated_scar, gated_edema], dim=1))
