"""Unit tests for M2-Max architecture and skip fusion."""
import pytest
import torch
from training.models.cmspa_net import CMSPANet, get_config
from training.models.modules.m2_max import M2Max_Fusion, SkipGateFusion
from training.loss.losses import SegmentationLoss


def test_m2_max_fusion_shape():
    m = M2Max_Fusion(in_channels=1024, out_channels=512, num_heads=8)
    cine = torch.randn(2, 1024, 8, 8)
    psir = torch.randn(2, 1024, 8, 8)
    t2w = torch.randn(2, 1024, 8, 8)
    out = m(cine, psir, t2w)
    assert out.shape == (2, 512, 8, 8)


def test_skip_gate_fusion_shape():
    s = SkipGateFusion(channels=256)
    cine = torch.randn(2, 256, 32, 32)
    psir = torch.randn(2, 256, 32, 32)
    t2w = torch.randn(2, 256, 32, 32)
    out = s(cine, psir, t2w)
    assert out.shape == (2, 256, 32, 32)


def test_cmspa_net_m2_max():
    config = get_config("M2-MAX")
    config.resnet.num_layers = (1, 1, 1)
    config.resnet.width_factor = 0.5
    config.decoder_channels = (64, 32, 16, 8)
    config.skip_channels = [256, 128, 32, 0]
    config.fused_channels = 64
    config.cross_attention_heads = 4

    model = CMSPANet(config, img_size=64, num_classes=4)
    c = torch.randn(1, 1, 64, 64)
    p = torch.randn(1, 1, 64, 64)
    t = torch.randn(1, 1, 64, 64)
    logits = model(c, p, t)
    assert logits.shape == (1, 4, 64, 64)


def test_segmentation_loss_conditional_aar():
    criterion = SegmentationLoss(n_classes=4, aar_weight=0.25)
    logits = torch.randn(2, 4, 32, 32)
    # Target without pathology
    target_clean = torch.randint(0, 2, (2, 32, 32))
    loss_clean = criterion(logits, target_clean)
    assert loss_clean["aar_loss"].item() == 0.0
    assert torch.isfinite(loss_clean["loss"])

    # Target with pathology
    target_patho = torch.randint(0, 4, (2, 32, 32))
    target_patho[0, 5, 5] = 2
    target_patho[1, 10, 10] = 3
    loss_patho = criterion(logits, target_patho)
    assert loss_patho["aar_loss"].item() > 0.0
    assert torch.isfinite(loss_patho["loss"])
