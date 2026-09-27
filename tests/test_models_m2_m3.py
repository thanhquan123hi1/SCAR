import unittest
import torch

from training.models import build_model, get_testing, CMSPANet


class TestModelsM2M3(unittest.TestCase):
    def test_m3_cmspa_net(self):
        config = get_testing()
        config.ablation = "M3"
        model = CMSPANet(config, img_size=32)
        model.eval()

        cine = torch.randn(2, 1, 32, 32)
        psir = torch.randn(2, 1, 32, 32)
        t2w = torch.randn(2, 1, 32, 32)

        with torch.no_grad():
            out = model(cine, psir, t2w)
        self.assertEqual(out.shape, (2, 4, 32, 32))

    def test_m2_cross_attention(self):
        config = get_testing()
        config.ablation = "M2"
        model = CMSPANet(config, img_size=32)
        model.eval()

        cine = torch.randn(2, 1, 32, 32)
        psir = torch.randn(2, 1, 32, 32)
        t2w = torch.randn(2, 1, 32, 32)

        with torch.no_grad():
            out = model(cine, psir, t2w)
        self.assertEqual(out.shape, (2, 4, 32, 32))

    def test_registry_build(self):
        config = get_testing()
        m3 = build_model("cmspa_net", config=config, img_size=32)
        self.assertEqual(m3.ablation, "M3")

        m2 = build_model("cross_attn_baseline", config=config, img_size=32)
        self.assertEqual(m2.ablation, "M2")

        with self.assertRaises(ValueError):
            build_model("concat_baseline", config=config)

        with self.assertRaises(ValueError):
            build_model("sspanet_baseline", config=config)

    def test_invalid_ablation(self):
        config = get_testing()
        with self.assertRaises(ValueError):
            CMSPANet(config, ablation="M0")
        with self.assertRaises(ValueError):
            CMSPANet(config, ablation="M1")


if __name__ == "__main__":
    unittest.main()
