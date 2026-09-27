import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch
import numpy as np
import torch

from training.evaluation_logging import evaluation_logger


class EvaluationLoggingTests(unittest.TestCase):
    def test_evaluation_without_tensorboard_directory(self):
        from training.evaluate import main
        from training.run_layout import RunLayout
        from training.metrics.surface_distance import BENCHMARK_PROTOCOL
        from training.dataset.data_contract import resolve_label_order
        events = []
        mask = np.arange(8).reshape(2, 2, 2) % 4
        samples = [dict(case_name=name, image=mask, image1=mask, image2=mask, label=mask)
                   for name in ('case_a', 'case_b')]
        checkpoint = dict(benchmark_protocol=BENCHMARK_PROTOCOL,
                          benchmark_data={'source_label_order': resolve_label_order('canonical')},
                          args={'label_order': 'canonical', 'img_size': 32},
                          model_config={'ablation': 'M3'}, model={}, split_hashes={}, epoch=0)
        def predict(*args):
            events.append('predict')
            return mask.copy()
        def split_names(directory, split):
            return {'train': ['train_1'], 'val': ['val_2'],
                    'test_vol': ['case_a', 'case_b']}[split]
        with tempfile.TemporaryDirectory() as tmp:
            layout = RunLayout.create(tmp, 'Model', 1)
            layout.tensorboard.rmdir()
            with patch('training.evaluate.load_checkpoint', return_value=checkpoint), \
                 patch('training.evaluate.model_from_config', return_value=Mock()), \
                 patch('training.evaluate.resolve_device', return_value=torch.device('cpu')), \
                 patch('training.evaluate.resolve_amp', return_value=None), \
                 patch('training.evaluate.read_split_names', side_effect=split_names), \
                 patch('training.evaluate.patient_id', side_effect=lambda name: name), \
                 patch('training.evaluate.lock_benchmark_data', return_value=checkpoint['benchmark_data']), \
                 patch('training.evaluate.MyopsDataset', return_value=samples), \
                 patch('training.evaluate.predict_volume', side_effect=predict):
                main(['--checkpoint', str(layout.checkpoints / 'best.pth'),
                      '--data-root', tmp, '--no-save-predictions'])
            self.assertEqual(events, ['predict', 'predict'])
            self.assertFalse(layout.tensorboard.exists())
            output = layout.evaluation('test_vol')
            self.assertTrue((output / 'metrics.json').is_file())
            self.assertTrue((output / 'per_case.csv').is_file())
            self.assertIn('Dice=', (output / 'test.log').read_text(encoding='utf-8'))

    def test_log_captures_progress_and_failure_and_closes_handlers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.log'
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                with evaluation_logger(path) as logger:
                    logger.info('case 1 completed')
                    raise RuntimeError('inference failed')
            content = path.read_text(encoding='utf-8')
            self.assertIn('case 1 completed', content)
            self.assertIn('Traceback', content)
            self.assertIn('inference failed', content)
            self.assertEqual(logger.handlers, [])

