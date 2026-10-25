import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch
import numpy as np
import torch

from training.evaluation_logging import evaluation_logger


class EvaluationLoggingTests(unittest.TestCase):
    def test_evaluation_without_tensorboard_directory(self):
        self._assert_primary_evaluation()

    def test_legacy_checkpoint_evaluation_exports_only_primary_metrics(self):
        self._assert_primary_evaluation(legacy=True)

    def _assert_primary_evaluation(self, legacy=False):
        from training.evaluate import main
        from training.run_layout import RunLayout
        from training.metrics.surface_distance import BENCHMARK_PROTOCOL
        from training.dataset.data_contract import resolve_label_order
        events = []
        mask = np.zeros((32, 32, 2), dtype=np.uint8)
        mask[8:24, 8:24] = 1
        mask[2, 2] = 3
        mask[29, 29] = 2
        samples = [dict(case_name=name, image=mask, image1=mask, image2=mask, label=mask,
                        spacing=np.full(3, np.nan), affine=np.full((4, 4), np.nan),
                        spacing_unit='unknown')
                   for name in ('case_a', 'case_b')]
        checkpoint = dict(benchmark_protocol=BENCHMARK_PROTOCOL,
                          benchmark_data={'source_label_order': resolve_label_order('canonical')},
                          args={'label_order': 'canonical', 'img_size': 32},
                          model_config={'ablation': 'M2-PLUS'}, model={}, split_hashes={}, epoch=0)
        if legacy:
            checkpoint['benchmark_protocol'] = dict(
                BENCHMARK_PROTOCOL, id='myops380_voxel_v1',
                selection_metric='avg_pathology_dice',
                official_reference='I_MMSeg@90f46c4eb72924509895fcda6bc6a3b8c3316e66/utils.py:171-182',
                official_empty_policy='if either empty: Dice=1 when pred<200 and target<=200, else 0; HD95=0')
        def predict(*args, **kwargs):
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
                main(['--checkpoint', str(layout.checkpoints / 'best.pt'),
                      '--data-root', tmp])
            self.assertEqual(events, ['predict', 'predict'])
            self.assertFalse(layout.tensorboard.exists())
            output = layout.evaluation('test_vol')
            self.assertTrue((output / 'metrics.json').is_file())
            self.assertTrue((output / 'per_case.csv').is_file())
            self.assertIn('Dice=', (output / 'test.log').read_text(encoding='utf-8'))
            for name in ('case_a', 'case_b'):
                with np.load(output / f'{name}_pred.npz') as saved:
                    np.testing.assert_array_equal(saved['prediction'], mask)
            report = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(report['inference_protocol']['id'], 'single_pass_argmax_v1')
            self.assertEqual(report['inference_protocol']['mask_processing'], 'none')
            self.assertEqual(report['benchmark_protocol']['id'], 'myops380_voxel_v3')
            self.assertEqual(report['checkpoint_selection_metric'], checkpoint['benchmark_protocol']['selection_metric'])
            self.assertEqual(report['checkpoint_metric_protocol_id'], checkpoint['benchmark_protocol']['id'])
            self.assertEqual(report['scar']['mean_dice'], 1.0)
            self.assertFalse(any('official_' in key for key in report))
            self.assertNotIn('official_', json.dumps(report))
            self.assertNotIn('official_', (output / 'per_case.csv').read_text(encoding='utf-8'))

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
