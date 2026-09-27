import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.run_layout import RunLayout


class RunLayoutTests(unittest.TestCase):
    def test_train_resume_resolves_run_root(self):
        from training.train import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = RunLayout.create(root / 'run', 'Testing-Small', 1234)
            checkpoint = layout.checkpoints / 'last.pth'
            checkpoint.touch()
            with patch('training.train.Trainer') as trainer:
                main(['--config', 'testing', '--data-root', str(root),
                      '--list-dir', str(root), '--resume', str(checkpoint)])
            self.assertEqual(trainer.call_args.args[2], layout.root)
            self.assertEqual(trainer.call_args.args[1].model_name, 'Testing-Small')

    def test_run_all_passes_nested_checkpoint_to_evaluation(self):
        import run_all
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data'
            for modality in ('bSSFP', 'LGE', 'T2w'):
                (data / modality / 'train_npz').mkdir(parents=True)
            commands = []
            def run(command, description):
                commands.append(command)
                if command[1] == 'training/train.py':
                    layout = RunLayout.create(root / 'runs' / 'demo', 'CMSPA-Net', 1234)
                    (layout.checkpoints / 'best.pth').touch()
            with patch('run_all.run_command', side_effect=run):
                run_all.main(['--data-root', str(data), '--run-root', str(root / 'runs'),
                              '--run-id', 'demo', '--skip-cache'])
            evaluation = commands[-1]
            self.assertEqual(Path(evaluation[evaluation.index('--checkpoint') + 1]),
                             root / 'runs' / 'demo' / 'checkpoints' / 'best.pth')

    def test_new_run_and_checkpoint_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'custom_run'
            layout = RunLayout.create(root, 'CMSPA-Net', 1234)
            self.assertEqual({p.name for p in root.iterdir()},
                             {'tensorboard', 'checkpoints', 'logs_CMSPA-Net_seed1234'})
            checkpoint = layout.checkpoints / 'last.pth'
            checkpoint.touch()
            self.assertEqual(RunLayout.from_checkpoint(checkpoint), layout)
            self.assertEqual(layout.evaluation('test_vol'), layout.logs / 'test_results')
            self.assertEqual(layout.evaluation('val_vol'), layout.logs / 'validation_results')

    def test_legacy_flat_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'splits').mkdir()
            layout = RunLayout.from_checkpoint(root / 'last.pth')
            self.assertEqual(layout.root, root.resolve())
            self.assertEqual(layout.logs, root.resolve())
            self.assertEqual(layout.checkpoints, root.resolve())

    def test_legacy_checkpoint_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'splits').mkdir()
            (root / 'checkpoints').mkdir()
            layout = RunLayout.from_checkpoint(root / 'checkpoints' / 'best.pth')
            self.assertEqual(layout.root, root.resolve())
            self.assertEqual(layout.logs, root.resolve())
            self.assertEqual(layout.checkpoints, root.resolve() / 'checkpoints')
