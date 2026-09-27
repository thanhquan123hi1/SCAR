"""Shared output paths for new runs and existing checkpoint layouts."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    root: Path
    logs: Path
    checkpoints: Path

    @property
    def tensorboard(self):
        return self.root / 'tensorboard'

    @classmethod
    def create(cls, root, model_name, seed):
        root = Path(root).resolve()
        name = str(model_name).replace(' ', '-').replace('/', '-').replace('\\', '-')
        layout = cls(root, root / f'logs_{name}_seed{seed}', root / 'checkpoints')
        for path in (layout.logs, layout.checkpoints, layout.tensorboard):
            path.mkdir(parents=True, exist_ok=True)
        return layout

    @classmethod
    def from_checkpoint(cls, checkpoint):
        parent = Path(checkpoint).resolve().parent
        root = parent.parent if parent.name == 'checkpoints' else parent
        if root == parent and not (root / 'splits').exists() and (root.parent / 'splits').exists():
            root = root.parent
        if (root / 'splits').is_dir():
            return cls(root, root, parent)
        candidates = [p for p in root.glob('logs_*') if p.is_dir()]
        if len(candidates) != 1:
            raise ValueError(f'Expected one logs_<model>_seed<seed> directory in {root}')
        return cls(root, candidates[0], parent)

    def evaluation(self, split):
        return self.logs / {'test_vol': 'test_results', 'val_vol': 'validation_results'}[split]
