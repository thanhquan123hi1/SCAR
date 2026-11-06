"""Small, portable provenance record for reproducible training reports."""
import hashlib
from pathlib import Path
import subprocess


def code_provenance():
    root = Path(__file__).resolve().parents[1]
    files = [p for folder in ("training", "preprocessing") for p in (root / folder).rglob("*")
             if p.suffix in {".py", ".yaml", ".txt", ".json"}]
    files += [root / name for name in ("run_all.py", "train.py", "test.py", "requirements.txt")]
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(files) if p.is_file()}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"git_revision": revision, "dirty": dirty, "files_sha256": hashes}


def initialization_provenance(args):
    for kind in ("pretrained", "init_weights"):
        value = getattr(args, kind, None)
        if value:
            path = Path(value).resolve()
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            return {"kind": kind, "path": str(path), "sha256": digest}
    return {"kind": "random", "seed": args.seed}
