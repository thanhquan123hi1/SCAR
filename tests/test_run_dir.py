import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from training.config.config_utils import generate_run_dir


class RunDirNamingTests(unittest.TestCase):
    def test_run_dir_format_with_seed_and_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = generate_run_dir(
                run_root=tmp_dir,
                model_name="M2-Plus",
                seed=1234,
                timestamp="17h05",
            )
            self.assertEqual(run_dir.name, "M2-Plus_seed1234_17h05")
            self.assertTrue(run_dir.is_dir())

    def test_run_dir_format_default_timestamp_pattern(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = generate_run_dir(
                run_root=tmp_dir,
                model_name="M2-Plus",
                seed=42,
            )
            self.assertTrue(run_dir.name.startswith("M2-Plus_seed42_"))
            self.assertRegex(run_dir.name, r"^M2-Plus_seed42_\d{4}-\d{2}-\d{2}_\d{2}h\d{2}$")

    def test_run_dir_format_seed_prefixed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = generate_run_dir(
                run_root=tmp_dir,
                model_name="M2-Plus",
                seed="seed999",
                timestamp="12h30",
            )
            self.assertEqual(run_dir.name, "M2-Plus_seed999_12h30")

    def test_run_dir_collision_handling(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dir1 = generate_run_dir(
                run_root=tmp_dir,
                model_name="M2-Plus",
                seed=1234,
                timestamp="17h05",
            )
            (dir1 / "best.pt").write_text("dummy")

            dir2 = generate_run_dir(
                run_root=tmp_dir,
                model_name="M2-Plus",
                seed=1234,
                timestamp="17h05",
            )
            self.assertEqual(dir2.name, "M2-Plus_seed1234_17h05_01")
            self.assertTrue(dir2.is_dir())

    def test_empty_run_directories_remain_reserved(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            directories = [generate_run_dir(tmp_dir, "M2-Plus", 1234, "17h05")
                           for _ in range(3)]
            self.assertEqual([path.name for path in directories], [
                "M2-Plus_seed1234_17h05",
                "M2-Plus_seed1234_17h05_01",
                "M2-Plus_seed1234_17h05_02",
            ])
            self.assertTrue(all(not any(path.iterdir()) for path in directories))

    def test_concurrent_runs_receive_distinct_directories(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            def reserve(_):
                return generate_run_dir(tmp_dir, "M2-Plus", 1234, "17h05")

            with ThreadPoolExecutor(max_workers=8) as pool:
                directories = list(pool.map(reserve, range(16)))
            self.assertEqual(len(set(directories)), 16)
            self.assertTrue(all(path.is_dir() for path in directories))


if __name__ == "__main__":
    unittest.main()
