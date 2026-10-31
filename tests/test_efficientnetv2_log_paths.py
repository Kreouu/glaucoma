import tempfile
import unittest
from pathlib import Path
from scripts.run_efficientnetv2_suite import experiment_log_path


class ExperimentLogPaths(unittest.TestCase):
    def test_distinct_decimal_learning_rates_do_not_collide(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ('lr0.0001_seed42','lr0.0003_seed42'):
                path=experiment_log_path(Path(folder)/name)
                with path.open('x') as stream:stream.write(name)
                self.assertEqual(path.name,name+'.log')

    def test_plain_smoke_name(self):
        self.assertEqual(experiment_log_path(Path('outputs/smoke')),Path('outputs/smoke.log'))


if __name__=='__main__':unittest.main()
