from pathlib import Path

import pandas as pd

from src.experiment import REPETITIONS, build_configurations, load_data, run_experiment


def test_wdbc_partition_is_reproducible():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape == (455, 30)
    assert X_test.shape == (114, 30)
    assert len(y_train) + len(y_test) == 569


def test_has_six_homogeneous_configurations():
    assert set(build_configurations(42)) == {
        "svm", "svm_pca", "random_forest", "random_forest_pca", "boosting", "boosting_pca"
    }


def test_experiment_writes_repetitions_and_medians(tmp_path):
    results, median = run_experiment(tmp_path)
    assert len(results) == len(REPETITIONS) * 6
    assert len(median) == 6
    assert (tmp_path / "results_repetitions.csv").exists()
    assert (tmp_path / "results_median.csv").exists()
    assert len(list(Path(tmp_path).glob("model_*.joblib"))) == 6
    assert pd.read_csv(tmp_path / "results_median.csv").shape[0] == 6
