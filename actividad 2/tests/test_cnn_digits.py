import numpy as np

from src.cnn_digits import TinyCNN, load_data


def test_data_is_stratified_and_normalized():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[1:] == (8, 8)
    assert X_test.shape[1:] == (8, 8)
    assert np.max(X_train) <= 1
    assert set(np.unique(y_train)) == set(range(10))
    assert len(y_train) + len(y_test) == 1797


def test_cnn_predicts_valid_classes():
    X_train, X_test, y_train, _ = load_data()
    model = TinyCNN(filters=2, epochs=2, seed=42)
    model.fit(X_train[:160], y_train[:160])
    predictions = model.predict(X_test[:12])
    assert predictions.shape == (12,)
    assert set(predictions).issubset(set(range(10)))


def test_cnn_rejects_invalid_input():
    model = TinyCNN()
    try:
        model.predict(np.zeros((2, 4, 4)))
    except ValueError as error:
        assert "8x8" in str(error)
    else:
        raise AssertionError("Se esperaba ValueError para imágenes con forma inválida")
