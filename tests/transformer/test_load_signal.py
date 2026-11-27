import numpy as np

from src.transformer.load_signal import create_data_loaders


def test_custom_batch_size_and_evaluation_order():
    x = np.arange(12).reshape(6, 2)
    labels = np.arange(6)
    train, evaluation = create_data_loaders(x, x, labels, labels, batch_size=4)
    assert train.batch_size == 4
    assert [ys.tolist() for _, ys in evaluation] == [[0, 1, 2, 3], [4, 5]]
