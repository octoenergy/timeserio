from pytest_mock import mocker  # noqa

import numpy as np

from timeserio.keras.callbacks import (
    TimeLogger, _get_log_metrics, _format_epoch_metric, _format_epoch_summary
)

history = {
    'batches': [
        {
            'batch': [0, 1],
            'size': [300, 200],
            'loss': [0.4, 0.3],
            'mean_absolute_error': [0.28948462, 0.2989089]
        },
        {
            'batch': [0, 1],
            'size': [300, 200],
            'loss': [0.2, 0.1],
            'mean_absolute_error': [0.30267844, 0.2740341]
        }
    ],
    'val_loss': [0.37, 0.17],
    'val_mean_absolute_error': [0.2915948450565338, 0.2897853195667267],
    'loss': [0.35, 0.15],
    'mean_absolute_error': [0.29325432777404786, 0.2912207067012787],
    'lr': [0.01, 0.01],
    'epoch': [0, 1]
}


class TestFormatting:
    def test_get_log_metrics(self):
        metrics = _get_log_metrics(history)
        assert 'loss' in metrics
        assert 'epochs' not in metrics
        assert 'batches' not in metrics

    def test_format_epoch_metric(self):
        message = _format_epoch_metric(history, metric='loss', errors=True)
        assert '+-' in message

    def test_format_epoch_summary(self):
        message = _format_epoch_summary(history)
        assert 'epoch:' in message


class TestTimeLogger():
    def test_epoch_stats(self, mocker):  # noqa
        time = mocker.patch("timeserio.keras.callbacks._now")
        times = [1, 2, 3, 6, 7, 11]

        time.side_effect = times
        logger = TimeLogger()

        for epoch, _ in enumerate(times[::2]):
            logger.on_epoch_begin(epoch)
            logger.on_epoch_end(epoch)

        # These are the actual faked elapsed times
        actual_times = [2 - 1, 6 - 3, 11 - 7]
        mean = np.mean(actual_times)
        std = np.std(actual_times)
        max_time = np.max(actual_times)
        min_time = np.min(actual_times)

        np.testing.assert_allclose(
            logger._get_stats(), [mean, std, max_time, min_time]
        )

    def test_train_(self, mocker):  # noqa
        time = mocker.patch("timeserio.keras.callbacks._now")
        times = [1, 2, 3, 11]

        time.side_effect = times
        logger = TimeLogger()
        logger.on_train_begin()
        logger.on_epoch_begin(0)
        logger.on_epoch_end(0)
        logger.on_train_end()

        assert logger.training_time == 11 - 1
