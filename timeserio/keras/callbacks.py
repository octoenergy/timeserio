import logging
import time

import numpy as np
from keras.callbacks import Callback

logger = logging.getLogger(__name__)


def _now():
    """Return the current time, helper for mocking."""
    return time.time()


def _get_log_metrics(history, exclude=('epoch', 'batches')):
    metrics = [k for k in history.keys() if k not in exclude]
    return metrics


def _format_epoch_metric(
    history, metric='loss', fmt='.2e', errors=False, idx=-1, with_name=True
):
    if metric not in history:
        return KeyError(f"Metric '{metric}' not found in history.")
    mean_value = history[metric][idx]
    formatted_value = f"{mean_value:{fmt}}"
    if errors and 'batches' in history and metric in history['batches'][idx]:
        batch_values = history['batches'][idx][metric]
        batch_min, batch_max = np.min(batch_values), np.max(batch_values)
        batch_std = np.std(batch_values)
        formatted_errors = (
            f" +- {batch_std:{fmt}}"
            f" [{batch_min:{fmt}} .. {batch_max:{fmt}}]"
        )
    else:
        formatted_errors = ""
    if with_name:
        return f"{metric}: {formatted_value}{formatted_errors}"
    else:
        return f"{formatted_value}{formatted_errors}"


def _format_epoch_summary(history, fmt='.2e', idx=-1):
    metrics = _get_log_metrics(history)
    formatted_values = [
        _format_epoch_metric(history, metric=metric, fmt=fmt, idx=idx)
        for metric in metrics
    ]
    formatted_epoch = _format_epoch_metric(
        history, metric='epoch', fmt='d', idx=idx
    )
    formatted_values = [formatted_epoch] + formatted_values
    message = ' - '.join(formatted_values)
    return message


class HistoryLogger(Callback):
    """Log all history, including per-batch losses and metrics.

    Based on keras.callbacks.History
    """

    def __init__(self, batches=True):
        self.batches = batches
        super().__init__()

    def on_train_begin(self, logs=None):
        logger.info('Training started.')
        self.history = {}

    def on_epoch_begin(self, epoch, logs=None):
        batch_history = self.history.setdefault('batches', [])
        batch_history.append({})
        self.epoch_time_start = _now()

    def on_batch_end(self, batch, logs=None):
        if not self.batches:
            return
        batch_history = self.history['batches'][-1]
        for k, v in logs.items():
            batch_history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = _now() - self.epoch_time_start
        logs.update(epoch=epoch)
        logs.update(epoch_duration=elapsed)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        message = _format_epoch_summary(self.history)
        logger.info(message)

    def _log_training_metric_improvement(self, fmt='.2e'):
        metrics = _get_log_metrics(
            self.history, exclude=('batches', 'epoch', 'epoch_duration')
        )
        for metric in metrics:
            before = _format_epoch_metric(
                self.history, metric=metric, idx=0, errors=True
            )
            after = _format_epoch_metric(
                self.history,
                metric=metric,
                idx=-1,
                with_name=False,
                errors=True
            )
            logger.info(f"{metric}: {before} --> {after}")

    def _log_training_time(self, fmt='.2e'):
        times = self.history['epoch_duration']
        mean, std, total = np.mean(times), np.std(times), np.sum(times)
        min_duration, max_duration = np.min(times), np.max(times)
        logger.info(f"Total duration: {total:{fmt}} seconds")
        logger.info(
            f"Epoch duration: {mean:{fmt}} +- {std:{fmt}}"
            f" [{min_duration:{fmt}} .. {max_duration:{fmt}}] seconds"
        )

    def on_train_end(self, logs=None):
        num_epochs = len(self.history['epoch'])
        logger.info(f"Training finished in {num_epochs} epochs.")
        self._log_training_time()
        self._log_training_metric_improvement()


class TimeLogger(Callback):
    """
    Logs the times of training the network and per epoch.

    A summary of the total statistics is given when the network finishes
    the training process.
    """

    def __init__(self):
        super(Callback, self).__init__()
        self.start_training_time = 0
        self.epoch_times = []
        self.training_time = 0

    def on_train_begin(self, logs=None):
        self.start_training_time = _now()
        logger.info("TimeLogger on_train_begin")

    def on_train_end(self, logs=None):
        self.training_time = _now() - self.start_training_time
        logger.info("Timing: Training took %0.2f", self.training_time)
        mean, std, max_elapsed, min_elapsed = self._get_stats()
        logger.info(
            "Timing: Epochs summary mean/std/max/min: %0.5f/%0.5f/%0.5f/%0.5f",
            mean, std, max_elapsed, min_elapsed
        )

    def _get_stats(self):
        mean = np.mean(self.epoch_times)
        std = np.std(self.epoch_times)
        max_elapsed = np.max(self.epoch_times)
        min_elapsed = np.min(self.epoch_times)
        return mean, std, max_elapsed, min_elapsed

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = _now()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = _now() - self.epoch_time_start
        self.epoch_times.append(elapsed)
        logger.info("Timing: Epoch %d took %0.5f", epoch, float(elapsed))
