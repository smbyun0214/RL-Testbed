import tensorflow as tf
from collections import OrderedDict


class Logger:
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)
        self._metrics = OrderedDict({})

    def add_metric(self, name, type, category):
        if type == "sum":
            self._metrics[(name, category)] = tf.keras.metrics.Sum(name, dtype=tf.float32)
        elif type == "mean":
            self._metrics[(name, category)] = tf.keras.metrics.Mean(name, dtype=tf.float32)
        return self._metrics[(name, category)]
    
    def write_metrics(self, step):
        with self.writer.as_default():
            for (name, category), metric in self._metrics.items():
                tf.summary.scalar(category, metric.result(), step=step)
    
    def write_scalar(self, data, category, step):
        with self.writer.as_default():
            tf.summary.scalar(category, data=data, step=step)
    
    def reset_metrics(self):
        for (name, category), metric in self._metrics.items():
            metric.reset_states()
