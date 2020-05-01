import numpy as np
import time
import torch

import general_config
from utils import prints


class Model_Statistics():
    """
    Args:
    loader_size: int - size of the data loader

    Class to track and print model statistics

    Note: this prints statistics averaged per slice, not per volume, which can be much different
    """

    def __init__(self, loader_size, params, n_classes, type, model=None):
        self.loader_size = loader_size
        self.params = params
        self.n_classes = n_classes
        self.print_step = loader_size // general_config.statistics_print_step + 1
        self.type = type
        self.model = model
        self.reset(0)

    def update(self, loss_value, dice):
        self.batch_count += 1
        self.dice = self.dice + dice
        self.loss_value += loss_value

        if (self.batch_count + 1) % self.print_step == 0:
            self.print_batches_statistics()

    def print_batches_statistics(self):
        current_time = time.time()
        print("Epoch: ", self.epoch, "\n")
        print("Statistics until batch number: ", self.batch_count, " / ", self.loader_size, "\n")
        print("LV, RV, Myo and mean dice: ", self.dice /
              self.batch_count, np.mean(self.dice) / self.batch_count, "\n")
        print("Loss: ", self.loss_value / self.batch_count, "\n")
        if self.model is not None:
            prints.gradient_weight_check(self.model)
        print("Time taken for past batches: ", current_time - self.start, "\n\n")
        self.start = current_time

    def get_dice(self):
        return self.dice / self.batch_count

    def get_loss(self):
        return self.loss_value / self.batch_count

    def reset(self, epoch):
        self.start = time.time()
        self.loss_value = 0
        self.dice = np.array([0] * self.n_classes)
        self.batch_count = 0
        self.epoch = epoch
