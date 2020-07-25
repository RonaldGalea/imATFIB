import numpy as np
import time

import general_config
import constants
from utils.training_utils import prints


class Model_Statistics():
    """
    Args:
    loader_size: int - size of the data loader

    Class to track and print model statistics

    Note: this prints statistics averaged per slice, not per volume, which can be much different
    """

    def __init__(self, loader_size, params, type, model=None):
        self.loader_size = loader_size
        self.params = params
        self.print_step = loader_size // general_config.statistics_print_step + 1
        self.type = type
        self.model = model

    def print_batches_statistics(self):
        current_time = time.time()
        print(self.type + " Epoch: ", self.epoch, "\n")
        print("Statistics until batch number: ", self.batch_count, " / ", self.loader_size, "\n")
        self.print_metrics_n_loss()
        if self.model is not None:
            prints.gradient_weight_check(self.model)
        n_samples = self.print_step * self.params.batch_size
        print("Time taken for past ", n_samples, " samples: ", current_time - self.start, "\n\n")
        self.start = current_time

    def print_metrics_n_loss(self):
        raise NotImplementedError

    def update(self, loss, dice):
        raise NotImplementedError

    def update_stats(self, stats):
        raise NotImplementedError

    def get_metrics(self):
        raise NotImplementedError

    def get_performance(self):
        raise NotImplementedError

    def get_loss(self):
        raise NotImplementedError

    def reset(self, epoch):
        raise NotImplementedError


class Segmentor_Statistics(Model_Statistics):
    def __init__(self, loader_size, params, n_classes, type, model=None):
        super(Segmentor_Statistics, self).__init__(loader_size, params, type, model=None)
        self.n_classes = n_classes
        if self.params.dataset == "imatfib-whs":
            self.hearts = ["whole"]
        elif self.params.dataset == "ACDC_training":
            self.hearts = constants.acdc_heart
        elif self.params.dataset == "mmwhs":
            self.hearts = constants.mmwhs_heart
        self.reset(0)

    def update(self, loss, dice):
        self.batch_count += 1
        self.dice = self.dice + dice
        self.loss += loss

        if (self.batch_count + 1) % self.print_step == 0:
            self.print_batches_statistics()

    def update_stats(self, stats, train_metrics):
        val_dice = self.dice / self.batch_count
        train_dice = train_metrics

        if len(val_dice) > 1:
            for value, name in zip(val_dice, self.hearts):
                stats.dict[name + '_val'] = value
            for value, name in zip(train_dice, self.hearts):
                stats.dict[name + '_train'] = value
        stats.val = np.mean(val_dice)
        stats.train = np.mean(train_dice)

    def print_metrics_n_loss(self):
        print(','.join(self.hearts) + " and mean dice: ", self.dice /
              self.batch_count, np.mean(self.dice) / self.batch_count, "\n")
        print("Loss: ", self.loss / self.batch_count, "\n")

    def get_metrics(self):
        return self.dice / self.batch_count

    def get_performance(self):
        return np.mean(self.dice / self.batch_count)

    def get_loss(self):
        return self.loss / self.batch_count

    def reset(self, epoch):
        self.start = time.time()
        self.loss = 0
        self.dice = np.array([0] * self.n_classes)
        self.batch_count = 0
        self.epoch = epoch


class Detection_Statistics(Model_Statistics):
    def __init__(self, loader_size, params, type, model=None):
        super(Detection_Statistics, self).__init__(loader_size, params, type, model=None)

    def update(self, loss, metrics):
        iou, f1 = metrics
        loc_loss, score_loss = loss
        self.batch_count += 1
        self.iou += iou
        self.f1 += f1
        self.loc_loss += loc_loss
        self.score_loss += score_loss

        if (self.batch_count + 1) % self.print_step == 0:
            self.print_batches_statistics()

    def update_stats(self, stats, train_metrics):
        val_iou = self.iou / self.batch_count
        val_f1 = self.f1 / self.batch_count
        train_iou, train_f1 = train_metrics

        stats.dict['iou_val'] = val_iou
        stats.dict['f1_val'] = val_f1
        stats.dict['iou_train'] = train_iou
        stats.dict['f1_train'] = train_f1

        stats.val = (val_iou + val_f1) / 2
        stats.train = (train_iou + train_f1) / 2

    def print_metrics_n_loss(self):
        print("Mean IOU: ", self.iou / self.batch_count)
        print("Mean F1: ", self.f1 / self.batch_count)
        print("Localization Loss: ", self.loc_loss / self.batch_count)
        print("Score Loss: ", self.score_loss / self.batch_count)
        print("Mean Loss: ", (self.loc_loss + self.score_loss) / self.batch_count, "\n")

    def get_metrics(self):
        return self.iou / self.batch_count, self.f1 / self.batch_count

    def get_performance(self):
        return (self.iou + self.f1) / (2 * self.batch_count)

    def get_loss(self):
        return [self.loc_loss / self.batch_count, self.score_loss / self.batch_count]

    def reset(self, epoch):
        self.start = time.time()
        self.loc_loss = 0
        self.score_loss = 0
        self.iou = 0
        self.f1 = 0
        self.batch_count = 0
        self.epoch = epoch
