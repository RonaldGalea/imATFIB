import torch
import numpy as np
import time

try:
    from apex import amp
    amp_available = True
except ImportError:
    print("Cannot import NVIDIA Apex...")

import general_config
import constants
from training import model_statistics, lr_handler
from utils import training_setup, training_processing


class Model_Trainer():
    """
    Class that handles training of a model
    """

    def __init__(self, model, training_dataloader, validation_dataloader, optimizer, params, stats,
                 dataset_name, start_epoch=0):
        self.model = model
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.params = params
        self.stats = stats
        self.train_statistics = model_statistics.Model_Statistics(len(training_dataloader),
                                                                  params, self.model.n_classes - 1,
                                                                  'train', self.model)
        self.val_statistics = model_statistics.Model_Statistics(len(validation_dataloader),
                                                                params, self.model.n_classes - 1,
                                                                'val')
        self.start_epoch = start_epoch
        self.dataset_name = dataset_name

        # weight = torch.tensor([0.05, 0.95]).to(general_config.device)
        self.loss_function = torch.nn.NLLLoss(weight=None, reduction='mean')
        self.optimizer = optimizer
        self.lr_handling = lr_handler.Learning_Rate_Handler(len(training_dataloader), params)
        self.threshold = 1

    def train(self):
        for epoch in range(self.params.n_epochs):
            print("Starting epoch: ", epoch, "\n")
            start = time.time()

            self.model.train()
            self.train_statistics.reset(epoch)
            self.lr_handling.reset_batch_count()
            for batch_nr, (image, mask, infos) in enumerate(self.training_dataloader):
                image, mask = image.to(general_config.device), mask.to(general_config.device)
                mask = mask.to(torch.int64)
                loss_value, dice = self.process_sample_train(image, mask, infos)
                self.train_statistics.update(loss_value, dice)
                self.lr_handling.step(epoch, self.optimizer)

            print("--------------------------------------------------------------")
            print("Base learning rate vs current learning rate: ",
                  self.params.learning_rate,
                  self.params.learning_rate * self.lr_handling.poly_decay, "\n")

            print("Final training results: ")
            self.train_statistics.print_batches_statistics()
            print("--------------------------------------------------------------")
            print("Epoch finished in: ", time.time() - start, "\n\n\n")

            if (epoch + 1) % general_config.evaluation_step == 0:
                self.evaluate(epoch)

    def evaluate(self, epoch):
        print("Starting evaluation at epoch: ", epoch, "\n")
        self.model.eval()
        self.val_statistics.reset(epoch)
        with torch.no_grad():
            for batch_nr, (image, mask, infos) in enumerate(self.validation_dataloader):
                image, mask = image.to(general_config.device), mask.to(general_config.device)
                mask = mask.to(torch.int64)
                loss_value, dice = self.process_sample_val(image, mask, infos)
                self.val_statistics.update(loss_value, dice)

        print("Final validation results: ")
        self.val_statistics.print_batches_statistics()
        print("\n\n\n")

        val_dice = self.val_statistics.get_dice()
        if np.mean(val_dice) > self.stats.val:
            self.update_stats()
            training_setup.save_model(epoch, self.model, self.optimizer, self.params, self.stats,
                                      self.dataset_name)

    def process_sample_train(self, image, mask, infos):
        """
        Process batch in train
        """
        prediction = self.feed_forward_batch(image, mask)

        self.optimizer.zero_grad()
        loss = self.loss_function(prediction, mask)
        if general_config.use_amp and amp_available:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()

        dice = training_processing.compute_dice(prediction, mask)
        return loss.item(), dice

    def process_sample_val(self, volume, mask, infos):
        """
        Processes volume slice by slice, upsamples output to original shape, computes metrics
        """
        processed_volume = training_processing.process_volume(self.model, volume, mask)
        loss = self.loss_function(processed_volume, mask)
        dice = training_processing.compute_dice(processed_volume, mask)
        return loss.item(), dice

    def feed_forward_batch(self, image, mask):
        batch, height, width = image.shape
        image = image.view(batch, 1, height, width)
        return self.model(image)

    def update_stats(self):
        val_dice = self.val_statistics.get_dice()
        train_dice = self.train_statistics.get_dice()
        if len(val_dice) > 1:
            for value, name in zip(val_dice, constants.heart):
                self.stats.dict[name + '_val'] = value
            for value, name in zip(train_dice, constants.heart):
                self.stats.dict[name + '_train'] = value
        self.stats.val = np.mean(val_dice)
        self.stats.train = np.mean(train_dice)
