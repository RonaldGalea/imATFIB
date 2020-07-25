import torch
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
try:
    from apex import amp
    amp_available = True
except ImportError:
    print("Cannot import NVIDIA Apex...")

import general_config
import constants
from training import model_statistics, freezer
from utils.training_utils import training_setup, training_processing, prints, box_utils
from utils import metrics


class Model_Trainer():

    def __init__(self, model, training_dataloader, validation_dataloader, optimizer, params, stats,
                 dataset_name, start_epoch=0, experiment_info="no_info", experiment_name="no_name"):
        self.model = model
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.params = params
        self.stats = stats
        self.start_epoch = start_epoch
        self.dataset_name = dataset_name
        self.experiment_info = experiment_info
        self.experiment_name = experiment_name
        self.writer = None
        self.optimizer = optimizer
        self.lr_handling = training_setup.lr_decay_setup(len(training_dataloader), params)
        self.model_freezer = freezer.Model_Freezer(model, optimizer, params)

    def setup(self):
        if hasattr(self.params, "freeze_type"):
            # otherwise training is resumed, so model should be loaded frozen
            if self.start_epoch == 0:
                self.model_freezer.freeze()

    def train(self):
        self.setup()
        for epoch in range(self.start_epoch, self.params.n_epochs):
            print("Starting epoch: ", epoch, "\n")
            start = time.time()
            self.model.train()
            self.train_statistics.reset(epoch)
            for batch_nr, (image, target) in enumerate(self.training_dataloader):
                image, target = image.to(general_config.device), target.to(general_config.device)
                loss, metrics = self.process_sample_train(image, target)
                self.train_statistics.update(loss, metrics)
                self.lr_handling.step(epoch, self.optimizer)
            self.model_freezer.step(epoch)

            self.print_epoch_stats(start)
            self.check_evaluation_step(epoch)

    def check_evaluation_step(self, epoch):
        if (epoch + 1) % general_config.evaluation_step == 0 or epoch == 0:
            # init writer just as model is actually save to stop spamming empty stuff
            if self.writer is None:
                self.writer = SummaryWriter(log_dir="runs/"+self.experiment_info,
                                            filename_suffix=self.params.model_id)
            self.evaluation(epoch)

    def evaluation(self, epoch):
        val_metrics, val_loss = self.evaluate(epoch)
        train_metrics = self.train_statistics.get_metrics()
        train_loss = self.train_statistics.get_loss()
        self.update_tensorboard(val_metrics, val_loss, train_metrics, train_loss, epoch)

    def post_validation_steps(self, epoch):
        self.print_validation_stats()
        self.check_saving(epoch)
        return self.val_statistics.get_metrics(), self.val_statistics.get_loss()

    def check_saving(self, epoch):
        train_metrics = self.train_statistics.get_metrics()
        val_perf = self.val_statistics.get_performance()

        if val_perf > self.stats.val:
            self.val_statistics.update_stats(self.stats, train_metrics)
            training_setup.save_model(epoch, self.model, self.optimizer, self.stats,
                                      self.dataset_name, self.experiment_name)

    def print_epoch_stats(self, start):
        print("--------------------------------------------------------------")
        print("Base learning rate vs current learning rate: ",
              self.params.learning_rate, self.lr_handling.get_current_lr(), "\n")

        print("Final training results: ")
        self.train_statistics.print_batches_statistics()
        print("--------------------------------------------------------------")
        print("Epoch finished in: ", time.time() - start, "\n\n\n")

    def print_validation_stats(self):
        print("Final validation results: ")
        self.val_statistics.print_batches_statistics()
        print("\n\n\n")

    def evaluate(self):
        raise NotImplementedError

    def process_sample_train(self):
        """
        Process batch in train
        """
        raise NotImplementedError

    def process_sample_val(self):
        """
        Processes volume slice by slice, upsamples output to original shape, computes metrics
        """
        raise NotImplementedError

    def update_tensorboard(self, val_metrics, val_loss, train_metrics, train_loss, epoch):
        raise NotImplementedError


class Segmentation_Trainer(Model_Trainer):
    """
    Class that handles training of a segmentation model
    """

    def __init__(self, model, training_dataloader, validation_dataloader, optimizer, params, stats,
                 dataset_name, start_epoch=0, experiment_info="no_info", experiment_name="no_name"):
        super(Segmentation_Trainer, self).__init__(model, training_dataloader,
                                                   validation_dataloader,
                                                   optimizer, params, stats, dataset_name,
                                                   start_epoch, experiment_info, experiment_name)

        self.train_statistics = model_statistics.Segmentor_Statistics(len(training_dataloader),
                                                                      params, self.model.n_classes - 1,
                                                                      'train', self.model)
        self.val_statistics = model_statistics.Segmentor_Statistics(len(validation_dataloader),
                                                                    params, self.model.n_classes - 1,
                                                                    'val')

        # weight = torch.tensor([0.05, 0.95]).to(general_config.device)
        self.loss_function = torch.nn.NLLLoss(weight=None, reduction='mean')

    def evaluate(self, epoch=0):
        print("Starting evaluation at epoch: ", epoch, "\n")
        self.model.eval()
        self.val_statistics.reset(epoch)
        with torch.no_grad():
            for batch_nr, (image, mask, r_info, _) in enumerate(self.validation_dataloader):
                image, mask = image.to(general_config.device), mask.to(general_config.device)
                loss_value, dice = self.process_sample_val(image, mask, r_info)
                self.val_statistics.update(loss_value, dice)
        return self.post_validation_steps(epoch)

    def process_sample_train(self, image, mask):
        prediction = self.model(image)

        self.optimizer.zero_grad()
        loss = self.loss_function(prediction, mask)
        if general_config.use_amp and amp_available:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()

        dice, _, _ = training_processing.compute_dice(prediction, mask)

        return loss.item(), dice

    def process_sample_val(self, volume, mask, r_info):
        processed_volume = training_processing.process_volume(self.model, volume, mask,
                                                              self.params, r_info)
        loss = self.loss_function(processed_volume, mask)
        dice, _, _ = training_processing.compute_dice(processed_volume, mask)
        return loss.item(), dice

    def update_tensorboard(self, val_metrics, val_loss, train_metrics, train_loss, epoch):
        prints.update_tensorboard_graphs_segmentation(self.writer, np.mean(train_metrics),
                                                      train_loss, np.mean(val_metrics),
                                                      val_loss, epoch + 1)


class Detector_Trainer(Model_Trainer):
    """
    Class that handles training of a detection model
    """

    def __init__(self, model, training_dataloader, validation_dataloader, optimizer, params, stats,
                 dataset_name, start_epoch=0, experiment_info="no_info", experiment_name="no_name"):
        super(Detector_Trainer, self).__init__(model, training_dataloader,
                                               validation_dataloader,
                                               optimizer, params, stats, dataset_name,
                                               start_epoch, experiment_info, experiment_name)

        self.train_statistics = model_statistics.Detection_Statistics(len(training_dataloader),
                                                                      params, 'train', self.model)
        self.val_statistics = model_statistics.Detection_Statistics(len(validation_dataloader),
                                                                    params, 'val')

        self.encompassing_penalty_factor = 3
        # much more positive than negative examples
        self.weight = torch.tensor([1]).to(general_config.device)

    def evaluate(self, epoch=0):
        print("Starting evaluation at epoch: ", epoch, "\n")
        self.model.eval()
        self.val_statistics.reset(epoch)
        with torch.no_grad():
            for batch_nr, (image, target) in enumerate(self.validation_dataloader):
                image, target = image.to(general_config.device), target.to(general_config.device)
                loss, metrics = self.process_sample_val(image, target)
                self.val_statistics.update(loss, metrics)
        return self.post_validation_steps(epoch)

    def process_sample_train(self, image, target):
        """
        target: batch x 5 tensor
        """
        ROI, heart_presence, ROI_pred, score_pred, [iou, f1] = self.process_sample(image, target)

        self.optimizer.zero_grad()
        loc_loss, score_loss = self.compute_losses(ROI, heart_presence, ROI_pred, score_pred)
        loss = loc_loss + score_loss
        loss.backward()
        self.optimizer.step()

        return [loc_loss.item(), score_loss.item()], [iou.item(), f1.item()]

    def process_sample_val(self, volume, target):
        """
        volume - (D, H, W) tensor
        target - list of coords and score
        """
        volume = volume.unsqueeze(1)
        ROI, heart_presence, ROI_pred, score_pred, [iou, f1] = self.process_sample(volume, target)
        loc_loss, conf_loss = self.compute_losses(ROI, heart_presence, ROI_pred, score_pred)
        return [loc_loss.item(), conf_loss.item()], [iou.item(), f1.item()]

    def process_sample(self, image, target):
        # normaliza coords in 0 - 1 range
        ROI = target[:, :4] / self.params.default_height

        heart_presence = target[:, 4]
        heart_presence = heart_presence.to(torch.int64)

        ROI_pred, score_pred = self.model(image)

        with torch.no_grad():
            iou = metrics.harsh_IOU(ROI_pred, ROI, heart_presence, self.model.anchor)
            f1 = metrics.f1_score(score_pred, heart_presence)

        return ROI, heart_presence, ROI_pred, score_pred, [iou, f1]

    def compute_losses(self, ROI, heart_presence, ROI_pred, score_pred):
        loc_loss = training_processing.compute_loc_loss(ROI_pred, ROI, heart_presence, self.model.anchor,
                                                        self.encompassing_penalty_factor)
        score_loss = training_processing.compute_confidence_loss(score_pred, heart_presence,
                                                                 self.weight)
        return loc_loss, score_loss

    def update_tensorboard(self, val_metrics, val_loss, train_metrics, train_loss, epoch):
        prints.update_tensorboard_graphs_detection(self.writer, train_metrics, train_loss,
                                                   val_metrics, val_loss, epoch + 1)
