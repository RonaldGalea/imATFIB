import torch

try:
    from apex import amp
    use_amp = True
except ImportError:
    print("Cannot import NVIDIA Apex...")

import general_config
from training import model_statistics
from utils import metrics


class Model_Trainer():
    """
    Class that handles training of a model
    """

    def __init__(self, model, training_dataloader, validation_dataloader, params,
                 optimizer=None):
        self.model = model
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.params = params
        self.train_statistics = model_statistics.Model_Statistics(params)
        self.val_statistics = model_statistics.Model_Statistics(params)

        self.loss_function = torch.nn.NLLLoss(weight=None, reduction='mean')
        self.optimizer = optimizer

    def train(self):
        for epoch in range(1, self.params.n_epochs + 1):
            print("Starting epoch: ", epoch, "\n")

            self.train_statistics.reset()
            for volume_batch_nr, (image, mask, infos) in enumerate(self.training_dataloader):
                print("In training, batch shape: ", image.shape)

                loss_value, dice = self.process_sample(image, mask, infos)
                self.train_statistics.update(loss_value, dice)

    def process_sample(self, image, mask, infos):
        """
        Process batch
        """
        image, mask = image.to(general_config.device), mask.to(general_config.device)
        prediction = self.model(image)
        loss = self.loss_function(image, prediction)
        if use_amp:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()

        dice = metrics.metrics(mask, prediction)

        return loss.item(), dice
