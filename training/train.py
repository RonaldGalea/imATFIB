import general_config
from training import model_statistics
from utils import metrics


class Model_Trainer():
    """
    Class that handles training of a model
    """

    def __init__(self, model, training_dataloader, validation_dataloader, params):
        self.model = model
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.params = params
        self.train_statistics = model_statistics.Model_Statistics(params)
        self.val_statistics = model_statistics.Model_Statistics(params)

    def train(self):
        for epoch in range(1, self.params.n_epochs + 1):
            print("Starting epoch: ", epoch, "\n")

            self.train_statistics.reset()
            for volume_batch_nr, (image, mask, infos) in enumerate(self.training_dataloader):
                print("In training, batch shape: ", image.shape)
                image, mask = image.to(general_config.device), mask.to(general_config.device)
                # loss_value, metrics = self.process_training_sample(image, mask, infos)
                # self.train_statistics.update(loss_value, metrics)

    def process_sample(self, image, mask, infos):
        """
        Process batch
        """
