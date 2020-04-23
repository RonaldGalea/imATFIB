from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import general_config


def get_dataloaders(training_dataset, validation_dataset, params):
    """
    Creates and returns Dataloader objects from datasets
    These dataloaders support creation of customized batches
    """
    training_dataloader = DataLoader(training_dataset, batch_size=None,
                                     shuffle=False, num_workers=0,
                                     sampler=BatchSampler(SubsetRandomSampler([i for i in range(len(training_dataset))]),
                                                          batch_size=params.batch_size, drop_last=True))

    validation_dataloader = DataLoader(validation_dataset, batch_size=None,
                                       shuffle=False, num_workers=0,
                                       sampler=BatchSampler(SubsetRandomSampler([i for i in range(len(validation_dataset))]),
                                                            batch_size=1, drop_last=False))

    return training_dataloader, validation_dataloader
