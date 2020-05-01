from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
import general_config


def get_dataloaders(training_dataset, validation_dataset, params):
    """
    Creates and returns Dataloader objects from datasets
    The training dataloader supports creation of customized batches
    The validation dataloader simply returns one volume at a time, which will be processed
    slice by slice
    """
    train_sampler = SubsetRandomSampler([i for i in range(len(training_dataset))])
    val_sampler = SequentialSampler([i for i in range(len(validation_dataset))])
    training_dataloader = DataLoader(training_dataset, batch_size=None,
                                     shuffle=False, num_workers=0,
                                     sampler=BatchSampler(train_sampler,
                                                          batch_size=params.batch_size,
                                                          drop_last=True))

    validation_dataloader = DataLoader(validation_dataset, batch_size=None,
                                       shuffle=False, num_workers=0,
                                       sampler=BatchSampler(val_sampler,
                                                            batch_size=1,
                                                            drop_last=False))

    return training_dataloader, validation_dataloader
