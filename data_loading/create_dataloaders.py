from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler


def get_dataloaders(training_dataset, validation_dataset, params):
    """
    Creates and returns Dataloader objects from datasets
    The training dataloader supports creation of customized batches
    The validation dataloader simply returns one volume at a time, which will be processed
    slice by slice
    """
    training_dataloader = get_training_loader(training_dataset, params)
    validation_dataloader = get_validation_loader(validation_dataset, params)

    return training_dataloader, validation_dataloader


def get_training_loader(training_dataset, params):
    train_sampler = SubsetRandomSampler([i for i in range(len(training_dataset))])

    training_dataloader = DataLoader(training_dataset, batch_size=None,
                                     shuffle=False, num_workers=0,
                                     sampler=BatchSampler(train_sampler,
                                                          batch_size=params.batch_size,
                                                          drop_last=True))

    return training_dataloader


def get_validation_loader(validation_dataset, params):
    val_sampler = SequentialSampler([i for i in range(len(validation_dataset))])

    validation_dataloader = DataLoader(validation_dataset, batch_size=None,
                                       shuffle=False, num_workers=0,
                                       sampler=BatchSampler(val_sampler,
                                                            batch_size=1,
                                                            drop_last=False))

    return validation_dataloader
