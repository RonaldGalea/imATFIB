import torch
import torch.optim as optim

import general_config
import constants
from data_loading import create_datasets, create_dataloaders
from models import _2D_Unet, deeplabv3_plus
from training import lr_handler


def model_setup(dset_name, params):
    """
    creates model and moves it on to cpu/gpu
    """
    if general_config.seg_type == constants.whole_heart_seg:
        # heart plus background
        n_classes = 2
    else:
        if dset_name == constants.acdc_root_dir:
            n_classes = 4
    if params.model_id == constants.unet:
        model = _2D_Unet.UNet(n_channels=1, n_classes=n_classes)
    elif params.model_id == constants.deeplab:
        model = deeplabv3_plus.DeepLabV3_plus(n_channels=1,
                                              n_classes=n_classes, use_aspp=params.use_aspp)
    model.to(general_config.device)
    return model


def optimizer_setup(model, params):
    """
    creates optimizer
    """
    if params.optimizer == 'adam':
        optimizer = plain_adam(model, params)
    elif params.optimizer == 'sgd':
        optimizer = plain_sgd(model, params)

    return optimizer


def prepare_dataloaders(dset_name, params):
    training_dataset, validation_dataset = create_datasets.train_val(dataset_name=dset_name,
                                                                     params=params)
    training_dataloader, validation_dataloader = create_dataloaders.get_dataloaders(
        training_dataset, validation_dataset, params)
    return training_dataloader, validation_dataloader


def load_model(model, optimizer, params, dset_name):
    checkpoint = torch.load(constants.model_path.format(dset_name, params.model_id))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # start training from the next epoch, do not repeat the same epoch it was saved on
    start_epoch = checkpoint['epoch'] + 1
    print('Model loaded successfully')

    return start_epoch


def save_model(epoch, model, optimizer, params, stats, dset_name):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, constants.model_path.format(dset_name, params.model_id))
    stats.save(constants.stats_path.format(dset_name, params.model_id))
    print("Model saved successfully!")


def lr_decay_setup(loader_size, params):
    if params.lr_decay == constants.divide_decay:
        lr_handling = lr_handler.Learning_Rate_Handler_Divide(loader_size, params)
    elif params.lr_decay == constants.poly_decay:
        lr_handling = lr_handler.Learning_Rate_Handler_Poly(loader_size, params)
    return lr_handling


def plain_adam(model, params):
    return optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)


def plain_sgd(model, params):
    return optim.SGD(model.parameters(), lr=params.learning_rate,
                     weight_decay=params.weight_decay, momentum=0.9)
