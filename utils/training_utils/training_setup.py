import torch
import torch.optim as optim

import general_config
import constants
from data_loading import create_datasets, create_dataloaders
from models import _2D_Unet, deeplabv3_plus, ROI_detector
from training import lr_handler


def model_setup(params, config):
    """
    creates model and moves it on to cpu/gpu
    """
    dset_name = config.dataset
    if config.model_id in constants.segmentor_ids:
        if dset_name == constants.imatfib_root_dir:
            n_classes = 2
        elif dset_name == constants.acdc_root_dir or dset_name == constants.acdc_test_dir:
            n_classes = 4
        else:
            n_classes = 8
        if config.model_id == constants.unet:
            model = _2D_Unet.UNet(n_channels=1, n_classes=n_classes,
                                  shrinking_factor=params.shrinking_factor)
        elif config.model_id == constants.deeplab or config.model_id == constants.resnext_deeplab:
            model = deeplabv3_plus.DeepLabV3_plus(
                n_channels=1, n_classes=n_classes, params=params, config=config)

        print("Number of classes: ", n_classes)

    elif config.model_id in constants.detectors:
        model = ROI_detector.ROI_Detector(params, config)
        print("Getting ready to get that ROI")

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


def prepare_dataloaders(params, config):
    dset_name = config.dataset
    training_dataset, validation_dataset = create_datasets.train_val(dataset_name=dset_name,
                                                                     params=params, config=config)
    training_dataloader, validation_dataloader = create_dataloaders.get_dataloaders(
        training_dataset, validation_dataset, params)
    return training_dataloader, validation_dataloader


def prepare_val_loader(params, config):
    dset_name = config.dataset
    val_dataset = create_datasets.create_val_set(dset_name, params, config)
    val_dataloader = create_dataloaders.get_test_loader(val_dataset, params)
    return val_dataloader


def prepare_test_loader(params, config):
    dset_name = config.dataset
    test_dataset = create_datasets.create_test_set(dset_name, params, config)
    test_dataloader = create_dataloaders.get_test_loader(test_dataset, params)
    return test_dataloader


def load_model(model, optimizer, params, dset_name, experiment_name):
    load_path = constants.model_path.format(dset_name, experiment_name)
    print("Loading model from: ", load_path)
    checkpoint = torch.load(load_path)
    start_epoch = 0
    if params.load_type == constants.load_simple:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # start training from the next epoch, do not repeat the same epoch it was saved on
        start_epoch = checkpoint['epoch'] + 1
    elif params.load_type == constants.load_transfer:
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        keys_to_delete = []
        for k_pre, v_pre in pretrained_dict.items():
            if v_pre.shape != model_dict[k_pre].shape:
                print(k_pre)
                print("pretrained: ", v_pre.shape, "current: ", model_dict[k_pre].shape)
                print("Found a layer with different shape in the pretrained model... This should only happen at most once (for the last layer)")
                # delete the mismatched key
                keys_to_delete.append(k_pre)
        for k in keys_to_delete:
            del pretrained_dict[k]
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    else:
        raise Warning("There should be some kind of loading?")
    print('Model loaded successfully')

    return start_epoch


def load_model_weights(model, dset_name, experiment_name):
    load_path = constants.model_path.format(dset_name, experiment_name)
    print("Loading model from: ", load_path)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Weights loaded successfully!")


def save_model(epoch, model, optimizer, stats, dset_name, experiment_name, last_model=False):
    if last_model:
        save_path = constants.model_last_path.format(dset_name, experiment_name)
    else:
        save_path = constants.model_path.format(dset_name, experiment_name)
        stats.save(constants.stats_path.format(dset_name, experiment_name))

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
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
