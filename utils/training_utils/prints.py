import torch
import numpy as np
from random import randint


def show_training_info(params):
    """
    prints trainig settings
    """
    print("Hyperparameters and training settings\n")
    params_ = params.dict
    for k, v in params_.items():
        print(str(k) + " : " + str(v))

    print("-------------------------------------------------------\n")


def print_trained_parameters_count(model, optimizer=None):
    print('Total number of parameters of model: ', sum(p.numel() for p in model.parameters()))
    print('Total number of trainable parameters of model: ',
          sum(p.numel() for p in model.parameters() if p.requires_grad), "\n")

    if optimizer:
        print_parameters_given_to_opt(optimizer)


def print_parameters_given_to_opt(optimizer):
    print('Total number of parameters given to optimizer: ',
          sum(p.numel() for pg in optimizer.param_groups for p in pg['params']))
    print('Total number of trainable parameters given to optimizer: ',
          sum(p.numel() for pg in optimizer.param_groups for p in pg['params'] if p.requires_grad))

    print("-------------------------------------------------------\n")


def print_dataset_stats(train_loader, valid_loader):
    print('Train size: ', len(train_loader), len(train_loader.sampler.sampler))
    print('Val size: ', len(valid_loader), len(valid_loader.sampler))

    print("------------------------------------------------------\n")


def gradient_weight_check(model):
    '''
    will print mean abs value of gradients and weights during training to check for stability
    '''
    avg_grads, max_grads = [], []
    avg_weigths, max_weigths = [], []

    for n, p in model.named_parameters():
        if (p.requires_grad) and not isinstance(p.grad, type(None)):
            avg_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            avg_weigths.append(p.abs().mean())
            max_weigths.append(p.abs().max())

    avg_grads, max_grads = torch.FloatTensor(avg_grads), torch.FloatTensor(max_grads)
    avg_weigths, max_weigths = torch.FloatTensor(avg_weigths), torch.FloatTensor(max_weigths)

    print("Mean and max gradients: ", torch.mean(avg_grads), torch.mean(max_grads), "\n")
    print("Mean and max weights: ", torch.mean(avg_weigths), torch.mean(max_weigths), "\n\n")


def update_tensorboard_graphs_segmentation(writer, train_dice, train_loss, val_dice, val_loss, epoch):
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Dice/train', np.mean(train_dice), epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Dice/val', np.mean(val_dice), epoch)


def update_tensorboard_graphs_detection(writer, train_metrics, train_loss, val_metrics, val_loss, epoch):
    train_iou, train_iou_harsh, train_f1 = train_metrics
    val_iou, val_iou_harsh, val_f1 = val_metrics

    train_loc_loss, train_score_loss, train_box_conf_loss = train_loss
    val_loc_loss, val_score_loss, val_vox_conf_loss = val_loss

    total_train_loss = sum(train_loss)
    total_val_loss = sum(val_loss)

    writer.add_scalar('Loss/train', total_train_loss, epoch)
    writer.add_scalar('Loc Loss/train', train_loc_loss, epoch)
    writer.add_scalar('Score Loss/train', train_score_loss, epoch)
    writer.add_scalar('Box conf Loss/train', train_box_conf_loss, epoch)
    writer.add_scalar('IOU/train', train_iou, epoch)
    writer.add_scalar('IOU harsh/train', train_iou_harsh, epoch)
    writer.add_scalar('F1/train', train_f1, epoch)

    writer.add_scalar('Loss/val', total_val_loss, epoch)
    writer.add_scalar('Loc Loss/val', val_loc_loss, epoch)
    writer.add_scalar('Score Loss/val', val_score_loss, epoch)
    writer.add_scalar('Box conf Loss/val', val_vox_conf_loss, epoch)
    writer.add_scalar('IOU/val', val_iou, epoch)
    writer.add_scalar('IOU harsh/val', val_iou_harsh, epoch)
    writer.add_scalar('F1/val', val_f1, epoch)


def create_tensorboard_name(args, params):
    rand_id = randint(0, 10000)
    dir = args.experiment_name + "/" + args.dataset_name + "/"
    params_ = params.dict
    suffix = ''
    wanted_keys = ['n_epochs', 'default_width', 'roi_width', 'data_augmentation', 'roi_crop', 'relative_roi_perturbation', 'use_min_size']
    for k, v in params_.items():
        if k in wanted_keys:
            suffix += str(v) + "_"
    suffix += str(rand_id)
    print("Experiment suffix: ", suffix)
    return dir + suffix
