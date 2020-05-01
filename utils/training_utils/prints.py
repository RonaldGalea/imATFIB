import torch


def show_training_info(params):
    """
    prints trainig settings
    """
    print("Hyperparameters and training settings\n")
    params_ = params.dict
    for k, v in params_.items():
        print(str(k) + " : " + str(v))

    print("-------------------------------------------------------\n")


def print_trained_parameters_count(model, optimizer):
    print('Total number of parameters of model: ', sum(p.numel() for p in model.parameters()))
    print('Total number of trainable parameters of model: ',
          sum(p.numel() for p in model.parameters() if p.requires_grad), "\n")

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
