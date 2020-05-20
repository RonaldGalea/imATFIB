class Lr_Handler():
    def __init__(self, loader_size, params):
        self.params = params
        self.loader_size = loader_size
        self.decay = 1

    def step(self):
        raise NotImplementedError

    def get_current_lr(self):
        raise NotImplementedError


class Learning_Rate_Handler_Poly(Lr_Handler):
    """
    Poly learning rate annealing: https://arxiv.org/pdf/1506.04579.pdf
    """
    def __init__(self, loader_size, params):
        super(Learning_Rate_Handler_Poly, self).__init__(loader_size, params)
        self.batch_count = 0

    def step(self, epoch, optimizer):
        self.batch_count += 1
        max_iters = self.params.n_epochs * self.loader_size
        current_iters = (epoch * self.loader_size) + self.batch_count
        self.poly_decay = (1 - (current_iters / max_iters)) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.params.learning_rate * self.poly_decay

        if self.batch_count == self.loader_size:
            self.batch_count = 0

    def get_current_lr(self):
        return self.params.learning_rate * self.poly_decay


class Learning_Rate_Handler_Divide(Lr_Handler):
    """
    Divides learning rate by a factor at certain steps
    """
    def __init__(self, loader_size, params):
        super(Learning_Rate_Handler_Divide, self).__init__(loader_size, params)

    def step(self, epoch, optimizer):
        self.epoch = epoch
        if epoch == self.params.first_decay:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.params.learning_rate / 10

        if epoch == self.params.second_decay:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.params.learning_rate / 100

    def get_current_lr(self):
        if self.epoch < self.params.first_decay:
            return self.params.learning_rate
        elif self.epoch < self.params.second_decay:
            return self.params.learning_rate / 10
        return self.params.learning_rate / 100
