import constants
from utils.training_utils import prints


class Model_Freezer():
    def __init__(self, model, optimizer, params):
        self.model = model
        self.optimizer = optimizer
        self.params = params
        self.total_layers = 11 if params.model_id == "2D_Unet" else 5
        self.n_unfrozen = 0

    def freeze(self):
        if not hasattr(self.params, "freeze_type"):
            return
        else:
            self.classifier_layer()
        self.print_stats()

    def step(self, current_epoch):
        if not hasattr(self.params, "freeze_type"):
            return
        elif self.params.freeze_type == constants.classifier_freeze:
            if current_epoch == (self.params.n_epochs // 2):
                self.classifier_layer(unfreeze=True)
                print("Some unfreezing has been done: ")
                self.print_stats()
        elif self.params.freeze_type == constants.progressive_freeze:
            # uniformly unfreeze all layers until half the total epochs
            half = self.params.n_epochs // 2
            if current_epoch <= half:
                n_unfrozen = int((current_epoch / half) * self.total_layers)
                self.whole_layers(n_unfrozen, unfreeze=True)
                if n_unfrozen != self.n_unfrozen:
                    print("Some unfreezing has been done: ")
                    self.n_unfrozen = n_unfrozen
                    self.print_stats()

    def classifier_layer(self, unfreeze=False):
        """
        Freezes all layers but exactly last layer

        if unfreeze == True
        Unfreezes all layers
        """
        if unfreeze:
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            for idx, child in enumerate(self.model.children()):
                # actually need the penultimate layer trainable too in the case of unet,
                # otherwise only 130 params are left to train, which is a waste
                if self.params.model_id == "2D_Unet" and idx == 9:
                    break
                for param in child.parameters():
                    param.requires_grad = False

            # classifier should be trainable
            for param in child.parameters():
                param.requires_grad = True

    def whole_layers(self, n_layers, unfreeze=False):
        """
        Freezes layers from first to last
        layers = elements of model.children()

        index of first unfrozen layer is total_layers - n_layers

        if unfreeze == True
        Unfreezes layers from last to first
        """
        first_unfrozen = self.total_layers - n_layers
        for idx, child in enumerate(self.model.children()):
            if unfreeze is False:
                if idx < first_unfrozen:
                    for param in child.parameters():
                        param.requires_grad = False
            else:
                if idx >= first_unfrozen:
                    for param in child.parameters():
                        param.requires_grad = True

    def print_stats(self):
        total, total_frozen = 0, 0
        for pr in self.model.parameters():
            total += 1
            if pr.requires_grad is False:
                total_frozen += 1
        prints.print_trained_parameters_count(self.model)
        print("Total layers: ", total)
        print("Frozen layers: ", total_frozen)
