"""
Taken from Stanford DL course: http://cs230.stanford.edu/blog/hyperparameters/
"""

import json
import constants


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def validate_params(params):
    norm_type = params.norm_type
    data_augmentation = params.data_augmentation
    lr_decay = params.lr_decay
    roi_crop = params.roi_crop
    load_type = params.load_type
    freeze_type = params.freeze_type
    if norm_type not in constants.norm_types:
        raise AssertionError("Params not ok..." + "norm_type")
    if data_augmentation not in constants.aug_types:
        raise AssertionError("Params not ok..." + "data_augmentation")
    if lr_decay not in constants.lr_schedulers:
        raise AssertionError("Params not ok..." + "lr_decay")
    if roi_crop not in constants.roi_types:
        raise AssertionError("Params not ok..." + "roi_crop")
    if roi_crop != constants.no_roi_extraction and params.default_height != 256:
        print("ROI crop is used but the default height is not 256, is this want you truly want?")
    if load_type not in constants.load_types:
        raise AssertionError("Params not ok..." + "load_type")
    if freeze_type not in constants.freeze_types:
        raise AssertionError("Params not ok..." + "freeze_type")
