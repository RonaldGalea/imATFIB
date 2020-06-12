"""
Prepare models and data for visualization of results
"""
import constants
from utils.params import Params, validate_params
from utils.training_utils import training_setup, prints


def prepare(model_ids, dataset_name):
    """
    Args:
    model_ids: list of str - ids of models for inference
    dataset_name: str - will construct validation loader of the dataset

    ! Important:
    models are expected to have same settings regarding dataset and training (eg roi, input res)

    Returns:
    loaded models ready for inference in a list
    validation_loader
    params
    """
    models = []
    for id in model_ids:
        params = Params(constants.params_path.format(dataset_name, id))
        validate_params(params)
        print("Constructing model: ", id)
        model = training_setup.model_setup(dataset_name, params)
        # in this case, model id is equal to exp name
        training_setup.load_model_weights(model, dataset_name, id)
        model.eval()
        prints.print_trained_parameters_count(model)
        models.append(model)

    # all models should have the same dataset related settings, so any params should do
    validation_loader = training_setup.prepare_val_loader(dataset_name, params)

    return models, validation_loader, params
