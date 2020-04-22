import os
import glob
from pathlib import Path
import general_config


def acdc_train_val_split(input_folder, split_train_val=True):
    """
    Source:
    https://github.com/baumgach/acdc_segmenter/blob/master/acdc_data.py

    Get the same train and validation splits as Baumgarter et al. for comparison purposes
    """
    file_list = {'val': [], 'train': []}
    for folder in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder)

        if os.path.isdir(folder_path):

            if split_train_val:
                train_test = 'val' if (int(folder[-3:]) % 5 == 0) else 'train'
            else:
                train_test = 'train'

            search = 'patient???_frame??.nii.gz'
            if general_config.read_numpy:
                search = 'patient???_frame??.npy'

            for file in glob.glob(os.path.join(folder_path, search)):
                file_list[train_test].append(Path(file))

    return file_list
