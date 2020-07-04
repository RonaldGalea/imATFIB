from pathlib import Path
import numpy as np
import nibabel as nib
import shutil


def train_val_split(input_folder, k_split=5):
    """
    Args:
    k_split: int - ratio of train to val samples, if 0 everything goes in train
    Split imogen or mmwhs segmentation dataset in train and val

    we need the path to the individual input samples only, the ground truth labels will be
    matched automatically later
    """
    split = {'train': [], 'val': []}
    total = []
    for element in input_folder.glob('**/*'):
        if element.is_file():
            if element.parent.stem == 'img' or element.parent.stem == 'image':
                total.append(element)

    if k_split == 0:
        for file_path in total:
            split['train'].append(file_path)
        return split

    for i, file_path in enumerate(total):
        if i % k_split == 0:
            split['val'].append(file_path)
        else:
            split['train'].append(file_path)

    return split
