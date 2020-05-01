from pathlib import Path
import numpy as np
import nibabel as nib
import shutil


def nii_to_npy(root_src, destination):
    """
    Args:
    root_src: pathlib.Path - path to root source directory
    destination: string - name of the desired destination directory

    Copies all contents of root_src, maintaining directory structure, changing .nii and .nii.gz
    files to .npy files

    with the exception of 4d files in acdc which aren't relevant for segmentation

    Works for imogen, mmwhs and acdc
    """
    root_name = root_src.parts[-1]
    root_dest = Path(str(root_src).replace(root_name, destination))
    root_dest.mkdir(exist_ok=True)
    for element in root_src.glob("**/*"):
        new_element = Path(str(element).replace(root_name, destination))
        if element.is_dir():
            new_element.mkdir(exist_ok=True)
        if element.is_file() and '4d' not in element.stem:
            if element.suffix == '.cfg':
                shutil.move(element, new_element)
            else:
                image = np.array(nib.load(element).dataobj)
                name_with_ext = new_element.parts[-1]
                only_name = name_with_ext.split('.')[0]
                new_element = Path(str(new_element).replace(name_with_ext, only_name))
                np.save(new_element, image)


def train_val_split(input_folder, k_split=5, split_train_val=True):
    """
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

    for i, file_path in enumerate(total):
        if i % k_split == 0 and split_train_val is True:
            split['val'].append(file_path)
        else:
            split['train'].append(file_path)

    return split
