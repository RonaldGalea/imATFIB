import numpy as np
import nibabel as nib
from pathlib import Path
from collections import namedtuple

from utils.dataset_structuring import acdc, general
import constants

Info = namedtuple('Info', ['affine', 'header'])


def get_train_val_paths(dataset_name, k_split):
    """
    This function splits the samples from the dataset directory in two sets: train and val,
    creating two Dataset objects using them

    For the ACDC dataset the split is fixed, exactly as done by Baumgarter et al.
    For imogen and mmwhs, the split factor is controlled by k_split
    """
    dataset_dir = Path.cwd() / 'datasets' / dataset_name
    if constants.acdc_root_dir == dataset_name or constants.acdc_test_dir == dataset_name:
        split_dict = acdc.acdc_train_val_split(dataset_dir)
    else:
        split_dict = general.train_val_split(dataset_dir, k_split=k_split)

    return split_dict


def read_img_mask_pair(image_path, dset_name=constants.acdc_root_dir,
                       seg_type=constants.multi_class_seg):
    """
    Args:
    image_path - pathlib.Path: path to image
    dset_name - string: since the original datasets have different structures (and i don't want
    to modify them, paths will have to be contructed accordingly)
    seg_type - string: multi class or whole heart

    Finds the corresponding ground truth label for each input
    Loads files

    return:
    ndarray: image and mask pair
    """
    if constants.imatfib_root_dir in dset_name:
        # construct path to the label in imatfib dir structure
        mask_path = image_path.parent.parent / 'gt'
        if seg_type == constants.whole_heart_seg:
            mask_path = mask_path / 'oneregion'
        else:
            mask_path = mask_path / seg_type
        mask_path = mask_path / (image_path.stem + image_path.suffix)

    elif constants.mmwhs_root_dir in dset_name:
        parts = image_path.stem.split('.')
        name = ''.join(parts[0]+'mapped.' + parts[1])
        mask_path = image_path.parent.parent / 'ground-truth' / \
            seg_type / (name + image_path.suffix)
    else:
        # add _gt to get the path to the label
        name_with_ext = image_path.parts[-1]
        only_name = name_with_ext.split('.')[0]
        gt_name_with_ext = only_name + '_gt.nii.gz'
        mask_path = Path(str(image_path).replace(name_with_ext, gt_name_with_ext))

    image, _ = read_image(image_path, type="pred")
    mask, info = read_image(mask_path)

    return image, mask, info


def read_image(image_path, type="gt"):
    image_info = nib.load(image_path)
    image = np.array(image_info.dataobj)
    if type == "pred":
        image = image.astype(np.float32)

    info = Info(image_info.affine, image_info.header)

    return image, info
