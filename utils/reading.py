import numpy as np
import nibabel as nib
from pathlib import Path
import constants
from collections import namedtuple

Info = namedtuple('Info', ['affine', 'header'])


def get_img_mask_pair(image_path, numpy=False, dset_name=constants.acdc_root_dir,
                      seg_type=constants.multi_class_seg):
    """
    Args:
    image_path - pathlib.Path: path to image
    numpy - bool: if true this expects to find .npy files
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
        mask_path = image_path.parent.parent / 'ground-truth' / \
            seg_type / (image_path.stem + image_path.suffix)
    else:
        # add _gt to get the path to the label
        name_with_ext = image_path.parts[-1]
        only_name = name_with_ext.split('.')[0]
        gt_name_with_ext = only_name + '_gt.nii.gz'
        if numpy:
            gt_name_with_ext = only_name + '_gt.npy'
        mask_path = Path(str(image_path).replace(name_with_ext, gt_name_with_ext))

    # print("From reading: ", image_path)
    # print("From reading: ", mask_path, "\n")
    image_info, mask_info = nib.load(image_path), nib.load(mask_path)
    if numpy:
        image = np.load(image_path)
        mask = np.load(mask_path)
    else:
        image = np.array(image_info.dataobj)
        mask = np.array(mask_info.dataobj)
    # necessary information to save .nii file and compute metrics
    image = image.astype(np.float32)
    mask = mask.astype(np.float32)
    info = Info(mask_info.affine, mask_info.header)

    return image, mask, info
