from pathlib import Path
import numpy as np
import nibabel as nib


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
            split['val'].append(file_path)
        return split

    for i, file_path in enumerate(total):
        if i % k_split == 0:
            split['val'].append(file_path)
        else:
            split['train'].append(file_path)

    return split


def mmwhs_gt_mapping(input_folder):
    """
    get ground truth of mmwhs to be values 0 - #classes-1
    """
    total = []
    mapping = {500: 1, 205: 2, 600: 3, 420: 4, 421: 4, 550: 5, 820: 6, 850: 7}
    for element in input_folder.glob('**/*'):
        if element.is_file():
            if element.parent.stem == 'multiple-classes':
                total.append(element)

    # read each volume and perform mapping
    for img_path in total:
        nimg = nib.load(img_path)
        img, affine, header = nimg.get_data(), nimg.affine, nimg.header
        print("Labels before: ", np.unique(img))

        for k, v in mapping.items():
            img[img == k] = v
        print("Labels after: ", np.unique(img))

        nimg_mapped = nib.Nifti1Image(img, affine=affine, header=header)

        name_with_ext = img_path.parts[-1]
        only_name = name_with_ext.split('.')[0]
        mapped_name = only_name + 'mapped.nii.gz'
        save_path = Path(str(img_path).replace(name_with_ext, mapped_name))

        print(save_path)

        nimg_mapped.to_filename(save_path)

        # mmwhs_heart = ["LVC", "LVMyo", "RVC", "LA", "RA", "AA", "PA"]
        # (1) the left ventricle blood cavity (label value 500);
        # (2) the right ventricle blood cavity (label value 600);
        # (3) the left atrium blood cavity (label value 420);
        # (4) the right atrium blood cavity (label value 550);
        # (5) the myocardium of the left ventricle (label value 205);
        # (6) the ascending aorta (label value 820);
        # (7) the pulmonary artery (label value 850);
