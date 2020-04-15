import numpy as np
import nibabel as nib


def get_img_gt_pair(image_path, gts_path):
    stem, suffix = image_path.stem, image_path.suffix
    gt_path = gts_path / (stem + suffix)
    print(image_path)
    print(gt_path, "\n")

    image = np.array(nib.load(image_path).dataobj)
    gt = np.array(nib.load(gt_path).dataobj)

    return image, gt
