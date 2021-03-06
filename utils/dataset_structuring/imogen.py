from pathlib import Path
import numpy as np
import nibabel as nib


def multi_class_label(dir_name):
    """
    Constructs multi-class label for imogen data


    Args:
    dir_name: string - imogen_patients_3d or imogen_control_3d

    Takes separate labels and creates one multi_class label
    Each heart element will have its own id, as they go from 1-6 (total nr)
    Background will be 0

    Currently there's a small issue, the labels are not exactly perfect and some different
    classes overlap, so we'll see about that
    """
    root = Path.cwd() / 'datasets' / 'whs_imatfib' / dir_name / 'gt'
    imogen_3d_aorta = root / 'aorta'
    ids = []
    for file in imogen_3d_aorta.glob("*"):
        ids.append(file.stem + file.suffix)

    multiple_classes_dir = root / 'multiple_classes'
    multiple_classes_dir.mkdir(exist_ok=True)

    not_single_label = ['multiple_classes', 'oneregion']
    for img_id in ids:
        multi_class = None
        unique = 1
        for mask_path in root.glob("**/*{}".format(img_id)):
            if mask_path.parent.stem not in not_single_label:
                print("Current mask path:", mask_path)
                mask = np.array(nib.load(mask_path).dataobj) * unique
                if multi_class is None:
                    multi_class = mask
                else:
                    multi_class += mask
                print("Unique elements in current mask: ", np.unique(mask))
                print("Unique elements in multi_class: ", np.unique(multi_class), "\n")
                unique += 1

        mask_nifti = nib.Nifti1Image(multi_class, affine=np.eye(4))
        nib.save(mask_nifti, multiple_classes_dir / img_id)

        print("Done with id: ", img_id, "\n")
        # visualization.visualize_img_mask_pair(np.ones(multi_class.shape), multi_class)
