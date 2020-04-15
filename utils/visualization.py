import numpy as np
import cv2

from utils import reading


def visualize_img_gt_pair(image, gt, exist_label=False):
    """
    Args:


    Return:
    """
    print("Image shape: ", image.shape)
    print("Unique elements in gt: ", np.unique(gt), "\n")
    for i in range(image.shape[2]):
        if exist_label or (gt[:, :, i].max() > 0):

            print("Max value of slice: ", image[:, :, i].max())
            print("Unique elements in slice gt: ", np.unique(gt[:, :, i]))

            current = (image[:, :, i] * (255/image[:, :, i].max())).astype(np.uint8)
            current_gt = (gt[:, :, i] * (255/gt.max())).astype(np.uint8)

            cv2.imshow("img", current)
            cv2.imshow("gt", current_gt)
            cv2.waitKey(0)
            break

    cv2.destroyAllWindows()


def visualize_dataset(images_path, gts_path):
    """
    Args:


    Return:
    """
    for image_path in images_path.glob('**/*'):
        image, gt = reading.get_img_gt_pair(image_path, gts_path)
        visualize_img_gt_pair(image, gt)
