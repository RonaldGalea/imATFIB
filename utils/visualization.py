import torch
import numpy as np
import cv2
from albumentations import Resize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import general_config
import constants
from utils.training_utils import training_processing


height, width, matplot = general_config.height, general_config.width, general_config.matplot
max_plot_nr, exist_label = general_config.max_plot_nr, general_config.exist_label


def visualize_validation_dataset(dataloader, models, params, model_names):
    """
    Args:
    dataloader: validation dataloader, has to return one volume at a time
    models: list of nn.Module objects - trained models

    This function provides visualization of the input, ground truth and model predictions
    """
    dataloader.dataset.visualization_mode = True
    total = 0
    for vol_idx, (volume, mask, r_infos, orig_volume) in enumerate(dataloader):

        # process current volume by each model
        final_preds, all_dices = [], []
        for model in models:
            volume, mask = volume.to(general_config.device), mask.to(general_config.device)
            # process volume slice by slice so we can see dice per each one
            dices, pred_slices = [], []
            for slice_idx, (vol_slice, mask_slice) in enumerate(zip(volume, mask)):
                # simulate bs = 1
                vol_slice = vol_slice.unsqueeze(0)
                mask_slice = mask_slice.unsqueeze(0)
                # this has softmax channels, (cuda) tensor
                if r_infos is not None:
                    r_info = [r_infos[slice_idx]]
                else:
                    r_info = None
                processed_volume = training_processing.process_volume(model, vol_slice, mask_slice,
                                                                      params, r_info)
                # this is final pred, cpu, np.uint8
                slice_dice, final_slice, _ = training_processing.compute_dice(processed_volume,
                                                                              mask_slice)
                dices.append(slice_dice)
                pred_slices.append(final_slice)

            # save current model prediction and stats
            pred_volume = np.concatenate(pred_slices)
            final_preds.append(pred_volume)
            all_dices.append(dices)

        # having run all models, show results on the current volume
        orig_volume = orig_volume.cpu().numpy()
        orig_volume = np.transpose(orig_volume, (2, 0, 1))
        # orig_volume = (orig_volume / 255).astype(np.float32)
        mask = mask.cpu().numpy().astype(np.uint8)

        for idx, (inp, msk) in enumerate(zip(orig_volume, mask)):
            inp = inp.astype(np.float32)
            msk = msk.astype(np.float32)
            # convert msk to rgb and make it green
            msk_green = np.zeros((*msk.shape, 3))
            msk_green[:, :, 1] = msk
            image_list = [inp, msk_green]
            name_list = ["Input", "Mask"]
            total += 1
            save_id = "results/" + params.roi_crop + "".join(model_names) + "/" + "_" + \
                str(total) + "_" + str(idx) + "_" + str(vol_idx)

            overlays, overlay_names = [], []
            for pred, dice, model_name in zip(final_preds, all_dices, model_names):
                base = pred[idx]
                # convert pred to rgb and make it blue
                base_green = np.zeros((*base.shape, 3))
                base_green[:, :, 0] = base
                image_list.append(base_green)
                name_list.append(model_name)

                # overlay prediction on ground truth, with different color
                caption = model_name + " " + "{:.3f}".format(float(dice[idx]))
                overlays.append(base_green + msk_green)
                overlay_names.append(caption)

            image_list.extend(overlays)
            name_list.extend(overlay_names)
            show_images(image_list, cols=len(image_list)//2, titles=name_list, save_id=save_id)


def visualize_img_mask_pair_2d(image, mask, img_name='img', mask_name='mask', use_orig_res=False):
    """
    Args:
    image - ndarray: HxW image
    mask - ndarray: HxW label

    Return:
    """
    print("In visualization, original shape ", image.shape)
    if not use_orig_res:
        resize = Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC)
        augmented = resize(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']

    image = (image * (255/image.max())).astype(np.uint8)
    mask = (mask * (255/mask.max())).astype(np.uint8)
    cv2.imshow(img_name, image)
    cv2.imshow(mask_name, mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_img_mask_pair(image_3d, mask_3d):
    """
    Args:
    image_3d - ndarray: HxWxC image_3d
    mask_3d - ndarray: HxWxC label_3d
    Return:
    """
    resize = Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC)
    print("In visualization, image_3d shape: ", image_3d.shape)
    print("In visualization, Unique elements in mask_3d: ", np.unique(mask_3d), "\n")
    for i in range(image_3d.shape[2]):
        if exist_label or (mask_3d[:, :, i].max() > 0):

            current_img = image_3d[:, :, i]
            current_mask = mask_3d[:, :, i]

            augmented = resize(image=current_img, mask=current_mask)
            current_img = augmented['image']
            current_mask = augmented['mask']

            current_img = (current_img * (255/current_img.max())).astype(np.uint8)
            current_mask = (current_mask * (255/mask_3d.max())).astype(np.uint8)
            cv2.imshow("img", current_img)
            cv2.imshow("mask", current_mask)
            cv2.waitKey(0)

    cv2.destroyAllWindows()


def show_images(images, cols=1, titles=None, save_id=None):
    """
    taken from https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as images.
    """
    assert((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(np.ceil(n_images/float(cols)), cols, n + 1)
        plt.imshow(image)
        ax.set_title(title)

    labels = ["ground truth", "prediction", "intersection"]
    colors = [(0, 1, 0), (1, 0, 0), (1, 1, 0)]

    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    fig.set_figheight(10)
    fig.set_figwidth(19)
    plt.tight_layout()

    if save_id:
        plt.savefig(save_id, dpi=fig.dpi)
    else:
        plt.show()
    plt.close()


def show_image2d(image, img_name="img", box_coords=None, unnorm=False):
    if unnorm:
        image = (image * general_config.dataset_std) + general_config.dataset_mean
    image = (image * (255/(image.max()+1))).astype(np.uint8)
    print("IN visu: ", type(image), image.shape, image.dtype)
    if box_coords:
        (x1, y1), (x2, y2) = box_coords
        image = cv2.rectangle(image, (x1, y1), (x2, y2), 255, 1)
    cv2.imshow(img_name, image)
