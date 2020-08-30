import numpy as np
import cv2
from albumentations import Resize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import constants

from utils import inference
from utils.training_utils import training_setup
from utils import prepare_models


height, width = 400, 400
exist_label = False


def save_models_results(model_ids, dataset_name):
    """
    Args:
    model_ids - list of strings
    dataset_name - string
    """
    # prepare models
    if len(model_ids) > 1:
        models, params_list, configs_list = prepare_models.prepare(model_ids, dataset_name)
    else:
        model, params, config = prepare_models.prepare(model_ids, dataset_name)
        models, params_list, configs_list = [model], [params], [config]

    model_preds, model_dices = [], []
    for model, params, config in zip(models, params_list, configs_list):
        # get a dataloader
        validation_loader = training_setup.prepare_val_loader(params, config)

        # run inference
        validation_loader.dataset.visualization_mode = True
        preds, dices = inference.run_model_inference(validation_loader, model, params)

        model_preds.append(preds)
        model_dices.append(dices)

    # save predictions
    save_predictions(validation_loader, model_ids, dataset_name, model_preds, model_dices)


def save_predictions(dataloader, model_names, dataset_name, model_pred_lists, model_dice_lists=None):
    """
    dataloader - to get the original input and masks
    model_names - to print which models are being used
    dataset_name - str
    model_pred_list - the predictions of each model
    model_dice_lists (optional) - the dices for each slice
    """
    # note vij - volume j predicted by model i
    # currently we have a list of lists with each models predictions: so v11, v21, v31... v12, v22...
    # a reordering is desired such that v11, v12, v13, v14... v21, v22, v23, v24...
    n_models, n_volumes = len(model_names), len(dataloader)

    volumes_pred = [[None for i in range(n_models)] for j in range(n_volumes)]
    dice_values = [[None for i in range(n_models)] for j in range(n_volumes)]

    for i in range(n_volumes):
        for j in range(n_models):
            volumes_pred[i][j] = model_pred_lists[j][i]
            if model_dice_lists:
                dice_values[i][j] = model_dice_lists[j][i]

    total = 0
    for vol_idx, (_, mask, _, orig_volume) in enumerate(dataloader):

        c_pred_volume = volumes_pred[vol_idx]
        c_pred_dices = dice_values[vol_idx]

        # show results on the current volume
        orig_volume = orig_volume.cpu().numpy()
        mask = mask.cpu().numpy().astype(np.uint8)
        classes = list(np.unique(mask))

        for idx, (inp, msk) in enumerate(zip(orig_volume, mask)):
            total += 1
            inp = inp.astype(np.float32)
            inp = cv2.resize(inp, dsize=(msk.shape[1], msk.shape[0]))

            colored_mask = color_a_mask(msk, type="gt")

            image_list = [inp, colored_mask]

            name_list = ["Input", "Mask"]
            save_id = "results/" + dataset_name + "/" + \
                "".join(model_names) + "/" + str(total) + "_" + str(idx) + "_" + str(vol_idx)

            overlays, overlay_names = [], []
            for pred, dice, model_name in zip(c_pred_volume, c_pred_dices, model_names):
                base = pred[idx]
                # convert pred to rgb and make color it
                base_colored = color_a_mask(base)
                image_list.append(base_colored)
                name_list.append(model_name)

                # overlay prediction on ground truth, with different color
                if dice:
                    caption = model_name + " " + "{:.3f}".format(float(np.mean(dice[idx])))
                else:
                    caption = model_name
                inter = get_inter(base, msk, classes)
                overlays.append(get_overlay(base_colored, colored_mask, inter))
                overlay_names.append(caption)

            image_list.extend(overlays)
            name_list.extend(overlay_names)
            show_images(image_list, cols=len(image_list)//2, titles=name_list,
                        save_id=save_id, n_classes=len(classes))


def visualize_img_mask_pair_2d(image, mask, img_name='img', mask_name='mask', use_orig_res=False,
                               wait=False):
    """
    Args:
    image - ndarray: HxW image
    mask - ndarray: HxW label

    Return:
    """
    print("Unique elements in original mask: ", np.unique(mask))
    if not use_orig_res:
        print("Original shape: ", image.shape)
        resize = Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC)
        augmented = resize(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        print("Visualization shape: ", image.shape)

    print("Unique labels in mask: ", np.unique(mask))
    image = (image * (255/(image.max()+1))).astype(np.uint8)
    mask = (mask * (255/(mask.max()+1))).astype(np.uint8)
    cv2.imshow(img_name, image)
    cv2.imshow(mask_name, mask)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def visualize_img_mask_pair(image_3d, mask_3d):
    """
    Args:
    image_3d - ndarray: HxWxC image_3d
    mask_3d - ndarray: HxWxC label_3d
    Return:
    """
    print("In visualization, image_3d shape: ", image_3d.shape)
    print("In visualization, Unique elements in mask_3d: ", np.unique(mask_3d), "\n")
    for i in range(image_3d.shape[2]):
        if exist_label or (mask_3d[:, :, i].max() > 0):

            current_img = image_3d[:, :, i]
            current_mask = mask_3d[:, :, i]
            visualize_img_mask_pair_2d(current_img, current_mask, wait=True)

    cv2.destroyAllWindows()


def show_images(images, cols=1, titles=None, save_id=None, n_classes=2):
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

    gt_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    pred_colors = [(1, 1, 0), (0, 1, 1), (1, 0, 1)]
    # pred_colors = [(0, 1, 0), (1, 140 / 255, 0), (1, 1, 0)]
    labels, colors = [], []
    for idx, c in enumerate(range(1, n_classes)):
        labels.append("ground truth class - " + constants.acdc_heart[idx])
        labels.append("prediction class - " + constants.acdc_heart[idx])
        labels.append("intersection class - " + constants.acdc_heart[idx])
        colors.append(gt_colors[idx])
        colors.append(pred_colors[idx])
        colors.append([(x+y)/2 for x, y in zip(gt_colors[idx], pred_colors[idx])])
    labels.append("miss/overshot/wrong class")
    colors.append((1, 1, 1))

    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    fig.set_figheight(10)
    fig.set_figwidth(19)
    plt.tight_layout()

    plt.savefig(save_id, dpi=fig.dpi)
    plt.close()


def color_a_mask(mask, type="prediction"):
    """
    mask - H x W ndarray
    type - str - use colors for prediction or ground truth

    Colors used for ground truth
        red - (255, 0, 0)
        green - (0, 255, 0)
        blue - (0, 0, 255)
    """
    gt_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    pred_colors = [(1, 1, 0), (0, 1, 1), (1, 0, 1)]
    # pred_colors = [(0, 1, 0), (1, 140 / 255, 0), (1, 1, 0)]

    current_colors = gt_colors
    if type == "prediction":
        current_colors = pred_colors

    classes = [0, 1, 2, 3]
    # not counting background
    classes.pop(0)

    colored_mask = np.zeros((*mask.shape, 3))

    # not counting background
    for idx, c_id in enumerate(classes):
        color_a_class(current_colors[idx], c_id, colored_mask, mask)

    return colored_mask


def color_a_class(color_code, class_id, colored_mask, mask):
    """
    colored_mask - H x W x C ndarray
    mask - H x W ndarray

    the colored mask is modified in-place
    """
    class_indices = mask == class_id
    for idx, intensity in enumerate(color_code):
        c_intensity_sheet = colored_mask[:, :, idx]
        c_intensity_sheet[class_indices] = intensity


def get_overlay(msk1, msk2, inter):
    """
    To keep the same colors outside the intersection, only average the intersection area
    """
    over = np.zeros((msk1.shape))
    over[inter] = (msk1[inter] + msk2[inter]) / 2

    # overshoot, miss, wrong label
    over[~inter] = (1, 1, 1)

    return over


def get_inter(msk1, msk2, classes):
    inters_for_classes = []
    for c in classes:
        indices1 = msk1 == c
        indices2 = msk2 == c
        inters_for_classes.append(indices1 * indices2)

    inter_all = np.zeros(msk1.shape).astype(np.bool)
    for inter_c in inters_for_classes:
        inter_all = inter_c + inter_all

    return inter_all
