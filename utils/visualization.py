import numpy as np
import cv2
from albumentations import Resize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy

from utils import inference
from utils.ROI_crop import roi_crop
from utils.training_utils import training_setup
from utils import metrics
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
    models, params_list, configs_list = prepare_models.prepare(model_ids, dataset_name)

    model_preds, model_dices = [], []
    for model, params, config in zip(models, params_list, configs_list):
        # get a dataloader
        validation_loader = training_setup.prepare_val_loader(params, config)

        # run inference
        validation_loader.dataset.visualization_mode = True
        preds, dices = inference.run_model_inference(validation_loader, model, params, config)

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
    model_dice_lists - the dices for each slice
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

        for idx, (inp, msk) in enumerate(zip(orig_volume, mask)):
            total += 1
            inp = inp.astype(np.float32)

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
                overlays.append(base_colored + colored_mask)
                overlay_names.append(caption)

            image_list.extend(overlays)
            name_list.extend(overlay_names)
            show_images(image_list, cols=len(image_list)//2, titles=name_list, save_id=save_id)


def visualize_img_mask_pair_2d(image, mask, img_name='img', mask_name='mask', use_orig_res=False,
                               wait=False):
    """
    Args:
    image - ndarray: HxW image
    mask - ndarray: HxW label

    Return:
    """
    if not use_orig_res:
        resize = Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC)
        augmented = resize(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']

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
    # pred_colors = [(0, 1, 1), (1, 140 / 255, 0), (1, 1, 0)]
    pred_colors = [(0, 1, 0), (1, 140 / 255, 0), (1, 1, 0)]
    labels, colors = [], []
    for idx, c in enumerate(range(1, n_classes)):
        labels.append("ground truth class - " + str(c))
        labels.append("prediction class - " + str(c))
        labels.append("intersection class - " + str(c))
        colors.append(gt_colors[idx])
        colors.append(pred_colors[idx])
        colors.append([x+y for x, y in zip(gt_colors[idx], pred_colors[idx])])

    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    fig.set_figheight(10)
    fig.set_figwidth(19)
    plt.tight_layout()

    plt.savefig(save_id, dpi=fig.dpi)
    plt.close()


def visualize_pred_coords(volume, mask, coords, scores, params, config):
    print("unique pred elements: ", np.unique(volume))
    for i in range(volume.shape[0]):
        c_image = cv2.resize(volume[i], dsize=(512, 512)) * 200
        c_mask = mask[i]

        pos_1 = c_mask == 1
        pos_2 = c_mask == 2
        pos_3 = c_mask == 3
        dummy_mask = np.zeros((c_mask.shape))
        dummy_mask[pos_1] = 200.0
        dummy_mask[pos_2] = 180.0
        dummy_mask[pos_3] = 255.0
        dummy_mask = cv2.resize(dummy_mask, dsize=(c_image.shape))

        x_min, y_min, x_max, y_max = coords[i]
        params_cpy = copy.deepcopy(params)
        params_cpy.relative_roi_perturbation = [4, 20]
        setup = roi_crop.get_roi_crop_setup(params_cpy, config)
        print("In visualization QDFQWEF", params_cpy.relative_roi_perturbation, setup)
        gx_min, gy_min, gx_max, gy_max = roi_crop.compute_ROI_coords(
            dummy_mask, params_cpy, setup, validation=True)

        print("Pred coords :", x_min, y_min, x_max, y_max)
        print("GT coords :", gx_min, gy_min, gx_max, gy_max)

        c_image = cv2.rectangle(c_image, (x_min, y_min), (x_max, y_max), 1000, 1)
        c_image = cv2.rectangle(c_image, (gx_min, gy_min), (gx_max, gy_max), 1000, 1)

        dummy_mask = cv2.rectangle(dummy_mask, (x_min, y_min), (x_max, y_max), 1000, 1)
        dummy_mask = cv2.rectangle(dummy_mask, (gx_min, gy_min), (gx_max, gy_max), 1000, 1)

        classes = list(np.unique(c_mask))
        classes.pop(0)
        dice = metrics.metrics(mask[i], volume[i], classes=classes)

        print("Score: ", scores[i])
        print("Dice: ", dice, np.mean(dice))
        print("unique mask slice elems", np.unique(c_mask))

        visualize_img_mask_pair_2d(
            c_image, dummy_mask, str(np.mean(dice)), "after_mask", use_orig_res=True, wait=True)
        cv2.destroyAllWindows()


def color_a_mask(mask, type="prediction"):
    """
    mask - H x W ndarray
    type - str - use colors for prediction or ground truth

    Colors used for ground truth
        red - (255, 0, 0)
        green - (0, 255, 0)
        blue - (0, 0, 255)

    Colors used for prediction
        cyan - (0, 255, 255)
        orange - (255,140,0)
        yellow - (255,255,0)
    """
    gt_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # pred_colors = [(0, 1, 1), (1, 140 / 255, 0), (1, 1, 0)]
    pred_colors = [(0, 1, 0), (1, 140 / 255, 0), (1, 1, 0)]

    current_colors = gt_colors
    if type == "prediction":
        current_colors = pred_colors

    classes = list(np.unique(mask))
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
