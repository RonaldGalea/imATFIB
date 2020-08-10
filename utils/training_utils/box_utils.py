import torch


def wh2corners(bbox):
    """
    bbox - batch x 4
    """
    ctr = bbox[:, :2]
    wh = bbox[:, 2:]
    return torch.cat([ctr-wh/2, ctr+wh/2], dim=1)


def corners_to_wh(bbox):
    """
    (x_left, y_left, x_right, y_right) --> (x_ctr, y_ctr, width, height)
    """
    center_wh = torch.zeros(bbox.shape).to(bbox.device)
    center_wh[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2
    center_wh[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2
    center_wh[:, 2] = bbox[:, 2] - bbox[:, 0]
    center_wh[:, 3] = bbox[:, 3] - bbox[:, 1]

    return center_wh


def compute_offsets(gt, anchors):
    """
    Procedure from https://arxiv.org/abs/1311.2524

    gt - batch x 4 tensor
    anchors - #anchors x 4 tensor (center width height format)
    """
    gt = corners_to_wh(gt)

    # want to bring both target and anchors in shape batch x #anchors x 4
    # because anchors stay the same for each batch, and gt box stays the same for each anchor
    gt = gt.unsqueeze(1)
    anchors = anchors.unsqueeze(0)

    scale_xy = 10
    scale_wh = 5
    off_xy = scale_xy * (gt[:, :, :2] - anchors[:, :, :2])/anchors[:, :, 2:]
    off_wh = scale_wh * (gt[:, :, 2:] / anchors[:, :, 2:]).log()
    return torch.cat((off_xy, off_wh), dim=2).contiguous()


def convert_offsets_to_bboxes(pred_offsets, anchor):
    """
    Procedure from https://arxiv.org/abs/1311.2524

    pred_offsets - batch x 4
    anchor - batch x 4
    """
    scale_xy = 10
    scale_wh = 5

    pred_offsets[:, :2] = (1/scale_xy)*pred_offsets[:, :2]
    pred_offsets[:, 2:] = (1/scale_wh)*pred_offsets[:, 2:]

    pred_offsets[:, :2] = pred_offsets[:, :2] * anchor[:, 2:] + anchor[:, :2]
    pred_offsets[:, 2:] = pred_offsets[:, 2:].exp() * anchor[:, 2:]

    return pred_offsets


def get_intersection_area(pred, targ):
    """
    pred - batch x 4 tensor
    targ - batch x 4 tensor
    """
    # join columns, get the max and the min values to get the intersection bbox
    # unsqueeze to be able to concat along a new dimension
    max_x = torch.max(torch.cat([pred[:, 0].unsqueeze(1), targ[:, 0].unsqueeze(1)], dim=1), dim=1)[
        0].unsqueeze(1)
    max_y = torch.max(torch.cat([pred[:, 1].unsqueeze(1), targ[:, 1].unsqueeze(1)], dim=1), dim=1)[
        0].unsqueeze(1)
    min_x = torch.min(torch.cat([pred[:, 2].unsqueeze(1), targ[:, 2].unsqueeze(1)], dim=1), dim=1)[
        0].unsqueeze(1)
    min_y = torch.min(torch.cat([pred[:, 3].unsqueeze(1), targ[:, 3].unsqueeze(1)], dim=1), dim=1)[
        0].unsqueeze(1)

    inter_boxes = torch.cat([max_x, max_y, min_x, min_y], dim=1)
    return get_bbox_area(inter_boxes)


def get_bbox_area(bbox):
    """
    bbox - (batch x 4) tensor

    retruns - (batch) tensor
    """
    width = bbox[:, 2] - bbox[:, 0]
    height = bbox[:, 3] - bbox[:, 1]
    area = width * height

    invalid = (width < 0) * (height < 0)

    # set invalid boxes to have 0 area
    area[invalid] = 0

    return area


def get_IoU(pred, targ, intersection_area=None):
    """
    pred - batch x 4 tensor
    targ - batch x 4 tensor
    """
    if intersection_area is None:
        intersection_area = get_intersection_area(pred, targ)
    union_area = get_bbox_area(pred) + get_bbox_area(targ) - intersection_area
    invalid = (union_area == 0)

    # just to be able to divide
    union_area[invalid] = 1

    return (intersection_area / union_area) * (~invalid)


def get_gt_for_anchors(coords, anchors, threshold=0.55):
    """
    coords - batch x 4
    anchors - #anc x 4

    result - batch x #anc
    """
    # print("IN GT FOR ANCH")
    # print("coords", coords)
    # print("anchors", anchors)
    gt_for_anchors = []
    for gt_box in coords:
        # broadcast this box to the shape of anchors
        gt_box_br = torch.zeros(anchors.shape)
        gt_box_br[:] = gt_box

        # get iou for each anchor
        inter_area = get_intersection_area(gt_box_br, anchors)

        ious = get_IoU(gt_box_br, anchors, inter_area)
        # print("ious", ious)

        # get the indices of anchors that should predict the gt
        gt_for_anchor = ious > threshold

        # set the max to predict the gt regardless
        gt_for_anchor[torch.argmax(ious)] = 1

        gt_for_anchors.append(gt_for_anchor)

    gt_for_anchors = torch.stack(gt_for_anchors)
    gt_for_anchors = gt_for_anchors.to(torch.float32)
    # compare = torch.ones(coords.shape[0], dtype=torch.float32).unsqueeze(1)
    #
    # print("GT FOR ANCHORS", gt_for_anchors)
    # print("compare", compare)
    # assert torch.all(torch.eq(gt_for_anchors, compare)) == True

    return gt_for_anchors
