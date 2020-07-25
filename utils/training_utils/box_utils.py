import torch


import general_config


def wh2corners(bbox):
    ctr = bbox[:, :2]
    wh = bbox[:, 2:]
    return torch.cat([ctr-wh/2, ctr+wh/2], dim=1)


def corners_to_wh(bbox):
    """
    (x_left, y_left, x_right, y_right) --> (x_ctr, y_ctr, width, height)
    """
    center_wh = torch.zeros(bbox.shape).to(general_config.device)
    center_wh[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2
    center_wh[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2
    center_wh[:, 2] = bbox[:, 2] - bbox[:, 0]
    center_wh[:, 3] = bbox[:, 3] - bbox[:, 1]

    return center_wh


def compute_offsets(gt, anchor):
    """
    Procedure from https://arxiv.org/abs/1311.2524

    gt - batch x 4 tensor (top left, bottom right format)
    anchor - x 4 tensor (center width height format)
    """
    gt = corners_to_wh(gt)

    # print("In compute offsets: ")
    # print("Target xyWH", gt)

    scale_xy = 10
    scale_wh = 5
    off_xy = scale_xy * (gt[:, :2] - anchor[:, :2])/anchor[:, 2:]
    off_wh = scale_wh * (gt[:, 2:] / anchor[:, 2:]).log()
    return torch.cat((off_xy, off_wh), dim=1).contiguous()


def convert_offsets_to_bboxes(pred_offsets, anchor):
    """
    Procedure from https://arxiv.org/abs/1311.2524
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
    max_x = torch.max(torch.cat([pred[:, 0].unsqueeze(1), targ[:, 0].unsqueeze(1)], dim=1), dim=1)[0].unsqueeze(1)
    max_y = torch.max(torch.cat([pred[:, 1].unsqueeze(1), targ[:, 1].unsqueeze(1)], dim=1), dim=1)[0].unsqueeze(1)
    min_x = torch.min(torch.cat([pred[:, 2].unsqueeze(1), targ[:, 2].unsqueeze(1)], dim=1), dim=1)[0].unsqueeze(1)
    min_y = torch.min(torch.cat([pred[:, 3].unsqueeze(1), targ[:, 3].unsqueeze(1)], dim=1), dim=1)[0].unsqueeze(1)

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


def get_IoU(pred, targ, intersection_area):
    """
    pred - batch x 4 tensor
    targ - batch x 4 tensor
    """
    union_area = get_bbox_area(pred) + get_bbox_area(targ) - intersection_area
    invalid = (union_area == 0)

    # just to be able to divide
    union_area[invalid] = 1

    return (intersection_area / union_area) * (~invalid)
