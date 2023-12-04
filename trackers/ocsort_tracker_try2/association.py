import os
import numpy as np


def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
              + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    return (o)


def iou_batch_3d(dets, tracks):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(tracks, 0)
    bboxes1 = np.expand_dims(dets, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    pd = np.minimum(bboxes1[..., 5], bboxes2[..., 4])
    pd2 = bboxes2[..., 4]
    pd1 = bboxes1[..., 5]
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    pd = np.maximum(0., pd)
    whd = w * h * pd
    o = whd / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) * pd1
               + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) * pd2 - whd)

    # if(((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) * pd1
    #            + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) * pd2 - whd) == 0).any():
    #     print("bboxes2:",bboxes2,"\n","bboxes1:",bboxes1,"\n","w:",w,"\n","h:",h,"\n","pd:",pd,"\n")
    return (o)

def iou_batch_3d_2(dets, last_obs):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(last_obs, 0)
    bboxes1 = np.expand_dims(dets, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    pd = np.minimum(bboxes1[..., 5], bboxes2[..., 5])
    pd2 = bboxes2[..., 5]
    pd1 = bboxes1[..., 5]
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    pd = np.maximum(0., pd)
    whd = w * h * pd
    o = whd / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) * pd1
               + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) * pd2 - whd)
    if (((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) * pd1
         + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) * pd2 - whd) == 0).any():
        print("bboxes2:", bboxes2, "\n", "bboxes1:", bboxes1, "\n", "w:", w, "\n", "h:", h, "\n", "pd:", pd, "\n")
    return (o)

def giou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
             + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    iou = wh / union

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    wc = xxc2 - xxc1
    hc = yyc2 - yyc1
    assert ((wc > 0).all() and (hc > 0).all())
    area_enclose = wc * hc
    giou = iou - (area_enclose - union) / area_enclose
    giou = (giou + 1.) / 2.0  # resize from (-1,1) to (0,1)
    return giou


def diou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # calculate the intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
             + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    iou = wh / union
    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    diou = iou - inner_diag / outer_diag

    return (diou + 1) / 2.0  # resize from (-1,1) to (0,1)


def ciou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # calculate the intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
             + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    iou = wh / union

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2

    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1]
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1]

    # prevent dividing over zero. add one pixel shift
    h2 = h2 + 1.
    h1 = h1 + 1.
    arctan = np.arctan(w2 / h2) - np.arctan(w1 / h1)
    v = (4 / (np.pi ** 2)) * (arctan ** 2)
    S = 1 - iou
    alpha = v / (S + v)
    ciou = iou - inner_diag / outer_diag - alpha * v

    return (ciou + 1) / 2.0  # resize from (-1,1) to (0,1)


def ct_dist(bboxes1, bboxes2):
    """
        Measure the center distance between two sets of bounding boxes,
        this is a coarse implementation, we don't recommend using it only
        for association, which can be unstable and sensitive to frame rate
        and object speed.
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    ct_dist2 = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    ct_dist = np.sqrt(ct_dist2)

    # The linear rescaling is a naive version and needs more study
    ct_dist = ct_dist / ct_dist.max()
    return ct_dist.max() - ct_dist  # resize to (0,1)


def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    CX2, CY2 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    PD1 = dets[:, 5]
    PD2 = tracks[:, 5]
    dx = CX1 - CX2
    dy = CY1 - CY2
    # ** 加入检测和轨迹之间的伪深度之差**
    dpd = PD1 - PD2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    # norm = np.sqrt(dx ** 2 + dy ** 2 + dpd ** 2) + 1e-6  # ** 加入检测和轨迹之间的伪深度之差**
    dx = dx / norm
    dy = dy / norm
    dpd = dpd / norm  # ** 加入检测和轨迹之间的伪深度之差**
    return dy, dx, dpd  # size: num_track x num_det


# 计算伪深度损失矩阵:检测框和轨迹之间伪深度之差的绝对值
def cal_pdepth_diff_batch(dets, tracks):
    tracks = np.expand_dims(tracks, 1)
    dets = np.expand_dims(dets, 0)

    pd2 = tracks[..., 4]
    pd1 = dets[..., 5]

    diff = abs(pd2 - pd1)
    diff /= np.max(diff)

    return diff


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


# def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
#     """
#     Assigns detections to tracked object (both represented as bounding boxes)
#     Returns 3 lists of matches, unmatched_detections and unmatched_trackers
#     """
#     if(len(trackers)==0):
#         return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
#
#     iou_matrix = iou_batch(detections, trackers)
#
#     if min(iou_matrix.shape) > 0:
#         a = (iou_matrix > iou_threshold).astype(np.int32)
#         if a.sum(1).max() == 1 and a.sum(0).max() == 1:
#             matched_indices = np.stack(np.where(a), axis=1)
#         else:
#             matched_indices = linear_assignment(-iou_matrix)
#     else:
#         matched_indices = np.empty(shape=(0,2))
#
#     unmatched_detections = []
#     for d, det in enumerate(detections):
#         if(d not in matched_indices[:,0]):
#             unmatched_detections.append(d)
#     unmatched_trackers = []
#     for t, trk in enumerate(trackers):
#         if(t not in matched_indices[:,1]):
#             unmatched_trackers.append(t)
#
#     #filter out matched with low IOU
#     matches = []
#     for m in matched_indices:
#         if(iou_matrix[m[0], m[1]]<iou_threshold):
#             unmatched_detections.append(m[0])
#             unmatched_trackers.append(m[1])
#         else:
#             matches.append(m.reshape(1,2))
#     if(len(matches)==0):
#         matches = np.empty((0,2),dtype=int)
#     else:
#         matches = np.concatenate(matches,axis=0)
#
#     return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def cost_vel(Y, X, inertia_Y, inertia_X, detections, trackers, previous_obs, vdc_weight):
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)  # 计算轨迹和检测框之间的角度（弧度制）矩阵
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi  # 归一化角度
    scores = np.repeat(detections[:, -2][:, np.newaxis], trackers.shape[0],
                       axis=1)  # 检测框的置信度分数（得detection中的倒数第二维，并拓展为矩阵）
    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0  # <0应该是指为-1，即没有已记录的previous obs
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)  # 转化为矩阵形式

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores  # 检测框与轨迹之间的角度（速度方向）*检测框的置信度，应该是越大越好

    return angle_diff_cost


def associate(detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight):
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    # 计算IoU矩阵并获取置信度得分
    # scores = np.repeat(detections[:,-1][:, np.newaxis], trackers.shape[0], axis=1)  # 检测框的置信度分数（得detection中的最后一维，并拓展为矩阵）
    scores = np.repeat(detections[:, -2][:, np.newaxis], trackers.shape[0],
                           axis=1)  # 检测框的置信度分数（得detection中的倒数第二维，并拓展为矩阵）
    median_score = np.median(scores[:, 0], axis=0)
    if (median_score > 0.89):
        iou_matrix = iou_batch_3d(detections, trackers)  # ** 检测框与轨迹之间的3DIoU矩阵 **
    else:
        iou_matrix = iou_batch(detections, trackers)  # 检测框与轨迹之间的IoU矩阵

    # iou_matrix = iou_batch(detections, trackers)  # 检测框与轨迹之间的IoU矩阵
    # iou_matrix = iou_batch_3d(detections, trackers)  # ** 检测框与轨迹之间的3DIoU矩阵 **

    # id (np.median(scores[:,0], axis=0) > C):

    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this

    # ** 可以增加速度方向with p_depth velocities(轨迹历史检测框之间的速度):vy,vx,vpd**
    # Y, X = speed_direction_batch(detections, previous_obs)  # 检测框和轨迹之间的中心点坐标sin,cos值矩阵（num_track*num_det）
    Y, X, PD = speed_direction_batch(detections,
                                     previous_obs)  # 检测框中心点和轨迹中心点之间的方向向量（dy/斜边,dx/斜边，dpd/斜边）（num_track*num_det）
    # inertia_Y, inertia_X = velocities[:,0], velocities[:,1] # 轨迹中现有的最后两个检测框中心点之间的sin，cos值
    inertia_Y, inertia_X, inertia_PD = velocities[:, 0], velocities[:, 1], velocities[:,
                                                                           2]  # 轨迹中现有的最后两个检测框中心点之间的方向向量（dy/斜边,dx/斜边，dpd/斜边）
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    inertia_PD = np.repeat(inertia_PD[:, np.newaxis], PD.shape[1], axis=1)
    # diff_angle_cos = inertia_X * X + inertia_Y * Y
    # # # diff_angle_cos = inertia_X * X + inertia_Y * Y + inertia_PD * PD
    # diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    # diff_angle = np.arccos(diff_angle_cos)  # 计算轨迹和检测框之间的角度（弧度制）矩阵
    # # # 但第二帧三还没有经过update，没有历史观测和速度，不知道这种关联是否合理，或许可以试试在第二帧时使用简单的IoU
    # diff_angle = (np.pi /2.0 - np.abs(diff_angle)) / np.pi  # 归一化角度
    # valid_mask = np.ones(previous_obs.shape[0])
    # valid_mask[np.where(previous_obs[:,4]<0)] = 0  # <0应该是指为-1，即没有已记录的previous obs
    # valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)  # 转化为矩阵形式
    # # 使用valid_mask来过滤不合理的角度计算，并乘上传入的权重vdc_weight
    # angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    # angle_diff_cost = angle_diff_cost.T
    # angle_diff_cost = angle_diff_cost * scores  # 检测框与轨迹之间的角度（速度方向）*检测框的置信度，应该是越大越好

    # ** 分别计算X-Y,X-PD,Y-PD三个方向上的速度方向并取平均 **
    # X-Y
    # angle_diff_cost_XY = cost_vel(Y, X, inertia_Y, inertia_X, detections, trackers, previous_obs, vdc_weight)
    # min-max归一化
    # if np.max(angle_diff_cost_XY)>0:
    #     angle_diff_cost_XY = (angle_diff_cost_XY-np.min(angle_diff_cost_XY))/(np.max(angle_diff_cost_XY)-np.min(angle_diff_cost_XY))
    # X-PD
    # min-max归一化
    # angle_diff_cost_XPD = cost_vel(PD, X, inertia_PD, inertia_X, detections, trackers, previous_obs, vdc_weight)
    # if np.max(angle_diff_cost_XPD) > 0:
    #     angle_diff_cost_XPD = (angle_diff_cost_XPD - np.min(angle_diff_cost_XPD)) / (
    #                 np.max(angle_diff_cost_XPD) - np.min(angle_diff_cost_XPD))
    # Y-PD
    # angle_diff_cost_YPD = cost_vel(PD, Y, inertia_PD, inertia_Y, detections, trackers, previous_obs, vdc_weight)
    # min-max归一化
    # if np.max(angle_diff_cost_XPD) > 0:
    #     angle_diff_cost_XPD = (angle_diff_cost_XPD - np.min(angle_diff_cost_XPD)) / (
    #                 np.max(angle_diff_cost_XPD) - np.min(angle_diff_cost_XPD))
    # angle_diff_cost = angle_diff_cost_XY
    # 取平均速度方向差
    # angle_diff_cost = (angle_diff_cost_XY + angle_diff_cost_XPD + angle_diff_cost_YPD)/3
    # angle_diff_cost = (angle_diff_cost_XY + angle_diff_cost_XPD + angle_diff_cost_YPD)  # 参照hybridsort，不取均值
    # angle_diff_cost = 0.6*angle_diff_cost_XY + 0.2*angle_diff_cost_XPD + 0.2*angle_diff_cost_YPD

    # 向量夹角计算公式
    # diff_direction_cos = inertia_X * X + inertia_Y * Y
    diff_direction_cos = inertia_X * X + inertia_Y * Y + inertia_PD * PD
    # 向量模长都是1，就不用再除模了
    diff_direction_cos = np.clip(diff_direction_cos, a_min=-1, a_max=1)
    diff_direction = np.arccos(diff_direction_cos)
    diff_direction = 1 - diff_direction / np.pi
    scores = np.repeat(detections[:, -2][:, np.newaxis], trackers.shape[0],
                       axis=1)  # 检测框的置信度分数（得detection中的倒数第二维，并拓展为矩阵）
    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0  # <0应该是指为-1，即没有已记录的previous obs
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)  # 转化为矩阵形式
    angle_diff_cost = (valid_mask * diff_direction) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores  # 检测框与轨迹之间的角度（速度方向）*检测框的置信度，应该是越大越好

    # if (median_score > 0.9):
    #     diff_direction_cos = inertia_X * X + inertia_Y * Y + inertia_PD * PD
    #     # 向量模长都是1，就不用再除模了
    #     diff_direction_cos = np.clip(diff_direction_cos, a_min=-1, a_max=1)
    #     diff_direction = np.arccos(diff_direction_cos)
    #     diff_direction = 1 - diff_direction / np.pi
    #     scores = np.repeat(detections[:, -2][:, np.newaxis], trackers.shape[0],
    #                        axis=1)  # 检测框的置信度分数（得detection中的倒数第二维，并拓展为矩阵）
    #     valid_mask = np.ones(previous_obs.shape[0])
    #     valid_mask[np.where(previous_obs[:, 4] < 0)] = 0  # <0应该是指为-1，即没有已记录的previous obs
    #     valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)  # 转化为矩阵形式
    #     angle_diff_cost = (valid_mask * diff_direction) * vdc_weight
    #     angle_diff_cost = angle_diff_cost.T
    #     angle_diff_cost = angle_diff_cost * scores  # 检测框与轨迹之间的角度（速度方向）*检测框的置信度，应该是越大越好  # ** 检测框与轨迹之间的3DIoU矩阵 **
    # else:
    #     angle_diff_cost_XY = cost_vel(Y, X, inertia_Y, inertia_X, detections, trackers, previous_obs, vdc_weight)  # 检测框与轨迹之间的IoU矩阵
    #     angle_diff_cost = angle_diff_cost_XY


    # ** pdepth的损失 **
    # pdepth_diff_cost = cal_pdepth_diff_batch(detections, trackers).T
    if min(iou_matrix.shape) > 0:
        # 筛选IoU矩阵中>阈值iou_threshold的部分设置为1，其余为0
        a = (iou_matrix > iou_threshold).astype(np.int32)
        # 每一行和每一列都只有一个1，说明检测框和轨迹已经完成了一一对应
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # 否则，加上angle_diff_cost进行线性分配
            matched_indices = linear_assignment(-(iou_matrix + angle_diff_cost))
            # matched_indices = linear_assignment(-(iou_matrix+angle_diff_cost+pdepth_diff_cost))  # 加上了伪深度损失
    else:
        matched_indices = np.empty(shape=(0, 2))

    # 找到未成功关联的检测框和轨迹
    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        # 找到成功匹配但IoU小于阈值的轨迹和检测框，同样视作未成功关联
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        # 高于阈值，则视为关联成功
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    # 返回成功关联的检测框-轨迹索引对，未成功关联的检测框索引，未关联成功的轨迹索引
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

# def associate_kitti(detections, trackers, det_cates, iou_threshold,
#         velocities, previous_obs, vdc_weight):
#     if(len(trackers)==0):
#         return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
#
#     """
#         Cost from the velocity direction consistency
#     """
#     Y, X = speed_direction_batch(detections, previous_obs)
#     inertia_Y, inertia_X = velocities[:,0], velocities[:,1]
#     inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
#     inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
#     diff_angle_cos = inertia_X * X + inertia_Y * Y
#     diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
#     diff_angle = np.arccos(diff_angle_cos)
#     diff_angle = (np.pi /2.0 - np.abs(diff_angle)) / np.pi
#
#     valid_mask = np.ones(previous_obs.shape[0])
#     valid_mask[np.where(previous_obs[:,4]<0)]=0
#     valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)
#
#     scores = np.repeat(detections[:,-1][:, np.newaxis], trackers.shape[0], axis=1)
#     angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
#     angle_diff_cost = angle_diff_cost.T
#     angle_diff_cost = angle_diff_cost * scores
#
#     """
#         Cost from IoU
#     """
#     iou_matrix = iou_batch(detections, trackers)
#
#
#     """
#         With multiple categories, generate the cost for catgory mismatch
#     """
#     num_dets = detections.shape[0]
#     num_trk = trackers.shape[0]
#     cate_matrix = np.zeros((num_dets, num_trk))
#     for i in range(num_dets):
#             for j in range(num_trk):
#                 if det_cates[i] != trackers[j, 4]:
#                         cate_matrix[i][j] = -1e6
#
#     cost_matrix = - iou_matrix - angle_diff_cost - cate_matrix
#
#     if min(iou_matrix.shape) > 0:
#         a = (iou_matrix > iou_threshold).astype(np.int32)
#         if a.sum(1).max() == 1 and a.sum(0).max() == 1:
#             matched_indices = np.stack(np.where(a), axis=1)
#         else:
#             matched_indices = linear_assignment(cost_matrix)
#     else:
#         matched_indices = np.empty(shape=(0,2))
#
#     unmatched_detections = []
#     for d, det in enumerate(detections):
#         if(d not in matched_indices[:,0]):
#             unmatched_detections.append(d)
#     unmatched_trackers = []
#     for t, trk in enumerate(trackers):
#         if(t not in matched_indices[:,1]):
#             unmatched_trackers.append(t)
#
#     #filter out matched with low IOU
#     matches = []
#     for m in matched_indices:
#         if(iou_matrix[m[0], m[1]]<iou_threshold):
#             unmatched_detections.append(m[0])
#             unmatched_trackers.append(m[1])
#         else:
#             matches.append(m.reshape(1,2))
#     if(len(matches)==0):
#         matches = np.empty((0,2),dtype=int)
#     else:
#         matches = np.concatenate(matches,axis=0)
#
#     return matches, np.array(unmatched_detections), np.array(unmatched_trackers)