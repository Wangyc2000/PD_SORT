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


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


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


# 计算伪深度损失矩阵:检测框和轨迹之间伪深度之差的绝对值
def cost_pd_level(detections, previous_obs, pdl_weight):
    pd_level_num = 8
    pd_obs = previous_obs[:, 5]
    min_pd_obs = np.min(pd_obs)
    max_pd_obs = np.max(pd_obs)
    len_pd_obs = max_pd_obs - min_pd_obs + 1e-6
    pd_obs = (pd_obs - min_pd_obs) / len_pd_obs  # norm
    pd_obs_level = np.zeros(pd_obs.shape[0])
    previous_min = 1.
    for lev in range(pd_level_num):
        current_min = 1. - 1. * (lev + 1) / pd_level_num
        pd_obs_level[np.logical_and(pd_obs >= current_min, pd_obs <= previous_min)] = current_min + 1. / pd_level_num
        previous_min = current_min
    pd_obs_level = np.expand_dims(pd_obs_level, 0)
    pd_dets = detections[:, 5]
    min_pd_dets = np.min(pd_dets)
    max_pd_dets = np.max(pd_dets)
    len_pd_dets = max_pd_dets - min_pd_dets + 1e-6
    pd_dets = (pd_dets - min_pd_dets) / len_pd_dets  # norm
    pd_dets_level = np.zeros(pd_dets.shape[0])
    previous_min = 1.
    for lev in range(pd_level_num):
        current_min = 1. - 1. * (lev + 1) / pd_level_num
        pd_dets_level[np.logical_and(pd_dets >= current_min, pd_dets <= previous_min)] = current_min + 1. / pd_level_num
        previous_min = current_min
    pd_dets_level = np.expand_dims(pd_dets_level, 1)
    pd_level_diff = np.abs(pd_dets_level - pd_obs_level)

    # pd_obs = np.expand_dims(pd_obs, 0)
    # pd_dets = np.expand_dims(pd_dets, 1)
    # pd_level_diff = np.abs(pd_dets - pd_obs)  # 不分级，效果稍差

    pd_level_cost = 1. - pd_level_diff
    # pd_level_cost = ((1. - pd_level_diff) - 1./pd_level_num)/(1. - 1./pd_level_num)  # 不影响结果

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0  # <0应该是指为-1，即没有已记录的previous obs
    valid_mask = np.repeat(valid_mask[:, np.newaxis], detections.shape[0], axis=1).T  # 转化为矩阵形式
    pd_level_cost = pd_level_cost * valid_mask * pdl_weight

    return pd_level_cost


def associate(detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight):
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    # 计算IoU矩阵并获取置信度得分
    # iou_matrix = iou_batch(detections, trackers)  # 检测框与轨迹之间的IoU矩阵
    iou_matrix = iou_batch_3d(detections, trackers)  # ** 检测框与轨迹之间的3DIoU矩阵 **
    # iou_matrix = iou_matrix * scores  # a trick sometiems works, we don't encourage this

    # ** 可以增加速度方向with p_depth velocities(轨迹历史检测框之间的速度):vy,vx,vpd**
    Y, X, PD = speed_direction_batch(detections,
                                     previous_obs)  # 检测框中心点和轨迹中心点之间的方向向量（dy/斜边,dx/斜边，dpd/斜边）（num_track*num_det）
    inertia_Y, inertia_X, inertia_PD = velocities[:, 0], velocities[:, 1], velocities[:,
                                                                           2]  # 轨迹中现有的最后两个检测框中心点之间的方向向量（dy/斜边,dx/斜边，dpd/斜边）
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    inertia_PD = np.repeat(inertia_PD[:, np.newaxis], PD.shape[1], axis=1)

    # ** using the original 2D vel consist **
    angle_diff_cost_XY = cost_vel(Y, X, inertia_Y, inertia_X, detections, trackers, previous_obs, vdc_weight)
    angle_diff_cost = angle_diff_cost_XY

    # ** 3D vel: From ablation, we found our 3d-vdc works only for dancetrack-val, but not for mot17-val**
    # 向量夹角计算公式
    # diff_direction_cos = inertia_X * X + inertia_Y * Y + inertia_PD * PD
    # norm_inertia = np.sqrt(inertia_X**2 + inertia_Y**2 + inertia_PD**2)
    # norm_inter = np.sqrt(X**2 + Y**2 + PD**2)
    # diff_direction_cos /= (norm_inertia*norm_inter + 1e-6)  # 向量模长都是1，就不用再除模了
    # diff_direction_cos = np.clip(diff_direction_cos, a_min=-1, a_max=1)
    # diff_direction = np.arccos(diff_direction_cos)
    # diff_direction = 1 - diff_direction / np.pi
    # scores = np.repeat(detections[:, -2][:, np.newaxis], trackers.shape[0],
    #                    axis=1)  # 检测框的置信度分数（得detection中的倒数第二维，并拓展为矩阵）
    # valid_mask = np.ones(previous_obs.shape[0])
    # valid_mask[np.where(previous_obs[:, 4] < 0)] = 0  # <0应该是指为-1，即没有已记录的previous obs
    # valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)  # 转化为矩阵形式
    # angle_diff_cost = (valid_mask * diff_direction) * vdc_weight
    # angle_diff_cost = angle_diff_cost.T
    # angle_diff_cost = angle_diff_cost * scores  # 检测框与轨迹之间的角度（速度方向）*检测框的置信度，应该是越大越好

    # ** 不使用速度方向一致性损失 **
    # angle_diff_cost = 0

    # ** pdepth的损失 **
    pdl_weight = 0.2  # dance
    # pdl_weight = 0.36  # mot20
    # pdl_weight = 0.34  # mot20 w/o CMC
    # pdl_weight = 0.2  # mot17
    pd_level_cost = np.zeros_like(iou_matrix)
    if detections.shape[0] > 0 and previous_obs.shape[0] > 0:
        pd_level_cost = cost_pd_level(detections, previous_obs, pdl_weight)

    if min(iou_matrix.shape) > 0:
        # 筛选IoU矩阵中>阈值iou_threshold的部分设置为1，其余为0
        a = (iou_matrix > iou_threshold).astype(np.int32)
        # 每一行和每一列都只有一个1，说明检测框和轨迹已经完成了一一对应
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # 否则，加上angle_diff_cost进行线性分配
            # matched_indices = linear_assignment(-(iou_matrix + angle_diff_cost))
            matched_indices = linear_assignment(-(iou_matrix+angle_diff_cost+pd_level_cost))  # 加上了伪深度损失
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

