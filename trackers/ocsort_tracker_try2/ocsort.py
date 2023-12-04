"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import numpy as np
from .association import *


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        # return [-1, -1, -1, -1, -1]
        return [-1, -1, -1, -1, -1, -1]  # ** 补充了一维p_depth **
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    # w = bbox[2] - bbox[0]
    # h = bbox[3] - bbox[1]
    # x = bbox[0] + w / 2.
    # y = bbox[1] + h / 2.
    # s = w * h  # scale is just area
    # r = w / float(h + 1e-6)
    # return np.array([x, y, s, r]).reshape((4, 1))

    # ** [x1,y1,x2,y2,score,pd]->[x,y,pd,s,r] **
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    pd = bbox[5]
    s = w * h  # scale is just area
    r = w / float(h + 1e-6)
    return np.array([x, y, pd, s, r]).reshape((5, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
       # w = np.sqrt(x[2] * x[3])
    # h = x[2] / w
    # if (score == None):
    #     return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    # else:
    #     return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

    # ** x: [x,y,pd,s,r,vx,vy,vpd,vs] -> bbox: [x1, y1, x2, y2, pd] **
    w = np.sqrt(x[3] * x[4])
    h = x[3] / w
    if (score == None):
        # return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., x[2]]).reshape((1, 5))
    else:
        # return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score, x[2]]).reshape((1, 6))


def speed_direction(bbox1, bbox2):
    # ** bbox: [x1, y1, x2, y2, score, p_depth] **
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0  # bbox1的中心点坐标
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0  # bbox1的中心点坐标
    pd1 = bbox1[5]  # bbox1的伪深度
    pd2 = bbox2[5]  # bbox2的伪深度
    speed = np.array([cy2 - cy1, cx2 - cx1, pd2 - pd1])  # y,x,pd三个方向上的速度
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6  # 用于标准化速度
    # norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2 + (pd2 - pd1) ** 2) + 1e-6  # 用于标准化速度
    return speed / norm


class KalmanBoxTracklet(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, delta_t=3, orig=False):
        """
        Initialises a tracklet using initial bounding box.

        """
        # bbox: [x1, y1, x2, y2, score, p_depth]
        # --------- 以下为：卡尔曼滤波器配置 ---------
        # define constant velocity model
        if not orig:
            from .kalmanfilter import KalmanFilterNew as KalmanFilter
            # ** 引入伪深度和伪深度方向上的速度 self.kf = KalmanFilter(dim_x=7, dim_z=4) **
            # x: [x,y,pd,s,r,vx,vy,vpd,vs] z: [x,y,pd,s,r]
            self.kf = KalmanFilter(dim_x=9, dim_z=5)
        else:
            from filterpy.kalman import KalmanFilter
            # self.kf = KalmanFilter(dim_x=7, dim_z=4)
            # ** 引入伪深度和伪深度方向上的速度 self.kf = KalmanFilter(dim_x=7, dim_z=4) **
            # x: [x,y,s,r,pd,vx,vy,vs,vpd] z: [x,y,s,r,pd]
            self.kf = KalmanFilter(dim_x=9, dim_z=5)
        # self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
        #     0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        # self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
        #                       [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        # self.kf.R[2:, 2:] *= 10.
        # self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        # self.kf.P *= 10.
        # self.kf.Q[-1, -1] *= 0.01
        # self.kf.Q[4:, 4:] *= 0.01

        # 卡尔曼滤波使用的矩阵，针对额外引入的pd，vpd进行修改
        #                      x, y,pd, s, r,vx,vy,vpd,vs
        self.kf.F = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0]])
        self.kf.R[3:, 3:] *= 10.  # 这里我不确定，我设置仍是遵循从s开始往后乘10
        # self.kf.R[2:, 2:] *= 10.  # 从pd开始乘10
        self.kf.P[5:, 5:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        # self.kf.Q[-2, -2] *= 0.01
        self.kf.Q[5:, 5:] *= 0.01
        # bbox: [x1, y1, x2, y2, score, p_depth]
        # self.kf.x[:4] = convert_bbox_to_z(bbox)  # 将x1y1x2y2转化为xysr
        self.kf.x[:5] = convert_bbox_to_z(bbox)  # 将x1y1x2y2scorepd转化为xypdsr

        # --------- 以上为：卡尔曼滤波器配置 ---------

        self.time_since_update = 0  # 距离上一次与新检测框匹配成功并进行状态更新的帧数
        self.id = KalmanBoxTracklet.count  # 新轨迹的ID
        KalmanBoxTracklet.count += 1  # 更新跟踪到的目标数量
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        # self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.last_observation = np.array([-1, -1, -1, -1, -1, -1])  # placeholder
        # self.last_observation = bbox  # 会怎么样？
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

        # 试一试用第一次的观测作为历史值
        # self.last_observation = bbox
        # self.observations[self.age] = bbox
        # self.history_observations.append(bbox)

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:
            #  bbox: [x1, y1, x2, y2, score, p_depth]
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)  # 两个观测的中心点之间的方向向量(y之差/斜边, x之差/斜边, pd/斜边) 用来在association中计算角度（速度方向一致性的cost）

            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            # 将新的观测更新到轨迹的历史观测中
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0  # 重置time_since_update
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            # self.kf.update(convert_bbox_to_z(bbox))  # 将新的观测以[x,y,s,r]的形式用于卡尔曼滤波的状态更新，更新了x（包括3维速度）,K,y,P等，
            self.kf.update(convert_bbox_to_z(bbox))  # 将新的观测以[x,y,pd,s,r]的形式用于卡尔曼滤波的状态更新，更新了x（包括3维速度）,K,y,P等，
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # ** 面积不可以小于等于0 **
        # if (self.kf.x[6] + self.kf.x[2]) <= 0:
        #     self.kf.x[6] *= 0.0
        if (self.kf.x[8] + self.kf.x[3]) <= 0:
            self.kf.x[8] *= 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)  # [x,y,pd,s,r] ->  [x1,y1,x2,y2,pd]


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {"iou": iou_batch,
              "giou": giou_batch,
              "ciou": ciou_batch,
              "diou": diou_batch,
              "ct_dist": ct_dist}


class OCSort(object):
    def __init__(self, det_thresh, max_age=30, min_hits=3,
                 iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, use_byte=False):
        """
        Sets key parameters for SORT
        (det_thresh=args.track_thresh, iou_threshold=args.iou_thresh, use_byte=args.use_byte)
        """
        self.max_age = max_age  # 轨迹在没有成功关联的情况下的最大存活帧数
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        # self.trackers = []
        self.tracklets = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte  # 是否使用BYTE进行第二轮（针对低置信度检测框）的关联
        KalmanBoxTracklet.count = 0  # 目前跟踪到的目标总数

    def update(self, output_results, img_info, img_size):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        # self.frame_count += 1

        # 检测器在当前帧没有检测到目标，则当前帧跟踪到的目标同样为空
        if output_results is None:
            return np.empty((0, 5))

        # 更新帧编号，按照OC_SORT和Hybrid_SORT的写法放在判断检测器是否检测到目标后面
        self.frame_count += 1
        # if(self.frame_count == 440 or self.frame_count==450):
        #     print("frame reached")
        # post_process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]  # 第5维是置信度分数
            bboxes = output_results[:, :4]  # 前4维是边界框左上和右下坐标（x1y1x2y2）
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]  # 第5和第6维是2个置信度分数，相乘作为最终的置信度分数
            bboxes = output_results[:, :4]  # x1y1x2y2
        # 计算原始图像放缩到test_size所需的比例scale，并对边界框按scale进行放缩
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        # ** 获取伪深度 **
        # p_depth = img_size[0] - (bboxes[:, 1]+bboxes[:, 3])/2  # 中心点到画面底部的距离
        # p_depth = img_size[0] - bboxes[:, 3]  # 边界框底部到画面底部的距离，即h-y2
        p_depth = img_info[0] - bboxes[:, 3]  # 边界框底部到画面底部的距离，即h-y2
        # p_depth = img_size[0] + img_size[0] - bboxes[:, 3] - (bboxes[:, 3] - bboxes[:, 1])  # 边界框底部到画面底部的距离 - h + img_size[0]
        # p_depth = bboxes[:, 3] - bboxes[:, 3]  # 0
        # p_depth = 2000 - bboxes[:, 3]  # 2000-y2 参考sparseTrack
        p_depth[p_depth <= 0] = 1e-6
        # p_depth[p_depth > img_info[0]] = img_info[0]


        # 将边界框坐标信息bbox(num_detections,4)和置信度score（num_detections）进行合并得到完整的检测结果（num_detections,5）
        # dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

        # ** 将边界框坐标信息bbox(num_detections,4)、置信度score（num_detections）和伪深度（p_depth）进行合并得到完整的检测结果（num_detections,6） **
        # ** dets: [x1, y1, x2, y2, score, p_depth] **
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1), np.expand_dims(p_depth, axis=-1)), axis=1)
        # 按照置信度划分关联优先级
        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]

        # get predicted locations from existing tracklets.
        # trks = np.zeros((len(self.tracklets), 5))  # 用于记录卡尔曼滤波对之前的帧中观测到的轨迹的预测结果
        trks = np.zeros((len(self.tracklets), 6))  # ** 用于记录卡尔曼滤波对之前的帧中观测到的轨迹的预测结果 **
        to_del = []  # 记录存在NaN值（缺失值）的轨迹
        ret = []  # 用于记录当前帧最终返回的跟踪结果
        for t, trk in enumerate(trks):
            pos = self.tracklets[t].predict()[0]
            # trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], 0]  # ** x1, y1, x2, y2, p_depth, 0 **
            if np.any(np.isnan(pos)):
                to_del.append(t)
        # 处理numpy数组中的无效值（例如NaN），然后删除无效值并返回一个新的数组
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        # 删除存在NaN值（缺失值）的轨迹
        for t in reversed(to_del):
            self.tracklets.pop(t)

        # velocities = np.array(
        #     [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.tracklets])

        # ** 增加一维深度方向的速度，均为标准化后的结果：(v_y, v_x, v_pd) **
        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0, 0, 0)) for trk in self.tracklets])

        last_boxes = np.array([trk.last_observation for trk in self.tracklets])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.tracklets])

        """
            First round of association
        """
        # 第一轮关联 IoU + 速度方向*置信度
        matched, unmatched_dets, unmatched_trks = associate(
            dets, trks, self.iou_threshold, velocities, k_observations, self.inertia)
        # 使用成功关联的检测框更新轨迹
        for m in matched:
            self.tracklets[m[1]].update(dets[m[0], :])

        """
            Second round of associaton by OCR
        """
        # BYTE association 针对低置信度检测卡的关联 仅使用IoU
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(dets_second, u_trks)  # iou between low score detections and unmatched tracks
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.tracklets[trk_ind].update(dets_second[det_ind, :])  # 使用成功关联的检测框更新轨迹
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))  # 关联成功的轨迹移出未关联轨迹集

        # 针对第一轮关联（高置信度检测框）中未成功进行关联的检测框进行额外一轮关联 仅使用IoU
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            # scores_left_dets = np.repeat(left_dets[:, -2][:, np.newaxis], left_trks.shape[0],
            #                    axis=1)  # 检测框的置信度分数（得detection中的倒数第二维，并拓展为矩阵）
            # median_score_left_dets = np.median(scores_left_dets[:, 0], axis=0)
            # if median_score_left_dets > 0.89:
            #     iou_left = iou_batch_3d_2(left_dets, left_trks)  # ** 检测框与轨迹之间的3DIoU矩阵 **
            # else:
            #     iou_left = self.asso_func(left_dets, left_trks)  # 检测框与轨迹之间的IoU矩阵
            iou_left = self.asso_func(left_dets, left_trks)
            # iou_left_1 = iou_batch_3d_2(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)  # 在这一轮中匹配成功的检测框和轨迹索引
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.tracklets[trk_ind].update(dets[det_ind, :])  # 使用成功关联的检测框更新轨迹
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))  # 关联成功的检测框移出未关联检测框集
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))  # 关联成功的轨迹移出未关联轨迹集

        for m in unmatched_trks:
            self.tracklets[m].update(None)

        # create and initialise new tracklets for unmatched detections
        # 将第一轮关联中未成功关联的高置信度检测框创建为新的轨迹  那么reupdate在哪？
        for i in unmatched_dets:
            trk = KalmanBoxTracklet(dets[i, :], delta_t=self.delta_t)
            self.tracklets.append(trk)
        i = len(self.tracklets)
        # 更新这一帧的跟踪结果
        # 将这一帧的成功跟踪结果放入ret，超过最大未关联帧数的轨迹删除
        for trk in reversed(self.tracklets):
            if trk.last_observation.sum() < 0:
                # x1,y1,x2,y2,pd
                # d = trk.get_state()[0]
                # ** x1,y1,x2,y2 **
                d = trk.get_state()[0, :4]
            else:
                """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # x1,y1,x2,y2，id+1 由于结果要求ID从1开始所以要id+1
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.tracklets.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

