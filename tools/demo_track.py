import argparse
import os
import os.path as osp
import time
import cv2
import torch
from loguru import logger
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from trackers.ocsort_tracker.ocsort import OCSort
# from trackers.ocsort_tracker_ori.ocsort import OCSort
from trackers.tracking_utils.timer import Timer
from utils.args import make_parser
import numpy as np

# IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# def get_image_list(path):
#     image_names = []
#     for maindir, subdir, file_name_list in os.walk(path):
#         for filename in file_name_list:
#             apath = osp.join(maindir, filename)
#             ext = osp.splitext(apath)[1]
#             if ext in IMAGE_EXT:
#                 image_names.append(apath)
#     return image_names

# def image_demo(predictor, vis_folder, current_time, args):
#     if osp.isdir(args.path):
#         files = get_image_list(args.path)
#     else:
#         files = [args.path]
#     files.sort()
#     tracker = OCSort(det_thresh=args.track_thresh, iou_threshold=args.iou_thresh, use_byte=args.use_byte)
#     timer = Timer()
#     results = []
#
#     for frame_id, img_path in enumerate(files, 1):
#         outputs, img_info = predictor.inference(img_path, timer)
#         if outputs[0] is not None:
#             online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
#             online_tlwhs = []
#             online_ids = []
#             for t in online_targets:
#                 tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
#                 tid = t[4]
#                 vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
#                 if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
#                     online_tlwhs.append(tlwh)
#                     online_ids.append(tid)
#                     results.append(
#                         f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n"
#                     )
#             timer.toc()
#             online_im = plot_tracking(
#                 img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
#             )
#         else:
#             timer.toc()
#             online_im = img_info['raw_img']
#
#         # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
#         if args.save_result:
#             timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
#             save_folder = osp.join(vis_folder, timestamp)
#             os.makedirs(save_folder, exist_ok=True)
#             cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)
#
#         if frame_id % 20 == 0:
#             logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
#
#         ch = cv2.waitKey(0)
#         if ch == 27 or ch == ord("q") or ch == ord("Q"):
#             break
#
#     if args.save_result:
#         res_file = osp.join(vis_folder, f"{timestamp}.txt")
#         with open(res_file, 'w') as f:
#             f.writelines(results)
#         logger.info(f"save results to {res_file}")


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            trt_file=None,
            decoder=None,
            device=torch.device("cpu"),
            fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        # if trt_file is not None:
        #     from torch2trt import TRTModule
        #
        #     model_trt = TRTModule()
        #     model_trt.load_state_dict(torch.load(trt_file))
        #
        #     x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
        #     self.model(x)
        #     self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            # if self.decoder is not None:
            #     outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info


def imageflow_demo(predictor, vis_folder, current_time, args):
    # 创建VideoCapture对象cap，可以用于从摄像头或视频文件中捕获视频帧
    cap = cv2.VideoCapture(args.path if args.demo_type == "video" else args.camid)
    # 通过cap对象获得视频帧的宽度高度和帧率信息
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 记录当前时间
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    # 创建当前时间进行的实验的输出文件的存储目录
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo_type == "video":
        save_path = args.out_path
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    # 创建一个视频写入器对象 vid_writer，用于将图像帧写入视频文件,保存到save_path下
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    # 初始化跟踪器OCSort
    tracker = OCSort(det_thresh=args.track_thresh, iou_threshold=args.iou_thresh, use_byte=args.use_byte)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        # 每20帧输出一次log
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        # 读入当前帧 ret_val 表示是否成功读取帧，frame 包含读取的图像数据。
        ret_val, frame = cap.read()
        # 若还有可读帧，则用目标检测模型 (predictor) 对当前帧进行推理（包括预处理、模型预测、后处理），返回检测结果 outputs 和图像信息 img_info。
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            img_tensor = None
            tag = f"MOT17-04-FRCNN:{frame_id}"
            # 若有检测结果,outputs[0]的形状为[detection_num, 7]，包含当前帧检测到的detection_num个检测框的左上、右下坐标信息、回归置信度、分类置信度、应该还有类别（0）
            if outputs[0] is not None:
                # 使用目标跟踪器 (tracker) 来更新跟踪目标的状态。tracker.update 方法接受检测结果、图像尺寸和测试尺寸作为参数，返回当前帧跟踪到的目标。
                online_targets = tracker.update(outputs[0], img_tensor, frame,[img_info['height'], img_info['width']], exp.test_size, tag)
                # online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    # 边界框宽高比是否大于给定的阈值
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    # 若边界框面积大于命令行传入的最小阈值且宽高比也大于给定阈值，则记录这些边界框的信息及对应跟踪目标的ID，在当前帧中画出来
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        # 记录当前帧中的结果到results中
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n"
                        )
                timer.toc()
                # 将满足阈值筛选条件的边界框记录在原始帧图像中画出来并生成一帧图像online_im
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if args.save_result:
                # 若需保存结果则将当前帧写入目标文件
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1
    # 若需保存结果则将当前帧的result信息写入目标文件
    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(exp, args):
    # 创建输出文件目录
    if not args.expn:
        args.expn = exp.exp_name
    output_dir = osp.join(exp.output_dir, args.expn)
    os.makedirs(output_dir, exist_ok=True)
    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    # if args.trt:
    #     args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    # 按照命令行传入的参数更新实验参数
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    # 获取检测器模型，并设置为评估模式
    detection_model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(detection_model, exp.test_size)))
    detection_model.eval()

    # 读取预训练的模型权重（checkpoint）
    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        detection_model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    # 在评估阶段融合模型中的层，从而减少参数量和计算成本
    if args.fuse:
        logger.info("\tFusing model...")
        detection_model = fuse_model(detection_model)
    # 将模型转换为半精度表示，从而减少模型的大小和计算成本，但可能会导致精度损失。
    if args.fp16:
        detection_model = detection_model.half()  # to FP16

    # if args.trt:
    #     assert not args.fuse, "TensorRT model is not support model fusing!"
    #     trt_file = osp.join(output_dir, "model_trt.pth")
    #     assert osp.exists(
    #         trt_file
    #     ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
    #     model.head.decode_in_inference = False
    #     decoder = model.head.decode_outputs
    #     logger.info("Using TensorRT to inference")
    # else:
    #     trt_file = None
    #     decoder = None
    trt_file = None
    decoder = None

    predictor = Predictor(detection_model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    # if args.demo_type == "image":
    #     image_demo(predictor, vis_folder, current_time, args)
    # elif args.demo_type == "video" or args.demo_type == "webcam":
    #     imageflow_demo(predictor, vis_folder, current_time, args)
    if args.demo_type == "video" or args.demo_type == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)
    else:
        logger.info("please use a video demo or webcam")


if __name__ == "__main__":
    # utils/args.py中定义了make_parser()，包含实验所使用的参数，args接收命令行输入的参数
    args = make_parser().parse_args()
    # 通过文件路径或文件名获得实验文件，并创建实验对象
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
