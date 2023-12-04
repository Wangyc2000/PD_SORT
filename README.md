# OCPD

#### 介绍
OC with PD

#### 软件架构
软件架构说明


#### 安装教程

1.  Download MOT17, MOT20, CrowdHuman, Cityperson, ETHZ, DanceTrack and put them under <OCSORT_HOME>/datasets in the following structure:


```
datasets
|——————mot
|        └——————train
|        └——————test
└——————crowdhuman
|        └——————Crowdhuman_train
|        └——————Crowdhuman_val
|        └——————annotation_train.odgt
|        └——————annotation_val.odgt
└——————MOT20
|        └——————train
|        └——————test
└——————Cityscapes
|        └——————images
|        └——————labels_with_ids
└——————ETHZ
|        └——————eth01
|        └——————...
|        └——————eth07
└——————dancetrack        
         └——————train
         └——————val
         └——————test
```

2.  Turn the datasets to COCO format and mix different training data:

```
# replace "dance" with ethz/mot17/mot20/crowdhuman/cityperson for others
python3 tools/convert_dance_to_coco.py 
```

3.  [Optional] If you want to training for MOT17/MOT20, follow the following to create mixed training set.

```
# build mixed training sets for MOT17 and MOT20 
python3 tools/mix_data_{ablation/mot17/mot20}.py
```


#### Evaluation

1.  DanceTrack
- DanceTrack-val:

```
# CUDA_VISIBLE_DEVICES=1 python tools/run_ocsort_dance.py -f exps/example/mot/yolox_dancetrack_val.py -c pretrained/ocsort_dance_model.pth.tar -b 1 -d 1 --fp16 --fuse
python tools/run_ocsort_dance.py -f exps/example/mot/yolox_dancetrack_val.py -c pretrained/ocsort_dance_model.pth.tar -b 1 -d 1 --fp16 --fuse

# HOTA(CLEAR Identity) evaluation
python3 external/TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL val --METRICS HOTA CLEAR Identity --GT_FOLDER datasets/dancetrack/val --SEQMAP_FILE gt/seqmaps/DANCE-val.txt --SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL ocsorttest --USE_PARALLEL False --PLOT_CURVES False --TRACKERS_FOLDER res_dancetrack/val
```

- DanceTrack-test:

```
python tools/run_ocsort_dance.py -f exps/example/mot/yolox_dancetrack_val.py -c pretrained/ocsort_dance_model.pth.tar -b 1 -d 1 --fp16 --fuse --test
```


2.  MOT17
- MOT17-val:

```
python3 tools/run_ocsort.py -f exps/yolox_x_ablation.py -c pretrained/ocsort_mot17_ablation.pth.tar -b 1 -d 1 --fp16 --fuse

# HOTA(CLEAR Identity) evaluation
python3 external/TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL val  --METRICS HOTA CLEAR Identity  --GT_FOLDER gt/MOT17-val --SEQMAP_FILE gt/seqmaps/MOT17-val.txt --SKIP_SPLIT_FOL True  --TRACKERS_TO_EVAL data --USE_PARALLEL False --PLOT_CURVES False --TRACKERS_FOLDER /home/wangyc/My_OC_SORT/res_mot/MOT17-val/yolox_x_ablation_results

```

- MOT17-test:

```
python3 tools/run_ocsort.py -f exps/example/mot/yolox_x_mix_det.py -c pretrained/ocsort_x_mot17.pth.tar -b 1 -d 1 --fp16 --fuse

# Linear interpolation "res_mot/MOT17-test/yolox_x_mix_det_results/data" is the original path "res_mot/MOT17-test/yolox_x_mix_det_results/data_li" is the interpolated path 
python3 tools/interpolation.py res_mot/MOT17-test/yolox_x_mix_det_results/data res_mot/MOT17-test/yolox_x_mix_det_results/data_li
```


3.  MOT20
- MOT20-val:

```
python3 tools/run_ocsort.py -f exps/yolox_x_ablation_mot20.py -c pretrained/ocsort_x_mot20.pth.tar -b 1 -d 1 --fp16 --fuse --mot20 --track_thresh 0.4
```

- MOT20-Test:

```
python3 tools/run_ocsort.py -f exps/example/mot/yolox_x_mix_mot20_ch.py -c pretrained/ocsort_x_mot20.tar -b 1 -d 1 --fp16 --fuse --mot20

# HOTA(CLEAR Identity) evaluation
python3 external/TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL val  --METRICS HOTA CLEAR Identity  --GT_FOLDER gt/MOT20-val --SEQMAP_FILE gt/seqmaps/MOT20-val.txt --SKIP_SPLIT_FOL True  --TRACKERS_TO_EVAL data --USE_PARALLEL False --PLOT_CURVES False --TRACKERS_FOLDER /home/wangyc/My_OC_SORT/res_mot/MOT20-val/yolox_x_ablation_mot20_results

```


#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
