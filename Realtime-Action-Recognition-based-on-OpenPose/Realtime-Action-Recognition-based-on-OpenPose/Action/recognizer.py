# -*- coding: UTF-8 -*-
import numpy as np
import cv2 as cv
from pathlib import Path
from Tracking.deep_sort import preprocessing
from Tracking.deep_sort.nn_matching import NearestNeighborDistanceMetric
from Tracking.deep_sort.detection import Detection
from Tracking import generate_dets as gdet
from Tracking.deep_sort.tracker import Tracker
from keras.models import load_model
from .action_enum import Actions

# Use Deep-sort(Simple Online and Realtime Tracking)
# To track multi-person for multi-person actions recognition

# 定义基本参数
file_path = Path.cwd()
clip_length = 15
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0
fall_num = 0

# 初始化deep_sort
model_filename = str(file_path/'Tracking/graph_model/mars-small128.pb')
#对检测到的object path 编码
encoder = gdet.create_box_encoder(model_filename, batch_size=1) #encoder的索引 计算得到特征 对特征进行编码
#度量kalman预测的目标和下一帧的检测目标进行距离计算 使用余弦距离能够缓解遮挡 ID switch比较频繁的问题
metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)#
tracker = Tracker(metric) #根据度量结果追踪

# track_box颜色
trk_clr = (0, 255, 0)


# class ActionRecognizer(object):
#     @staticmethod
#     def load_action_premodel(model):
#         return load_model(model)
#
#     @staticmethod
#     def framewise_recognize(pose, pretrained_model):
#         frame, joints, bboxes, xcenter = pose[0], pose[1], pose[2], pose[3]
#         joints_norm_per_frame = np.array(pose[-1])
#
#         if bboxes:
#             bboxes = np.array(bboxes)
#             features = encoder(frame, bboxes)
#
#             # score to 1.0 here).
#             detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(bboxes, features)]
#
#             # 进行非极大抑制
#             boxes = np.array([d.tlwh for d in detections])
#             scores = np.array([d.confidence for d in detections])
#             indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
#             detections = [detections[i] for i in indices]
#
#             # 调用tracker并实时更新
#             tracker.predict()
#             tracker.update(detections)
#
#             # 记录track的结果，包括bounding boxes及其ID
#             trk_result = []
#             for trk in tracker.tracks:
#                 if not trk.is_confirmed() or trk.time_since_update > 1:
#                     continue
#                 bbox = trk.to_tlwh()
#                 trk_result.append([bbox[0], bbox[1], bbox[2], bbox[3], trk.track_id])
#                 # 标注track_ID
#                 trk_id = 'ID-' + str(trk.track_id)
#                 cv.putText(frame, trk_id, (int(bbox[0]), int(bbox[1]-45)), cv.FONT_HERSHEY_SIMPLEX, 0.8, trk_clr, 3)
#
#             for d in trk_result:
#                 xmin = int(d[0])
#                 ymin = int(d[1])
#                 xmax = int(d[2]) + xmin
#                 ymax = int(d[3]) + ymin
#                 # id = int(d[4])
#                 try:
#                     # xcenter是一帧图像中所有human的1号关节点（neck）的x坐标值
#                     # 通过计算track_box与human的xcenter之间的距离，进行ID的匹配
#                     tmp = np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter])
#                     j = np.argmin(tmp)
#                 except:
#                     # 若当前帧无human，默认j=0（无效）
#                     j = 0
#
#                 # 进行动作分类
#                 if joints_norm_per_frame.size > 0:
#                     joints_norm_single_person = joints_norm_per_frame[j*36:(j+1)*36]
#                     joints_norm_single_person = np.array(joints_norm_single_person).reshape(-1, 36)
#                     pred = np.argmax(pretrained_model.predict(joints_norm_single_person))
#                     init_label = Actions(pred).name
#                     # 显示动作类别
#                     cv.putText(frame, init_label, (xmin + 80, ymin - 45), cv.FONT_HERSHEY_SIMPLEX, 1, trk_clr, 3)
#                 # 画track_box
#                 cv.rectangle(frame, (xmin - 10, ymin - 30), (xmax + 10, ymax), trk_clr, 2)
#         return frame

def load_action_premodel(model):
    return load_model(model)

#输入的是要检测的图片  返回区域左上 右下坐标值
def draw_region_detection(frame):
    x_l = int(frame.shape[1] * 0.4)
    y_l = int(frame.shape[0]*0.2)
    x_r = int(frame.shape[1] * 0.68)
    y_r = frame.shape[0] - 10
    cv.rectangle(frame, (x_l,y_l), (x_r,y_r),(0, 0, 255),thickness=1)
    return x_l,y_l,x_r,y_r

def framewise_recognize(pose, pretrained_model):
    global fall_num
    frame, joints, bboxes, xcenter = pose[0], pose[1], pose[2], pose[3]
    #frame是已经标记好骨骼点的图片
    joints_norm_per_frame = np.array(pose[-1])
    # Ano_list = draw_region_detection(frame) #得到检测范围坐标
    # boxss = bboxes  #box的格式为tlxmin tlymin width height
    # boxs = []
    # # # 标定检测范围的代码
    # for i in range(len(boxss)):
    #     bb = boxss[i]
    #     if bb[0] > Ano_list[0] and bb[0] < Ano_list[2]:  # 判断要检测的范围 288,189
    #         boxs.append(boxss[i])  # 也就是不检测  不匹配
    # bboxes = boxs
    #追踪部分
    if bboxes:
        bboxes = np.array(bboxes)
        features = encoder(frame, bboxes) #获得128维的特征的编码 deep_sort中的 #每个box相当于一个patch
        #将patch送到deep sort的网络中去进行编码

        # score to 1.0 here).
        # 特征和bboxes对应 这个句话也就是转换了一下数据类型 也相当于建立了这个类型的实例
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(bboxes, features)]

        # 进行非极大抑制
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices] #选取最终的都的目标坐标

        # 调用tracker并实时更新
        # 第一帧作为初始化 后面根据前帧 后帧的距离判断 达到跟踪的目的
        tracker.predict()  #得到已有目标预测后的mean 和 covariance 用来作为判断依据
        tracker.update(detections) #是不是更新了跟踪的目标集合 检测到的目标集合

        # 记录track的结果，包括bounding boxes及其ID
        trk_result = []
        #好多利用for直接实例化的例子用法啊
        for trk in tracker.tracks: # 这是一个list 啊 对每个逐一判断
            if not trk.is_confirmed() or trk.time_since_update > 1:
                continue
            bbox = trk.to_tlwh() #tlah - > to top left coord and weight,height
            trk_result.append([bbox[0], bbox[1], bbox[2], bbox[3], trk.track_id]) #t,l
            # 标注track_ID
            trk_id = 'ID-' + str(trk.track_id)
            # cv.putText(frame, trk_id, (int(bbox[0]), int(bbox[1]-45)), cv.FONT_HERSHEY_SIMPLEX, 0.8, trk_clr, 2)

        for d in trk_result:
            xmin = int(d[0])
            ymin = int(d[1])
            xmax = int(d[2]) + xmin
            ymax = int(d[3]) + ymin
            # id = int(d[4])
            try:
                # xcenter是一帧图像中所有human的1号关节点（neck）的x坐标值
                # 通过计算track_box与human的xcenter之间的距离，进行ID的匹配
                tmp = np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter])
                j = np.argmin(tmp)
            except:
                # 若当前帧无human，默认j=0（无效）
                j = 0

            # 进行动作分类
            if joints_norm_per_frame.size > 0: #每个人具有36个维度 18个位置的关节点
                joints_norm_single_person = joints_norm_per_frame[j*36:(j+1)*36]
                joints_norm_single_person = np.array(joints_norm_single_person).reshape(-1, 36)
                pred = np.argmax(pretrained_model.predict(joints_norm_single_person)) #只是keras里的predict
                init_label = Actions(pred).name #得到enum标签的名称


                # 显示动作类别 ymin - 40
                # cv.putText(frame, init_label, (xmin + 30, ymax + 20), cv.FONT_HERSHEY_SIMPLEX, 1, trk_clr, 2)
                # 异常预警(under scene)
                if init_label == 'fall_down':
                    fall_num += 1
                    if fall_num > 2 :  #至少连续2帧出现异常时，才能出现报错的情况
                        fall_num = 2
                    if fall_num == 2:
                        cv.putText(frame, 'WARNING: someone is Falling down!', (20, 60), cv.FONT_HERSHEY_SIMPLEX,
                               1, (0, 0, 255), 3)
                else:
                    fall_num = 0
                print('fall_num:{}'.format(fall_num))
            # 画track_box
            # cv.rectangle(frame, (xmin - 10, ymin - 30), (xmax + 10, ymax), trk_clr, 2)
    return frame

