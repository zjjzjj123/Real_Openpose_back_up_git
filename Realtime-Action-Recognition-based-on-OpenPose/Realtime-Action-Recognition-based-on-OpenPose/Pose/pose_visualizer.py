# -*- coding: UTF-8 -*-
import cv2 as cv
import numpy as np
import tensorflow as tf
from .coco_format import CocoPart, CocoColors, CocoPairsRender
from .pose_estimator import estimate


# 输入参数'graph_opt_mobile.pb,(656,368)'
class TfPoseVisualizer:
    # the thickness of showing skeleton
    Thickness_ratio = 2

    def __init__(self, graph_path, target_size=(368, 368)):  # 输入参数'graph_opt_mobile.pb,(656,368)'
        self.target_size = target_size
        # load graph #obtain handle
        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef() #建立计算图 可根据索引，获得各个节点信息
            graph_def.ParseFromString(f.read()) #读取的信息直接化成节点信息 209299198 用这些节点建立了一个图

        self.graph = tf.get_default_graph() #获取当前默认图
        tf.import_graph_def(graph_def, name='TfPoseEstimator') #将定义的图 导入到当前默认的图中
        self.persistent_sess = tf.Session(graph=self.graph) #执行图

        self.tensor_image = self.graph.get_tensor_by_name('TfPoseEstimator/image:0')
        self.tensor_output = self.graph.get_tensor_by_name('TfPoseEstimator/Openpose/concat_stage7:0')
        self.heatMat = self.pafMat = None

    @staticmethod
    def draw_pose_rgb(npimg, humans, imgcopy=False): #raw_image ,joints
        if imgcopy:
            npimg = np.copy(npimg) #深度copy 而不只是引用
        image_h, image_w = npimg.shape[:2]
        joints, bboxes, xcenter = [], [], []

        # for record and get dataset
        record_joints_norm = []

        for human in humans:
            xs, ys, centers = [], [], {}
            # 将所有关节点绘制到图像上
            for i in range(CocoPart.Background.value): #18
                if i not in human.body_parts.keys():

                    # 对于缺失的数据，补0
                    record_joints_norm += [0.0, 0.0]
                    continue

                #分数和位置
                body_part = human.body_parts[i]
                center_x = body_part.x * image_w + 0.5
                center_y = body_part.y * image_h + 0.5
                center = (int(center_x), int(center_y))
                centers[i] = center

                record_joints_norm += [round(center_x/1280, 2), round(center_y/720, 2)]

                xs.append(center[0])
                ys.append(center[1])
                # 绘制关节点
                cv.circle(npimg, center, 3, CocoColors[i], thickness=TfPoseVisualizer.Thickness_ratio * 2,
                          lineType=8, shift=0)
            # 将属于同一人的关节点按照各个部位相连
            for pair_order, pair in enumerate(CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue
                cv.line(npimg, centers[pair[0]], centers[pair[1]], CocoColors[pair_order],
                        thickness=TfPoseVisualizer.Thickness_ratio, lineType=8, shift=0)

            # 根据每个人的关节点信息生成ROI区域
            tl_x = min(xs)
            tl_y = min(ys)
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            bboxes.append([tl_x, tl_y, width, height])

            # 记录每一帧的所有关节点
            joints.append(centers) #后检测出来的关节点被append后面去了

            # 记录coco的1号点作为xcenter
            if 1 in centers:
                xcenter.append(centers[1][0])
        return npimg, joints, bboxes, xcenter, record_joints_norm

    @staticmethod
    def draw_pose_only(npimg, humans): #只绘制关节点函数
        image_h, image_w = npimg.shape[:2]
        back_ground = np.ones((image_h, image_w), dtype=np.uint8)
        back_ground = cv.cvtColor(back_ground, cv.COLOR_GRAY2BGR)
        back_ground[:, :, :] = 0  # Black background
        result = TfPoseVisualizer.draw_pose_rgb(back_ground, humans)
        return result

    def inference(self, npimg): #bgr 图片 opencv is honesty
        if npimg is None:
            raise Exception('The frame does not exist.')

        rois = []
        infos = []
        # _get_scaled_img
        if npimg.shape[:2] != (self.target_size[1], self.target_size[0]):
            # resize
            npimg = cv.resize(npimg, self.target_size)
            rois.extend([npimg])
            infos.extend([(0.0, 0.0, 1.0, 1.0)])

        #the image into graph calculate to operate ,the OpenPose models
        #'tuple' (1,46,82,57)
        output = self.persistent_sess.run(self.tensor_output, feed_dict={self.tensor_image: rois})

        heat_mats = output[:, :, :, :19]
        paf_mats = output[:, :, :, 19:]

        output_h, output_w = output.shape[1:3]
        max_ratio_w = max_ratio_h = 10000.0
        for info in infos:
            max_ratio_w = min(max_ratio_w, info[2]) #(1000.0,1.0)
            max_ratio_h = min(max_ratio_h, info[3])
        mat_w, mat_h = int(output_w / max_ratio_w), int(output_h / max_ratio_h) #normalization?

        resized_heat_mat = np.zeros((mat_h, mat_w, 19), dtype=np.float32)
        resized_paf_mat = np.zeros((mat_h, mat_w, 38), dtype=np.float32)
        resized_cnt_mat = np.zeros((mat_h, mat_w, 1), dtype=np.float32)
        resized_cnt_mat += 1e-12
        #这里的PAF是指哪个技术吗 openPose里面提出的匹配问题
        for heatMat, pafMat, info in zip(heat_mats, paf_mats, infos):
            w, h = int(info[2] * mat_w), int(info[3] * mat_h)
            heatMat = cv.resize(heatMat, (w, h))
            pafMat = cv.resize(pafMat, (w, h))
            x, y = int(info[0] * mat_w), int(info[1] * mat_h)
            # add up
            resized_heat_mat[max(0, y):y + h, max(0, x):x + w, :] = np.maximum(
                resized_heat_mat[max(0, y):y + h, max(0, x):x + w, :], heatMat[max(0, -y):, max(0, -x):, :])
            resized_paf_mat[max(0, y):y + h, max(0, x):x + w, :] += pafMat[max(0, -y):, max(0, -x):, :]
            resized_cnt_mat[max(0, y):y + h, max(0, x):x + w, :] += 1

        self.heatMat = resized_heat_mat
        self.pafMat = resized_paf_mat / (np.log(resized_cnt_mat) + 1)

        humans = estimate(self.heatMat, self.pafMat) #返回的是热图 和pafMat的矩阵
        return humans
