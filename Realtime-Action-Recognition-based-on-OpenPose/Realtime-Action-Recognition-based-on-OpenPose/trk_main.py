# -*- coding: UTF-8 -*-
import cv2 as cv
import argparse
import numpy as np
import time
from utils import choose_run_mode, load_pretrain_model, set_video_writer
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize
parser = argparse.ArgumentParser(description='Action Recognition by OpenPose')
parser.add_argument('--video', default='Escalator/Alpha.avi',help='Path to video file.')
args = parser.parse_args()

# 导入相关模型 建立图
# estimator = load_pretrain_model('VGG_origin') #返回一个估计的模型
estimator = load_pretrain_model('mobilenet_thin')  #返回一个类的句柄TfPoseVisualizer 并且建立了计算图
# action_classifier = load_action_premodel('Action/Es_all_demo.h5') #返回动作分类模型 且里面定义了tracker
action_classifier = load_action_premodel('Action/framewise_recognition_bobei.h5') #返回动作分类模型 且里面定义了tracker

# 参数初始化
realtime_fps = '0.0000'
start_time = time.time()
fps_interval = 1
fps_count = 0
run_timer = 0
frame_count = 0

#读写视频文件（仅测试过webcam输入）
cap = choose_run_mode(args) #选择摄像头或者是本地文件
video_writer = set_video_writer(cap, write_fps=int(12)) #保存到本地的视频用到的参数初始化
video_1 = cv.VideoWriter('test_out/alpha.mp4',
                          cv.VideoWriter_fourcc(*'mp4v'),
                          int(12),
                          (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
# # 保存关节数据的txt文件，用于训练过程(for training)
# f = open('origin_data.txt', 'a+') #通过openPose提取keyPoint 然后标记keyPoint进行分类训练 最后通过openpose送到分类

while True and cap.isOpened(): #loop
    has_frame, show = cap.read() #framewise
    if has_frame:
        fps_count += 1
        frame_count += 1
        #crop image ,then ,the image into network
        #todo 缩小检测范围 显然影响到了检测精度 想要改变显示显然需要从节点生成后去做了
        # pose estimation body_parts PartPair uidx_list
        humans = estimator.inference(show) #返回heatMat pafMat空间地址
        # get pose info
        # return frame, joints, bboxes, xcenter
        pose = TfPoseVisualizer.draw_pose_rgb(show, humans) #pose绘制在画面上了
        #save picture  pose[0] 绘制好了关节点的pose图像
        # cv.imwrite('test_out/'+str(frame_count)+'.jpg',pose[0])

        # recognize the action framewise
        show = framewise_recognize(pose, action_classifier) #返回绘制好的frame
        #end Pose,Track,Detection.
        height, width = show.shape[:2]
        # 显示实时FPS值
        if (time.time() - start_time) > fps_interval:
            # 计算这个interval过程中的帧数，若interval为1秒，则为FPS
            realtime_fps = fps_count / (time.time() - start_time)
            fps_count = 0  # 帧数清零
            start_time = time.time()
        fps_label = 'FPS:{0:.2f}'.format(realtime_fps)
        cv.putText(show, fps_label, (width-160, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示检测到的人数
        num_label = "Human: {0}".format(len(humans))
        cv.putText(show, num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示目前的运行时长及总帧数
        if frame_count == 1:
            run_timer = time.time()
        run_time = time.time() - run_timer
        time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
        cv.putText(show, time_frame_label, (5, height-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv.imshow('Action Recognition based on OpenPose', show)
        # video_writer.write(show)
        video_1.write(show)
        cv.waitKey(1)

        # # 采集数据，用于训练过程(for training)
        # joints_norm_per_frame = np.array(pose[-1]).astype(np.str)
        # f.write(' '.join(joints_norm_per_frame))
        # f.write('\n')
    else:
        break
video_writer.release()
cap.release()
# f.close()
