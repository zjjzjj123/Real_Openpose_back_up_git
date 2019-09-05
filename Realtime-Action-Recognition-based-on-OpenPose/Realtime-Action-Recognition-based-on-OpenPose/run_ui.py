# -*- coding: UTF-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import numpy as np
import time
import settings

###另一个主函数里的
# -*- coding: UTF-8 -*-
import cv2 as cv
import argparse
import numpy as np
import time
from utils import choose_run_mode, load_pretrain_model, set_video_writer
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize
parser = argparse.ArgumentParser(description='Action Recognition by OpenPose')
parser.add_argument('--video', default='Escalator/xa_0051.mp4',help='Path to video file.')
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

# #读写视频文件（仅测试过webcam输入）
# cap = choose_run_mode(args) #选择摄像头或者是本地文件
# video_writer = set_video_writer(cap, write_fps=int(12)) #保存到本地的视频用到的参数初始化
# video_1 = cv.VideoWriter('test_out/xn_0007.mp4',
#                           cv.VideoWriter_fourcc(*'mp4v'),
#                           int(12),
#                           (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
#
# while True and cap.isOpened(): #loop
#     has_frame, show = cap.read() #framewise
#     if has_frame:
#         fps_count += 1
#         frame_count += 1
#         #crop image ,then ,the image into network
#         #todo 缩小检测范围 显然影响到了检测精度 想要改变显示显然需要从节点生成后去做了
#         # show = show[ymin:ymax,xmin:xmax]
#         # Ano_list = draw_region_detection(show) #绘制检测区域 并返回范围线
#         # pose estimation body_parts PartPair uidx_list
#         humans = estimator.inference(show) #返回heatMat pafMat空间地址
#         # get pose info
#         # return frame, joints, bboxes, xcenter
#         pose = TfPoseVisualizer.draw_pose_rgb(show, humans) #pose绘制在画面上了
#         # recognize the action framewise
#         show = framewise_recognize(pose, action_classifier) #返回绘制好的frame
#         #end Pose,Track,Detection.
#
#         height, width = show.shape[:2]
#         # 显示实时FPS值
#         if (time.time() - start_time) > fps_interval:
#             # 计算这个interval过程中的帧数，若interval为1秒，则为FPS
#             realtime_fps = fps_count / (time.time() - start_time)
#             fps_count = 0  # 帧数清零
#             start_time = time.time()
#         fps_label = 'FPS:{0:.2f}'.format(realtime_fps)
#         cv.putText(show, fps_label, (width-160, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
#         # 显示检测到的人数
#         num_label = "Human: {0}".format(len(humans))
#         cv.putText(show, num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
#         # 显示目前的运行时长及总帧数
#         if frame_count == 1:
#             run_timer = time.time()
#         run_time = time.time() - run_timer
#         time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
#         cv.putText(show, time_frame_label, (5, height-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
#         cv.imshow('Action Recognition based on OpenPose', show)
#         video_writer.write(show)
#         video_1.write(show)
#         cv.waitKey(1)
#
#         # # 采集数据，用于训练过程(for training)
#         # joints_norm_per_frame = np.array(pose[-1]).astype(np.str)
#         # f.write(' '.join(joints_norm_per_frame))
#         # f.write('\n')
#     else:
#         break
# video_writer.release()
# cap.release()
# f.close()
#主函数里面的

class Ui_MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        # self.tracker = Sort(settings.sort_max_age, settings.sort_min_hit)
        self.timer_camera = QtCore.QTimer()
        self.cap = cv.VideoCapture()
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_mode = 0
        self.fps = 0.00
        self.data = {}
        self.memory = {}
        self.joints = []
        self.current = []
        self.previous = []

    def set_ui(self):

        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()

        self.button_open_camera = QtWidgets.QPushButton(u'相机 OFF')

        self.button_mode_1 = QtWidgets.QPushButton(u'姿态估计 OFF')
        self.button_mode_2 = QtWidgets.QPushButton(u'多人跟踪 OFF')
        self.button_mode_3 = QtWidgets.QPushButton(u'行为识别 OFF')

        self.button_close = QtWidgets.QPushButton(u'退出')

        self.button_open_camera.setMinimumHeight(50)
        self.button_mode_1.setMinimumHeight(50)
        self.button_mode_2.setMinimumHeight(50)
        self.button_mode_3.setMinimumHeight(50)

        self.button_close.setMinimumHeight(50)

        self.button_close.move(10, 100)

        self.infoBox = QtWidgets.QTextBrowser(self)
        self.infoBox.setGeometry(QtCore.QRect(10, 300, 200, 180))

        # 信息显示
        self.label_show_camera = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(200, 200)

        self.label_show_camera.setFixedSize(settings.winWidth + 1, settings.winHeight + 1)
        self.label_show_camera.setAutoFillBackground(True)

        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_mode_1)
        self.__layout_fun_button.addWidget(self.button_mode_2)
        self.__layout_fun_button.addWidget(self.button_mode_3)
        self.__layout_fun_button.addWidget(self.button_close)
        self.__layout_fun_button.addWidget(self.label_move)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.label_show_camera)

        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        self.setWindowTitle(u'实时多人姿态估计与行为识别系统')

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_event)
        self.timer_camera.timeout.connect(self.show_camera) #多久连接一次showcamera

        self.button_mode_1.clicked.connect(self.button_event)
        self.button_mode_2.clicked.connect(self.button_event)
        self.button_mode_3.clicked.connect(self.button_event)
        self.button_close.clicked.connect(self.close)

    def button_event(self):
        sender = self.sender() #判断信号源  也就是那个按钮被按下
        if sender == self.button_mode_1 and self.timer_camera.isActive():
            if self.__flag_mode != 1:
                self.__flag_mode = 1
                self.button_mode_1.setText(u'姿态估计 ON')
                self.button_mode_2.setText(u'多人跟踪 OFF')
                self.button_mode_3.setText(u'行为识别 OFF')
                # self.show_camera()
            else:
                self.__flag_mode = 0
                self.button_mode_1.setText(u'姿态估计 OFF')
                self.infoBox.setText(u'相机已打开')
        elif sender == self.button_mode_2: #and self.timer_camera.isActive():
            if self.__flag_mode != 2:
                self.__flag_mode = 2
                self.button_mode_1.setText(u'姿态估计 OFF')
                self.button_mode_2.setText(u'多人跟踪 ON')
                self.button_mode_3.setText(u'行为识别 OFF')
            else:
                self.__flag_mode = 0
                self.button_mode_2.setText(u'多人跟踪 OFF')
                self.infoBox.setText(u'相机已打开')
        elif sender == self.button_mode_3:# and self.timer_camera.isActive():
            if self.__flag_mode != 3:
                self.__flag_mode = 3
                self.button_mode_1.setText(u'姿态估计 OFF')
                self.button_mode_2.setText(u'多人跟踪 OFF')
                self.button_mode_3.setText(u'行为识别 ON')
            else:
                self.__flag_mode = 0
                self.button_mode_3.setText(u'行为识别 OFF')
                self.infoBox.setText(u'相机已打开')
        else:
            self.__flag_mode = 0
            self.button_mode_1.setText(u'姿态估计 OFF')
            self.button_mode_2.setText(u'多人跟踪 OFF')
            self.button_mode_3.setText(u'行为识别 OFF')
            if self.timer_camera.isActive() == False:
                # flag = self.cap.open(self.CAM_NUM) #打开摄像头
                flag = self.cap.open(args.video) #返回bool变量 打开对应的视频
                self.cap.set(cv.CAP_PROP_FRAME_WIDTH, settings.winWidth)
                self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, settings.winHeight)
                if flag == False:
                    msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                                        buttons=QtWidgets.QMessageBox.Ok,
                                                        defaultButton=QtWidgets.QMessageBox.Ok)
                else:
                    self.timer_camera.start(100)
                    self.button_open_camera.setText(u'相机 ON')
                    self.infoBox.setText(u'相机已打开')
            else:
                self.timer_camera.stop()
                self.cap.release()
                self.label_show_camera.clear()
                self.button_open_camera.setText(u'相机 OFF')
                self.infoBox.setText(u'相机已关闭')

    def show_camera(self):
        start = time.time()
        ret, frame = self.cap.read()

        if ret:
            if self.__flag_mode == 1:
                self.infoBox.setText(u'当前为人体姿态估计模式')
                humans = estimator.inference(frame)  # 返回heatMat pafMat空间地址
                # get pose info
                # return frame, joints, bboxes, xcenter
                pose = TfPoseVisualizer.draw_pose_rgb(frame, humans)  # pose绘制在画面上了
                # recognize the action framewise
                show = framewise_recognize(pose, action_classifier)  # 返回绘制好的frame

            # elif self.__flag_mode == 2:
            #     self.infoBox.setText(u'当前为多人跟踪模式')
            #     humans = poseEstimator.inference(show)
            #     show, joints, bboxes, xcenter, sk = TfPoseEstimator.get_skeleton(show, humans, imgcopy=False)
            #     height = show.shape[0]
            #     width = show.shape[1]
            #     if bboxes:
            #         result = np.array(bboxes)
            #         det = result[:, 0:5]
            #         det[:, 0] = det[:, 0] * width
            #         det[:, 1] = det[:, 1] * height
            #         det[:, 2] = det[:, 2] * width
            #         det[:, 3] = det[:, 3] * height
            #         trackers = self.tracker.update(det)
            #
            #         for d in trackers:
            #             xmin = int(d[0])
            #             ymin = int(d[1])
            #             xmax = int(d[2])
            #             ymax = int(d[3])
            #             label = int(d[4])
            #             cv2.rectangle(show, (xmin, ymin), (xmax, ymax),
            #                           (int(settings.c[label % 32, 0]),
            #                            int(settings.c[label % 32, 1]),
            #                            int(settings.c[label % 32, 2])), 4)
            #
            # elif self.__flag_mode == 3:
            #     self.infoBox.setText(u'当前为人体行为识别模式')
            #     humans = poseEstimator.inference(show)
            #     ori = np.copy(show)
            #     show, joints, bboxes, xcenter, sk= TfPoseEstimator.get_skeleton(show, humans, imgcopy=False)
            #     height = show.shape[0]
            #     width = show.shape[1]
            #     if bboxes:
            #         result = np.array(bboxes)
            #         det = result[:, 0:5]
            #         det[:, 0] = det[:, 0] * width
            #         det[:, 1] = det[:, 1] * height
            #         det[:, 2] = det[:, 2] * width
            #         det[:, 3] = det[:, 3] * height
            #         trackers = self.tracker.update(det)
            #         self.current = [i[-1] for i in trackers]
            #
            #         if len(self.previous) > 0:
            #             for item in self.previous:
            #                 if item not in self.current and item in self.data:
            #                     del self.data[item]
            #                 if item not in self.current and item in self.memory:
            #                     del self.memory[item]
            #
            #         self.previous = self.current
            #         for d in trackers:
            #             xmin = int(d[0])
            #             ymin = int(d[1])
            #             xmax = int(d[2])
            #             ymax = int(d[3])
            #             label = int(d[4])
            #             try:
            #                 j = np.argmin(np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter]))
            #             except:
            #                 j = 0
            #             if joint_filter(joints[j]):
            #                 joints[j] = joint_completion(joint_completion(joints[j]))
            #                 if label not in self.data:
            #                     self.data[label] = [joints[j]]
            #                     self.memory[label] = 0
            #                 else:
            #                     self.data[label].append(joints[j])
            #
            #                 if len(self.data[label]) == settings.L:
            #                     pred = actionPredictor().move_status(self.data[label])
            #                     if pred == 0:
            #                         pred = self.memory[label]
            #                     else:
            #                         self.memory[label] = pred
            #                     self.data[label].pop(0)
            #
            #                     location = self.data[label][-1][1]
            #                     if location[0] <= 30:
            #                         location = (51, location[1])
            #                     if location[1] <= 10:
            #                         location = (location[0], 31)
            #
            #                     cv2.putText(show, settings.move_status[pred], (location[0] - 30, location[1] - 10),
            #                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            #                                 (0, 255, 0), 2)
            #
            #             cv2.rectangle(show, (xmin, ymin), (xmax, ymax),
            #                           (int(settings.c[label % 32, 0]),
            #                            int(settings.c[label % 32, 1]),
            #                            int(settings.c[label % 32, 2])), 4)
            #
            show = cv.resize(frame, (settings.winWidth, settings.winHeight))
            show = cv.cvtColor(show, cv.COLOR_BGR2RGB)
            end = time.time()
            self.fps = 1. / (end - start)
            cv.putText(show, 'FPS: %.2f' % self.fps, (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:
            self.cap.release()
            self.timer_camera.stop()

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cancel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cancel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cancel.setText(u'取消')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()
            print("System exited.")


if __name__ == '__main__':
    # load_model()
    print("Load all models done!")
    print("The system starts ro run.")
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
