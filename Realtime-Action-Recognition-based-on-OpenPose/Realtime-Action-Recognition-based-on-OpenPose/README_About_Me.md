# Train 
1.数据的形式
- 数据都是单人的动作标签？非连续性的吗 应该是连续的
- 将多个人的数值 

2.数据如何获取
    连续帧的图片骨骼点作为特征 然后训练跌倒

3.类别数量

## 模型
最好的效果目前是训练好的framewise_recognition_FallVideo.h5的模型了
这个模型只有正常和跌倒两种模式 也可以训练多个模式的样本 只要样本有就行了
8-14:准备添加一些电梯上的模型参与训练这样比较好

------
## Introduction
*The **pipline** of this work is:*   
 1.程序的设计

------
## Dependencies
 - python >= 3.5
 - Opencv >= 3.4.1   
 - sklearn
 - tensorflow & keras
 - numpy & scipy 
 - pathlib
 
 
------
## Usage
 - Download the openpose VGG tf-model with command line `./download.sh`(/Pose/graph_models/VGG_origin) or fork [here](https://pan.baidu.com/s/1XT8pHtNP1FQs3BPHgD5f-A#list/path=%2Fsharelink1864347102-902260820936546%2Fopenpose%2Fopenpose%20graph%20model%20coco&parentPath=%2Fsharelink1864347102-902260820936546), and place it under the corresponding folder; 
 - `python main.py`, it will **start the webcam**. 
 (you can choose to test video with command `python main.py --video=test.mp4`, however I just tested the webcam mode)   
 - By the way, you can choose different openpose pretrained model in script.    
 **VGG_origin**: training with the VGG net, as same as the CMU providing caffemodel, more accurate but slower, **mobilenet_thin**:  training with the Mobilenet, much smaller than the origin VGG, faster but less accurate.   
 **However, Please attention that the Action Dataset in this repo is collected along with the** ***VGG model*** **running**.


------
## Training with own dataset
 - prepare data(actions) by running `main.py`, remember to ***uncomment the code of data collecting***, the origin data will be saved as a `.txt`.
 - transforming the `.txt` to `.csv`, you can use EXCEL to do this.
 - do the training with the `traing.py` in `Action/training/`, remember to ***change the action_enum and output-layer of model***.
 
 
------
## Test result
 - ***actions detection***
<p align="center">
    <img src="https://github.com/LZQthePlane/Online-Realtime-Action-Recognition-based-on-OpenPose/blob/master/test_out/webcam_test_out.gif", width="540">
 
 - ***work surveilence***
<p align="center">
    <img src="https://github.com/LZQthePlane/Online-Realtime-Action-Recognition-based-on-OpenPose/blob/master/test_out/webcam_under_scene-1.gif", width="540">
<p align="center">
    <img src="https://github.com/LZQthePlane/Online-Realtime-Action-Recognition-based-on-OpenPose/blob/master/test_out/webcam_under_scene-2.gif", width="540">
 
  - ***multi people***
 <p align="center">
    <img src="https://github.com/LZQthePlane/Online-Realtime-Action-Recognition-based-on-OpenPose/blob/master/test_out/webcam_multi-people.gif", width="540">
 

-------
## Note
 - Action recognition in this work is framewise based, so it's technically "**Pose recognition**" to be exactly;   
 - Action is actually a dynamic motion which consists of sequential static poses, therefore classifying framewisely is not a good solution.
 - Considering of using ***RNN(LSTM) model*** to classify actions with dynamic sequential joints data is the next step to improve this project.


------
## Acknowledge
Thanks to the following awesome works:    
 - [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation)   
 - [deep_sort_yolov3](https://github.com/Qidian213/deep_sort_yolov3)    
 - [Real-Time-Action-Recognition](https://github.com/TianzhongSong/Real-Time-Action-Recognition)

##程序

- 使用openPose检测骨骼点，然后将绘制好骨骼点的图片再送deep_sort进行跟踪