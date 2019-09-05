import keras
from enum import Enum
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback #回调函数
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#1.load data
#2.struct model
#3.train model
#4.save model and test model

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[],'epoch':[]}
        self.accuracy = {'batch':[],'epoch':[]}
        self.val_loss = {'batch':[],'epoch':[]}
        self.val_acc = {'batch':[],'epoch':[]}
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
    def on_epoch_end(self, epoch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self,loss_type):
        iters = range(len(self.losses[loss_type])) #得到传入的loss类型的数量
        plt.figure()
        #acc
        plt.plot(iters,self.accuracy[loss_type],'r',label='train_acc')
        #loss
        plt.plot(iters,self.losses[loss_type],'g',label='trian_loss')

        if loss_type == 'epoch':
            #val_acc
            plt.plot(iters,self.val_acc[loss_type],'b',label='val_acc')
            plt.plot(iters,self.val_loss[loss_type],'k',label='val_loss')
        plt.grid(True) #显示网格
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc='upper right')
        plt.show()

raw_data = pd.read_csv('Es_all_bobei_Fall.csv',header=0) #得到所有数据和维度
print(raw_data)
dataset = raw_data.values #形成list形式的数据
print(dataset)
_x = dataset[0:2214,0:36].astype(float)
_y = dataset[0:2214,36] #标签
print(_x)
print(_y)
#生成 [0],[1]代表类别数量
encoder_y = [0]*1315 + [1]* 899
dnum_y = np_utils.to_categorical(encoder_y)
print(dnum_y) #
#随机选择训练和测试样本
x_trian,x_test,y_trian,y_test = train_test_split(_x,dnum_y,test_size=0.1,random_state=9)
print(x_trian)

model = Sequential() #
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(16,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(2,activation='softmax'))

#trian
his = LossHistory() #历史记录
model.compile(loss='categorical_crossentropy',optimizer=Adam(0.0001),metrics=['accuracy']) #metrics包含了网络性能指标
model.fit(x_trian,y_trian,batch_size=32,epochs=20,verbose=1,validation_data=(x_test,y_test),callbacks=[his])
model.summary() #隐藏模型
his.loss_plot('batch') #绘制loss曲线

model.save('Es_all_bobei_Fall_my.h5')
