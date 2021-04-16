
import numpy as np
import keras
from keras.layers import LSTM
from keras.layers import Dense, Activation,Flatten,Lambda,Permute,merge,Reshape,Conv2D
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as ktf
import os
import struct
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from tgan import gclstm
import tensorflow as tf
import matplotlib.pyplot as plt
from decoders import decoder
from getdata import fetdata
import datetime,csv
import math
from sklearn import metrics



W=np.load("wij.npy")#邻接矩阵
sqe=np.load("squence.npy")#用户时序特征

xall=[]
yall=[]



timesnap=3#每三个时序预测下一个
for i in range(28):
    xall.append(sqe[i:i+timesnap])
    yall.append(sqe[i+timesnap])

xall=np.array(xall)
yall=np.array(yall)


yall=np.int64(yall>0)

id=np.arange(0,xall.shape[0])
np.random.shuffle(id)
idx=id[0:math.ceil(0.7*id.shape[0])]#划分训练集
idt=id[math.ceil(0.7*id.shape[0]):]
print(idx)
print(idt)




#张量操作函数
def spi(x):
    newx=[]
    for i in range(x.shape[-2]):
        newx.append(x[:,i,:])
    return newx
def chen(x):
    return x[0]*x[1]

def sqz(x):
    return K.squeeze(x,axis=0)

def avg(x):
    return K.mean(x,axis=1)

def trs(x):
    return tf.transpose(x,perm=[0,2,3,1])

def resh(x):
    return K.reshape(x,[1,n_step,nodes,n_hidden])

def reshy(x):
    return K.reshape(x,[1,n_input,n_input])
def deco(x):
    x=K.squeeze(x,axis=0)
    z=K.dot(x,K.transpose(x))
    return K.stack([z])



#minist用于可视化实验
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
        print(labels.shape)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))

        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

learning_rate = 0.001
training_iters = 20

nodes=xall.shape[-2]
n_input = xall.shape[-1]
n_step = xall.shape[1]
n_hidden = 128
#n_classes = 10
batch=1






#以下为自定义loss
gamma=2.
alpha=.25
def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def myloss(y_true, y_pred):
    W1=y_true*100
    W0=(K.ones_like(y_true)-y_true)*0.1
    Wi=W1+W0
    return  K.mean(K.square(y_pred-y_true)*Wi)

#监测指标定义
def precisionP(y_true,y_pred):
    #print(y_true.shape)
    #y_pred=np.abs(y_pred-0.37)
    TP = np.sum((y_true * np.round( y_pred)))
    FP = np.sum((1 - y_true) * np.round(y_pred))
    TN = np.sum((1 - y_true) * (1 - np.round(y_pred)))
    FN = np.sum(y_true * (1 - np.round(y_pred)))
    precisionT=TP/(TP+FP+1e-12)
    recallT = TP / (TP + FN + 1e-12)
    precisionN = TN / (TN + FN + 1e-12)
    recallN = TN / (TN + FP + 1e-12)
    acc = (TP + TN) / (TP + TN + FP + FN)
    F1scoreT=2*precisionT*recallT/(precisionT+recallT+1e-12)
    F1scoreN=2*precisionN*recallN/(precisionN+recallN+1e-12)
    ax=np.reshape(y_true,[nodes*n_input,])
    bx=np.reshape(y_pred,[nodes*n_input,])
    try:
        auc=metrics.roc_auc_score(ax, bx)
    except ValueError:
        auc=-1
    return precisionT,recallT,auc

def auc(y_true, y_pred):
    ax = np.reshape(y_true, [nodes*n_input,])
    bx = np.reshape(y_pred, [nodes*n_input,])
    try:
        auc = metrics.roc_auc_score(ax, bx)
    except ValueError:
        auc = -1
    return auc

#模型搭建
ain=Input(batch_shape=[batch,nodes,nodes])
bin=Input(batch_shape=[batch,n_step,nodes,n_input])
x=gclstm(output_dim=n_hidden,timestep=n_step,kernel_regularizer=keras.regularizers.l2(0.001))([ain,bin])
x1=Lambda(trs)(x)
x1=Lambda(sqz)(x1)
x1=Dense(n_step,activation='softmax',kernel_regularizer=keras.regularizers.l2(0.001))(x1)#第一重注意力层
x1=Lambda(resh)(x1)
y=Lambda(chen)([x,x1])
#y=Lambda(avg)(y)
y=Conv2D(1,[1,1],data_format='channels_first',activation='relu')(y)
y=Reshape([nodes,n_hidden])(y)
y=decoder(output_dim=n_input,kernel_regularizer=keras.regularizers.l2(0.001))(y)



#y=Dense(n_classes,activation='softmax')(x1)
model=Model(inputs=[ain,bin],outputs=y)
adam = Adam(lr=learning_rate)
model.compile(optimizer=adam,loss=myloss,)
model.summary()

W=np.reshape(W,[1,nodes,nodes])

#xtr,ytr,z=fetdata(5,10)
best=0.9
#开始训练
for epoch in range(500):
    auclisttrain=[]
    precision=[]
    recall=[]
    losstrain=[]
    for i in range(len(idx)):
        xtr=xall[idx[i]]
        xtr=np.reshape(xtr,[1,timesnap,nodes,n_input])
        ytr=yall[idx[i]]

        ytr = np.reshape(ytr, [1,  nodes, n_input])
        loss=model.train_on_batch([W,xtr],ytr)
        ypre=model.predict([W,xtr])
        print(np.sum(np.round(ypre)))
        losstrain.append(loss)
        print("train loss",loss)
        r=precisionP(ytr,ypre)
        precision.append(r[0])
        recall.append(r[1])
        auclisttrain.append(r[-1])
        print("epoch ",epoch,"batch ",i,"auc",r[-1],'pre:',r[0],'recal:',r[1])
    print("avrerage train loss",sum(losstrain)/len(losstrain),'auc avg:',sum(auclisttrain)/len(auclisttrain),
          'precision:',sum(precision)/len(precision),'recall:',sum(recall)/len(recall),)
    auctest=[]
    if epoch%5==0:
        for j in range(len(idt)):
            xte=xall[idt[j]]
            xte=np.reshape(xte,[1,timesnap,nodes,n_input])
            yte=yall[idt[j]]
            yte = np.reshape(yte, [1,  nodes, n_input])
            ypre = model.predict([W, xte])
            auctest.append(auc(yte, ypre))
    if len(auctest)>0:
        avgauc=sum(auctest)/len(auctest)

    print("epoch ", epoch, "test avg auc", avgauc)
    if avgauc > best:
        best=avgauc
        model.save_weights("agclstmbest.h5")


