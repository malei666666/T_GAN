import numpy as np
import keras
from keras.layers import LSTM, SimpleRNN, ConvLSTM2D, Conv2D,MaxPooling2D
from keras.layers import Dense, Activation,Flatten,Lambda,Permute,merge,Reshape

from keras.optimizers import Adam
from gcns import gcn
from keras import backend as K
from keras.layers import Input
from keras.models import Model,load_model
from mylstmori import gclstm
import tensorflow as tf
import matplotlib.pyplot as plt

model=load_model('Ballroom_CNN.h5')

exit(-1)


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
    return K.reshape(x,[1,n_step,n_input,n_hidden])

def reshy(x):
    return K.reshape(x,[1,n_input,n_input])
def deco(x):
    x=K.squeeze(x,axis=0)
    z=K.dot(x,K.transpose(x))
    return K.stack([z])


learning_rate = 0.001
training_iters = 20
batch_size = 128
display_step = 10

n_input = 28
n_step = 6
n_hidden = 64
n_classes = 10
batch=1

xtr=np.load('xtr.npy')
print(xtr.shape)

ytr=np.load('ytr.npy')
xtest=np.load('xtest.npy')
print(xtest.shape)


#y_train = keras.utils.to_categorical(y_train, n_classes)


bin=Input(batch_shape=[batch,n_step,n_input,n_input])
x=gclstm(output_dim=n_hidden,timestep=n_step,)(bin)
x1=Lambda(trs)(x)
x1=Lambda(sqz)(x1)
x1=Dense(n_step,activation='softmax',name='att')(x1)
x1=Lambda(resh)(x1)
y=Lambda(chen)([x,x1])

y=Conv2D(1,[1,1],data_format='channels_first',activation='linear',name='conv')(y)
y=Reshape([n_input,n_hidden])(y)
y=gcn(output_dim=n_input,)(y)

model=Model(inputs=bin,outputs=y)
adam = Adam(lr=learning_rate)
model.compile(optimizer=adam,loss='mse')
model.summary()

model.fit(xtr,ytr,validation_split=0.25,batch_size=batch,epochs=5,verbose=1)

ypre=model.predict(xtest,batch_size=1)



#output the image

for i in range(6):
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        fname = 'res\\'+'TAN2' + str(i)  + '.png'
        plt.imshow(ypre[i,:,:] * 255, cmap='gray')
        plt.savefig(fname, dpi=96, pad_inches=0.0,bbox_inches = 'tight')
        plt.show()



