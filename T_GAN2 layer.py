from keras import activations, initializers, constraints
from keras import regularizers


from keras import backend as K
from keras.engine.topology import Layer

class gclstm(Layer):

    def __init__(self, output_dim, timestep, activation1='sigmoid',activation2='tanh',
                 kernel_initializer1='glorot_uniform',kernel_initializer2='zeros',
                 bias_initializer='zeros',
                 use_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.output_dim = output_dim
        self.activation1=activations.get(activation1)
        self.activation2 = activations.get(activation2)
        self.use_bias = use_bias
        self.times=timestep
        self.kernel_initializer1 = initializers.get(kernel_initializer1)
        self.kernel_initializer2 = initializers.get(kernel_initializer2)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        super(gclstm, self).__init__(**kwargs)

    def build(self, batch_input_shape):
        #assert isinstance(input_shape, list)
       
        self.bshape=batch_input_shape
        self.kernelix = self.add_weight(name='kernelix',
                                      shape=(batch_input_shape[-1],self.output_dim),
                                      initializer=self.kernel_initializer1,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)

        self.kernelim = self.add_weight(name='kernelim',
                                      shape=(self.output_dim,self.output_dim),
                                      initializer=self.kernel_initializer1,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)

        self.kernelfx = self.add_weight(name='kernelfx',
                                      shape=(batch_input_shape[-1],self.output_dim),
                                      initializer=self.kernel_initializer1,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)

        self.kernelfm = self.add_weight(name='kernelfm',
                                      shape=(self.output_dim,self.output_dim),
                                      initializer=self.kernel_initializer1,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)
        self.kernelcx = self.add_weight(name='kernelcx',
                                      shape=(batch_input_shape[-1],self.output_dim),
                                      initializer=self.kernel_initializer1,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)
        self.kernelcm = self.add_weight(name='kernelcm',
                                      shape=(self.output_dim,self.output_dim),
                                      initializer=self.kernel_initializer1,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)
        self.kernelox = self.add_weight(name='kernelox',
                                      shape=(batch_input_shape[-1],self.output_dim),
                                      initializer=self.kernel_initializer1,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)
        self.kernelom = self.add_weight(name='kernelom',
                                      shape=(self.output_dim,self.output_dim),
                                      initializer=self.kernel_initializer1,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)

        self.savestate = self.add_weight(name='state',
                                       shape=(batch_input_shape[0],batch_input_shape[-2],self.output_dim),
                                       initializer=self.kernel_initializer2,
                                       trainable=False)

        self.ctpre = self.add_weight(name='ctp',
                                       shape=(batch_input_shape[0],batch_input_shape[-2],self.output_dim),
                                       initializer=self.kernel_initializer2,
                                       trainable=False)



        if self.use_bias:
            self.biasi = self.add_weight(shape=(1,self.output_dim),
                                        initializer=self.bias_initializer,
                                        name='biasi',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.biasf = self.add_weight(shape=(1, self.output_dim),
                                         initializer=self.bias_initializer,
                                         name='biasf',
                                         regularizer=self.bias_regularizer,
                                         constraint=self.bias_constraint)
            self.biasc = self.add_weight(shape=(1, self.output_dim),
                                         initializer=self.bias_initializer,
                                         name='biasc',
                                         regularizer=self.bias_regularizer,
                                         constraint=self.bias_constraint)
            self.biaso = self.add_weight(shape=(1, self.output_dim),
                                         initializer=self.bias_initializer,
                                         name='biaso',
                                         regularizer=self.bias_regularizer,
                                         constraint=self.bias_constraint)

        super(gclstm, self).build(batch_input_shape)  # 

    def call(self, x):
            #print('self-shape:', self.kernel.shape)
            #print('t shape::',t.shape)
            savest=[]
            self.savestate = K.zeros([self.bshape[0],self.bshape[-2], self.output_dim])
            self.ctpre= K.zeros([self.bshape[0],self.bshape[-2], self.output_dim])
            for i in range(self.times):
                inputgate=self.activation1(K.dot(x[:,i,:,:],self.kernelix)+K.dot(self.savestate,self.kernelim)+self.biasi)
                forgetgate = self.activation1(
                    K.dot(x[:, i,:,:], self.kernelfx) + K.dot(self.savestate, self.kernelfm) + self.biasf)
                outputgate = self.activation1(
                    K.dot(x[:, i, :,:], self.kernelox) + K.dot(self.savestate, self.kernelom) + self.biaso)
                ct = forgetgate*self.ctpre+inputgate*(self.activation2(K.dot(x[:, i, :,:], self.kernelcx) + K.dot(self.savestate, self.kernelcm) + self.biasc))

                state=outputgate*self.activation2(ct)

                self.savestate=state
                self.ctpre=ct
                savest.append(state)





            return K.reshape(K.stack(savest),[self.bshape[0],self.times,self.bshape[-2], self.output_dim])

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.times,input_shape[-2],self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
        }

        base_config = super(gclstm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))