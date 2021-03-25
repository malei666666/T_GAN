from keras import activations, initializers, constraints
from keras import regularizers


from keras import backend as K
from keras.engine.topology import Layer

class gcn(Layer):

    def __init__(self, output_dim, activation1='tanh',
                 kernel_initializer1='glorot_uniform',
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

        self.use_bias = use_bias

        self.kernel_initializer1 = initializers.get(kernel_initializer1)

        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        super(gcn, self).__init__(**kwargs)

    def build(self, batch_input_shape):
        #assert isinstance(input_shape, list)
       
        self.bshape=batch_input_shape
        self.kernelg = self.add_weight(name='kernelg',
                                      shape=(batch_input_shape[-1],self.output_dim),
                                      initializer=self.kernel_initializer1,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)




        if self.use_bias:
            self.biasi = self.add_weight(shape=(1,self.output_dim),
                                        initializer=self.bias_initializer,
                                        name='biasi',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)


        super(gcn, self).build(batch_input_shape) 

    def call(self, x):

            return self.activation1(K.dot(x,self.kernelg)+self.biasi)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-2],self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
        }

        base_config = super(gclstm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))