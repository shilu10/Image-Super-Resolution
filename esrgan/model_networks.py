from tensorflow import keras 
import tensorflow as tf 
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Dense, Input, UpSampling2D, Conv2DTranspose, Conv2D, add, Add,\
                    Lambda, Concatenate, AveragePooling2D, BatchNormalization, GlobalAveragePooling2D, \
                    Add, LayerNormalization, Activation, LeakyReLU, Lambda, Flatten

                    

class ResidualDenseBlock(keras.Model): 
    def __init__(self, filters):
        super(ResidualDenseBlock, self).__init__()
        # conv blocks
        self.conv_1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')
        self.conv_2 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')
        self.conv_3 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')
        self.conv_4 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')
        self.conv_5 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')

        # Activation (Leaky Relu)
        self.act_1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.act_2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.act_3 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.act_4 = tf.keras.layers.LeakyReLU(alpha=0.2)
        
        # concatenation layer
        self.concat = tf.keras.layers.Concatenate()
        
    def call(self, x):
        x1 = self.act_1(self.conv_1(x))
        x2 = self.act_2(self.conv_2(layers.concatenate([x, x1])))
        x3 = self.act_3(self.conv_3(layers.concatenate([x, x1, x2])))
        x4 = self.act_4(self.conv_4(layers.concatenate([x, x1, x2, x3])))
        x5 = self.conv_5(layers.concatenate([x, x1, x2, x3, x4]))
        
        return x5 * 0.2 + x


class ResidualResidualDenseBlock(keras.Model):
    def __init__(self, filters):
        super(ResidualResidualDenseBlock, self).__init__()
        self.res_1 = ResidualDenseBlock(filters)
        self.res_2 = ResidualDenseBlock(filters)
        self.res_3 = ResidualDenseBlock(filters)
        
    def call(self, inputs): 
        x = self.res_1(inputs)
        x = self.res_2(x)
        x = self.res_3(x)
        
        return x + inputs * 0.2


