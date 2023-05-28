from tensorflow import keras 
import tensorflow as tf 
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Dense, Input, UpSampling2D, Conv2DTranspose, Conv2D, add, Add,\
                    Lambda, Concatenate, AveragePooling2D, BatchNormalization, GlobalAveragePooling2D, \
                    Add, LayerNormalization, Activation, LeakyReLU, Lambda, Flatten
from model_networks import ResidualResidualDenseBlock


class RRDBNet(keras.Model): 
    def __init__(self, num_rrdb, filters, upscale):
        super(RRDBNet, self).__init__()
        self.upscale = upscale
        
        self.conv_1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')
        
        trunk = []
        for _ in range(num_rrdb):
            trunk.append(ResidualResidualDenseBlock(filters))
        self.trunk = tf.keras.Sequential(trunk)
        
        self.conv_2 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')
        
        if upscale == 2:
            self.upsampling1 = tf.keras.Sequential(
               [
                    Conv2D(filters=filters * (upscale ** 2), kernel_size=3, strides=1, padding='same'),
                    tf.keras.layers.LeakyReLU(alpha=0.2),
                    Lambda(self.pixel_shuffle(scale_factor=2))
               ]
            )
            
        if upscale == 4:
            self.upsampling1 = tf.keras.Sequential(
               [
                    Conv2D(filters=filters * (upscale ** 2), kernel_size=3, strides=1, padding='same'),
                    tf.keras.layers.LeakyReLU(alpha=0.2),
                    Lambda(self.pixel_shuffle(scale_factor=2))
               ]
            )
            self.upsampling2 = tf.keras.Sequential(
                [
                    Conv2D(filters=filters * (upscale ** 2), kernel_size=3, strides=1, padding='same'),
                    tf.keras.layers.LeakyReLU(alpha=0.2),
                    Lambda(self.pixel_shuffle(scale_factor=2))
                ]
            )
            
        if upscale == 8:
            self.upsampling1 = tf.keras.Sequential(
                [
                    Conv2D(filters=filters * (upscale ** 2), kernel_size=3, strides=1, padding='same'),
                    tf.keras.layers.LeakyReLU(alpha=0.2),
                    Lambda(self.pixel_shuffle(scale_factor=2))
                ]
            )
            self.upsampling2 = tf.keras.Sequential(
                [
                    Conv2D(filters=filters * (upscale ** 2), kernel_size=3, strides=1, padding='same'),
                    tf.keras.layers.LeakyReLU(alpha=0.2),
                    Lambda(self.pixel_shuffle(scale_factor=2))
                ]
            )
            self.upsampling3 = tf.keras.Sequential(
                [
                    Conv2D(filters=filters * (upscale ** 2), kernel_size=3, strides=1, padding='same'),
                    tf.keras.layers.LeakyReLU(alpha=0.2),
                    Lambda(self.pixel_shuffle(scale_factor=2))
                ]
            )
            self.upsampling4 = tf.keras.Sequential(
                [
                    Conv2D(filters=filters * (upscale ** 2), kernel_size=3, strides=1, padding='same'),
                    tf.keras.layers.LeakyReLU(alpha=0.2),
                    Lambda(self.pixel_shuffle(scale_factor=2))
                ]
            )
            
        self.conv_3 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')
        self.lrelu_3 =  tf.keras.layers.LeakyReLU(alpha=0.2)
        
        self.conv_4 = Conv2D(filters=3, kernel_size=3, strides=1, padding='same')

    
    def call(self, x):
        conv_1 = self.conv_1(x)
        x = self.trunk(conv_1)
        x = self.conv_2(x)
        x = Add()([x, conv_1])

        if self.upscale == 2:
            x = self.upsampling1(x)
            
        if self.upscale == 4:
            x = self.upsampling1(x)
            x = self.upsampling2(x)
            
        if self.upscale == 8:
            x = self.upsampling1(x)
            x = self.upsampling2(x)
            x = self.upsampling3(x)
            #x = self.upsampling4(x)

        x = self.conv_3(x)
        x = self.lrelu_3(x)
        x = self.conv_4(x)

        return x
    
    def upsample(self, x, scale, num_filters):
        def upsample_1(x, factor, **kwargs):
            x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
            return Lambda(pixel_shuffle(scale=factor))(x)

        if scale == 2:
            x = upsample_1(x, 2, name='conv2d_1_scale_2')
        elif scale == 3:
            x = upsample_1(x, 3, name='conv2d_1_scale_3')
        elif scale == 4:
            x = upsample_1(x, 2, name='conv2d_1_scale_2')
            x = upsample_1(x, 2, name='conv2d_2_scale_2')

        return x
    
    def pixel_shuffle(self, scale_factor=2, **kwargs):
        return Lambda(lambda  x: tf.nn.depth_to_space(x, scale_factor), **kwargs)
    
    def summary(self):
        x = Input(shape=(24, 24, 3))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()



class Discriminator(keras.Model):
    def __init__(self, filters=64, strides=1) -> None:
        super(Discriminator, self).__init__()
        
        self.features = tf.keras.Sequential([
            # input size. (3) x 128 x 128
            Conv2D(64, kernel_size=3, strides=strides, padding='same', use_bias=False),
            tf.keras.layers.LeakyReLU(0.2),
            
            # state size. (64) x 64 x 64
            Conv2D(filters, kernel_size=3, strides=strides, padding='same', use_bias=False),
            BatchNormalization(momentum=0.8),
            tf.keras.layers.LeakyReLU(0.2),
            Conv2D(filters, kernel_size=3, strides=strides, padding='same', use_bias=False),
            BatchNormalization(momentum=0.8),
            tf.keras.layers.LeakyReLU(0.2),
            
            # state size. (128) x 32 x 32
            Conv2D(filters*2, kernel_size=3, strides=strides, padding='same', use_bias=False),
            BatchNormalization(momentum=0.8),
            tf.keras.layers.LeakyReLU(0.2),
            Conv2D(filters*2, kernel_size=3, strides=strides, padding='same', use_bias=False),
            BatchNormalization(momentum=0.8),
            tf.keras.layers.LeakyReLU(0.2),
            
            # state size. (256) x 16 x 16
            Conv2D(filters*4, kernel_size=3, strides=strides, padding='same', use_bias=False),
            BatchNormalization(momentum=0.8),
            tf.keras.layers.LeakyReLU(0.2),
            Conv2D(filters*4, kernel_size=3, strides=strides, padding='same', use_bias=False),
            BatchNormalization(momentum=0.8),
            tf.keras.layers.LeakyReLU(0.2),
            
            # state size. (512) x 8 x 8
            Conv2D(filters*8, kernel_size=3, strides=strides, padding='same', use_bias=False),
            BatchNormalization(momentum=0.8),
            tf.keras.layers.LeakyReLU(0.2),
            Conv2D(filters*8, kernel_size=3, strides=strides, padding='same', use_bias=False),
            BatchNormalization(momentum=0.8),
            tf.keras.layers.LeakyReLU(0.2),
            
            # state size. (512) x 4 x 4
            Conv2D(filters*8, kernel_size=3, strides=strides, padding='same', use_bias=False),
            BatchNormalization(momentum=0.8),
            tf.keras.layers.LeakyReLU(0.2),
        ])

        self.classifier = tf.keras.Sequential([
            Dense(filters*16),
            tf.keras.layers.LeakyReLU(0.2),
            Dense(1)
        ])

    def call(self, x):
        out = self.features(x)
        out = self.classifier(out)

        return out
    
    def summary(self):
        x = Input(shape=(96, 96, 3))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()