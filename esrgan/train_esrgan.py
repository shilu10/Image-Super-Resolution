import os 
import cv2
import time 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as ans 
from tqdm import tqdm 
import shutil 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import Model
from tensorflow.keras import layers 
from tensorflow.keras.layers import *
import tensorflow_datasets as tfds
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import *
from datetime import datetime
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Dense, Input, UpSampling2D, Conv2DTranspose, Conv2D, add, Add,\
                    Lambda, Concatenate, AveragePooling2D, BatchNormalization, GlobalAveragePooling2D, \
                    Add, LayerNormalization, Activation, LeakyReLU, Lambda, Flatten
try:
    import tensorflow_addons as tfa 
except:
    !pip install tensorflow_addons
    import tensorflow_addons as tfa
    from tensorflow_addons.layers import InstanceNormalization

from data import * 
from models import * 


def train_step(train_batch):
    lr_img, hr_img = train_batch
        
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # generate the super resolution image
        sr = generator(lr_img, training=True)
            
        # discriminator output
        disc_fake_output = discriminator(sr, training=True)
        disc_real_output = discriminator(hr_img, training=True)
            
        # feature maps from vgg.
        fmap_sr = perceptual_model(sr, training=False)
        fmap_hr = perceptual_model(hr_img, training=False)
            
        # loss
        pixel_loss = WEIGHT_PIXEL * compute_pixel_loss(hr_img, sr)
        content_loss = WEIGHT_FEATURE * compute_content_loss(fmap_hr, fmap_sr)
        adversial_loss = WEIGHT_GAN * compute_generator_loss_ragan(disc_real_output, disc_fake_output)
            
        total_disc_loss = compute_discriminator_loss_ragan(disc_real_output, disc_fake_output)
        total_gen_loss = pixel_loss + content_loss + adversial_loss

    ## Params
    disc_params = discriminator.trainable_weights
    gen_params = generator.trainable_weights
        
    # grads
    disc_grads = disc_tape.gradient(total_disc_loss, disc_params)
    gen_grads = gen_tape.gradient(total_gen_loss, gen_params)
        
    # Back Propagate
    generator_optimizer.apply_gradients(zip(gen_grads, gen_params))
    discriminator_optimizer.apply_gradients(zip(disc_grads, disc_params))
        
    # loss tracker
    generator_loss_tracker.update_state(total_gen_loss)
    discriminator_loss_tracker.update_state(total_disc_loss)

    with summary_writer.as_default():
        tf.summary.scalar('gen_loss', total_gen_loss, step=step//1000)
        tf.summary.scalar('gen_gan_loss', gan_loss, step=step//1000)
        tf.summary.scalar('gen_l1_loss', l1_loss, step=step//1000)
        tf.summary.scalar('disc_loss', total_disc_loss, step=step//1000)
        return {
                "gen_loss": generator_loss_tracker.result(), 
                "disc_loss": discriminator_loss_tracker.result(), 
               }    


def define_loss_tracker():
    pass 


def define_optimizer():
    pass 


def define_loss():
    pass 

def test_step():
    pass 

def trainer():
    pass 

def get_psnr(, ds):
    psnr_values = []
    for lr, hr in ds:
        # convert dtype to float32
        lr = tf.cast(lr, tf.float32)
        # get sr image from model
        sr = self.generator(lr)
        # clip values to 0,255
        sr = tf.clip_by_value(sr, 0, 255)
        # round up values
        sr = tf.round(sr)
        # change dtype to unint8
        sr = tf.cast(sr, tf.uint8)
        # get psnr value
        psnr_value = tf.image.psnr(hr, sr, max_val=255)[0]
        # append psnr value
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)