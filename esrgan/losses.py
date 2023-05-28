import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import * 
from tensorflow.keras import losses 


class AdversarialLoss(keras.losses.Losses):
    """
        Generator loss, which takes the hr_ouput and sr-output from Discriminator, uses
        the Binary crossentropy
    """
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.cross_entropy = losses.BinaryCrossentropy(from_logits=True)

    def call(self, sr_output): 
        real_labels = tf.ones_like(sr_output)
        loss_val = self.cross_entropy(real_labels, sr_output)

        return loss_val


class DiscriminatorLoss(keras.losses.Losses):
    """
        this takes hr_output and sr_output from Discriminator, uses the Binary crossentropy
    """
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.cross_entropy = losses.BinaryCrossentropy(from_logits=True)

    def call(self, hr_output, sr_output): 
        real_labels = tf.ones_like(hr_output)
        fake_labels = tf.zeros_like(sr_output)

        hr_output_loss = self.cross_entropy(real_labels, hr_output)
        sr_output_loss = self.cross_entropy(fake_labels, sr_output)

        total_loss = 0.5 * hr_output_loss + sr_output_loss

        return total_loss


class PixelLoss(keras.losses.Losses):
     """
        this function used to calculate the pixel loss, takes the sr_img and hr_img,
        uses either l1 or l2 loss.
    """
    def __init__(self, criterion):
        super(PixelLoss, self).__init__()
        self.criterion = criterion
        
        if self.criterion == "l1":
            self.loss_object = keras.losses.MeanAbsoluteError()

        if self.criterion == "l2":
            self.loss_object = keras.losses.MeanSquaredError()

        else: 
            raise NotImplementedError('Loss type {} is not recognized.'.format(criterion)) 

    def call(self, hr_img, sr_img): 
        loss_val = self.loss_object(hr_img, sr_img)

        return loss_val


class ContentLoss(keras.losses.Losses):
     """
        this function used to calculate the content loss, which is same as pixel loss
        instead of taking img, this takes feature map as input, uses either l1 or l2 loss.
    """
    def __init__(self, criterion):
        super(PixelLoss, self).__init__()
        self.criterion = criterion
        
        if self.criterion == "l1":
            self.loss_object = keras.losses.MeanAbsoluteError()

        if self.criterion == "l2":
            self.loss_object = keras.losses.MeanSquaredError()

        else: 
            raise NotImplementedError('Loss type {} is not recognized.'.format(criterion)) 

    def call(self, hr_fmap, sr_fmap): 
        loss_val = self.loss_object(hr_fmap, sr_fmap)

        return loss_val


class DiscriminatorRaGanLoss(keras.losses.Losses):
    """
        this class used to claculate the discriminator realavistic gan loss, which
        is described in the esrgan paper.
    """
    def __init__(self):
        super(DiscriminatorRaGanLoss, self).__init__()
        self.cross_entropy = losses.BinaryCrossentropy(from_logits=True)
        self.sigmoid = tf.sigmoid 

    def call(self, hr_output, sr_output):
        real_labels = tf.ones_like(sr_output)
        fake_labels = tf.zeros_like(hr_output)

        xr = hr_output - tf.reduce_mean(sr_output)
        xr = self.sigmoid(xr)

        xf = sr_output - tf.reduce_mean(hr_output)
        xf = self.sigmoid(xr)

        xr = self.cross_entropy(real_labels, xr)
        xf = self.cross_entropy(fake_labels, xf)

        loss_val = 0.5 * (xr + xf)

        return loss_val


class GeneratorRaGanLoss(keras.losses.Losses):
    """
        this class used to claculate the generator realavistic gan loss, which
        is described in the esrgan paper.
    """
    def __init__(self):
        super(DiscriminatorRaGanLoss, self).__init__()
        self.cross_entropy = losses.BinaryCrossentropy(from_logits=True)
        self.sigmoid = tf.sigmoid 

    def call(self, hr_output, sr_output):
        real_labels = tf.ones_like(sr_output)
        fake_labels = tf.zeros_like(hr_output)

        xr = hr_output - tf.reduce_mean(sr_output)
        xr = self.sigmoid(xr)

        xf = sr_output - tf.reduce_mean(hr_output)
        xf = self.sigmoid(xr)

        xr = self.cross_entropy(fake_labels, xr)
        xf = self.cross_entropy(real_labels, xf)

        loss_val = (xr + xf)

        return loss_val