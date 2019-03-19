from __future__ import print_function, division
import scipy

# from keras.datasets import mnist
# from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.utils import multi_gpu_model

from keras import backend as K

import datetime
import matplotlib.pyplot as plt
import sys
from data_loader1 import DataLoader
import numpy as np
import os
import random
import math
import statistics as stat

from tensorflowjs.converters import save_keras_model as exportTfjs

class Pix2Pix():
    def __init__(self, directory, location, size, scale, data_type, channelA, channelB):

        self.GPUs = 2
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1                                                                       #?NP - change to 1 input channel
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = '%s_%s_%s_%s_to_%s' % (location, size, scale, channelA, channelB)
        self.data_loader = DataLoader(directory, location, size, scale, data_type, channelA, channelB)

        self.randomSeed = 1488746598
        self.convInitializer = RandomNormal(0.0, 0.02, self.randomSeed)
        self.bnormInitializer = RandomNormal(1.0, 0.02, self.randomSeed)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        dOptimizer = Adam(0.0002, 0.5)
        gOptimizer = Adam(0.0002, 0.5)

        self.l1_weight = 100.0
        self.cGAN_weight = 1.0

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        if self.GPUs: self.discriminator =  multi_gpu_model(self.discriminator, gpus=self.GPUs)
        
        self.discriminator.compile(loss='binary_crossentropy', optimizer=dOptimizer, metrics=['accuracy'])


        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()
        # Input images and their conditioning images
        # img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        validity = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=img_B, outputs=[validity, fake_A])
        self.combined.summary()
        if self.GPUs: self.combined =  multi_gpu_model(self.combined, gpus=self.GPUs)
        self.combined.compile(loss=['binary_crossentropy', 'mean_absolute_error'],
                              loss_weights=[self.cGAN_weight, self.l1_weight],
                            #   metrics=['accuracy'],
                              optimizer=gOptimizer)

    def build_generator(self):
        """U-Net Generator"""

        #  Convolution-BatchNorm-ReLU layer
        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same', kernel_initializer=self.convInitializer)(layer_input)
            if bn: d = BatchNormalization(epsilon=1e-5, momentum=0.1, gamma_initializer=self.bnormInitializer)(d)
            d = LeakyReLU(alpha=0.2)(d)
            
            return d

        #  Convolution-BatchNorm-Dropout-ReLU layerwithadropoutrateof50%
        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            # u = UpSampling2D(size=2)(layer_input)
            # u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', kernel_initializer = self.convInitializer)(u)
            u = Conv2DTranspose(filters, kernel_size=f_size, strides=(2,2), padding='same', kernel_initializer = self.convInitializer)(layer_input)
            u = BatchNormalization(epsilon=1e-5, momentum=0.1, gamma_initializer=self.bnormInitializer)(u)
            if dropout_rate: u = Dropout(dropout_rate)(u)
            u = ReLU()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)
        d8 = conv2d(d7, self.gf*8)

        # Upsampling
        u0 = deconv2d(d8, d7, self.gf*8, dropout_rate=0.5)
        u1 = deconv2d(u0, d6, self.gf*8, dropout_rate=0.5)
        u2 = deconv2d(u1, d5, self.gf*8, dropout_rate=0.5)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        # u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2DTranspose(self.channels, kernel_size=4, strides=(2, 2), padding='same', activation='tanh', kernel_initializer = self.convInitializer)(u6)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True, stride=2, padding='same'):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=stride, padding=padding, kernel_initializer=self.convInitializer)(layer_input)
            if bn: d = BatchNormalization(epsilon=1e-5, momentum=0.1, gamma_initializer=self.bnormInitializer)(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)
        # d5 = d_layer(d4, self.df*8) # Tetsing larger patch size

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same', kernel_initializer=self.convInitializer, activation='sigmoid')(d4)  # NP added Sigmoid activation
        validity = Flatten()(validity)
        # self.numPatches = validity.shape[-1]

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=1, export_interval=20):

        start_time = datetime.datetime.now()
        self.batchCounter = 0

        # Adversarial loss ground truths
        valid = np.ones((batch_size, 256))
        fake = np.zeros((batch_size, 256))


        d_loss_buffer = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        g_loss_buffer = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        sma_d_loss = 1.0
        sma_g_loss = 1.0

        d_loss = 1.0
        d_acc = 0.0
        g_loss = 1.0

        for epoch in range(epochs+1):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # if the discriminator loss is greater than 1/4 of the generator loss,
                # train the discriminator
                if sma_d_loss > sma_g_loss / (2 + epoch/10)  or random.random() < 0.1:
                    fake_A = self.generator.predict(imgs_B)
                    d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                    d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    
                    d_loss_buffer[:-1] = d_loss_buffer[1:]
                    d_loss_buffer[-1] = float(d_loss[0])
                    sma_d_loss = stat.mean(d_loss_buffer)


                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss_stats = self.combined.train_on_batch(imgs_B, [valid, imgs_A])

                g_loss_cGAN = g_loss_stats[1]
                g_loss_l1 = g_loss_stats[2]
                g_loss = g_loss_stats[0] / (self.l1_weight + self.cGAN_weight)

                g_loss_buffer[:-1] = g_loss_buffer[1:]
                g_loss_buffer[-1] = float(g_loss)
                sma_g_loss = stat.mean(g_loss_buffer)


                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %3d/%3d] [Batch %3d/%3d] [D sma-loss: %3.3f, acc: %3d%%] [G  sma-loss: %3.3f cGAN-loss: %3.3f l1-loss: %3.3f] time: %s" % (epoch, epochs, batch_i, self.data_loader.n_batches, 1000*sma_d_loss, 100*d_loss[1], 1000*sma_g_loss, 1000*g_loss_cGAN, 1000*g_loss_l1, elapsed_time))

                # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, batch_i)

            if epoch % export_interval == 0:                                                                                                      
                self.exportModel(epoch)

    def sample_images(self, epoch, batch_i):
        os.makedirs('../images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)

        #?? Turn to RGB...
        imgs_A = np.concatenate([imgs_A, imgs_A, imgs_A], axis=3)
        imgs_B = np.concatenate([imgs_B, imgs_B, imgs_B], axis=3)
        fake_A = np.concatenate([fake_A, fake_A, fake_A], axis=3)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("../images/%s/%d.png" % (self.dataset_name, epoch))
        plt.close()

    # Saves a TFJS model
    def exportModel(self, epoch):
        path = '../exports/%s/gen_%s' % (self.dataset_name, str(epoch))
        os.makedirs(path, exist_ok=True)
        exportTfjs(self.generator, path)


if __name__ == '__main__':
    directory = '../public'
    location = 'Yos'
    data_type = 'train'
    size = '256'
    scale = '1'
    channelA = 'topo'
    

    #??----------------------------------------------------
    #??  RUN A FULL SET OF MODEL TRAININGS ON ALL CHANNELS
    #??----------------------------------------------------

    # HILLSHADE
    channelB = 'hillshade'
    gan = Pix2Pix(directory, location, size, scale, data_type, channelA, channelB)
    gan.train(epochs=200, batch_size=4, sample_interval=1, export_interval=20)

    # ASPECT
    channelB = 'aspect'
    gan = Pix2Pix(directory, location, size, scale, data_type, channelA, channelB)
    gan.train(epochs=200, batch_size=4, sample_interval=1, export_interval=20)

    # HYDRO
    channelB = 'hydro'
    gan = Pix2Pix(directory, location, size, scale, data_type, channelA, channelB)
    gan.train(epochs=200, batch_size=4, sample_interval=1, export_interval=20)

    # SLOPE
    channelB = 'slope'
    gan = Pix2Pix(directory, location, size, scale, data_type, channelA, channelB)
    gan.train(epochs=200, batch_size=4, sample_interval=1, export_interval=20)

    #??----------------------------------------------------
    scale = 2
    #??----------------------------------------------------

    # GRID 8 Binary
    channelB = 'contour100'
    gan = Pix2Pix(directory, location, size, scale, data_type, channelA, channelB)
    gan.train(epochs=200, batch_size=4, sample_interval=1, export_interval=20)

    # GRID 8 Binary
    channelB = 'grid8bin2'
    gan = Pix2Pix(directory, location, size, scale, data_type, channelA, channelB)
    gan.train(epochs=200, batch_size=4, sample_interval=1, export_interval=20)

    # GRID 8 4-Color
    channelB = 'grid8bin4'
    gan = Pix2Pix(directory, location, size, scale, data_type, channelA, channelB)
    gan.train(epochs=200, batch_size=4, sample_interval=1, export_interval=20)

    # GRID 16 Binary
    channelB = 'grid16bin2'
    gan = Pix2Pix(directory, location, size, scale, data_type, channelA, channelB)
    gan.train(epochs=200, batch_size=4, sample_interval=1, export_interval=20)

    # GRID 16 4-Color
    channelB = 'grid16bin4'
    gan = Pix2Pix(directory, location, size, scale, data_type, channelA, channelB)
    gan.train(epochs=200, batch_size=4, sample_interval=1, export_interval=20)
