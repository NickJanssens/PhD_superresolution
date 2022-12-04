import numpy as np
from keras.layers import Conv3D, BatchNormalization, MaxPooling2D, UpSampling3D, Conv3DTranspose
from keras.layers import Activation, Add, Concatenate, Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import concatenate, Reshape
from keras.layers import Dense, Flatten
from keras.layers import Input, add
from keras.models import Model
from Processing.instanceNormalization import InstanceNormalization
from keras.initializers import RandomNormal

init = RandomNormal(mean=0.0, stddev=0.02)

'''
Generator
'''


def generator_full_unet(input_shape):
    def conv2d(layer_input, filters, f_size=4):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same', kernel_initializer=init)(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', kernel_initializer=init, activation='relu')(
            u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    unet_input = Input(shape=input_shape)
    gf = 32
    channels = input_shape[-1]

    # Downsampling
    d1 = conv2d(unet_input, gf)
    d2 = conv2d(d1, gf * 2)
    d3 = conv2d(d2, gf * 4)
    d4 = conv2d(d3, gf * 8)

    # Upsampling
    u1 = deconv2d(d4, d3, gf * 4)
    u2 = deconv2d(u1, d2, gf * 2)
    u3 = deconv2d(u2, d1, gf)

    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(channels, kernel_size=4, strides=1, padding='same',
                        kernel_initializer=init, activation='tanh')(u4)

    return Model(unet_input, output_img)


def generator_unet_cyclegan_addinit(input_shape):
    def conv2d(layer_input, filters, f_size=4):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same',
                   kernel_initializer=init)(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same',
                   kernel_initializer=init)(u)
        u = LeakyReLU(alpha=0.2)(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    unet_input = Input(shape=input_shape)
    gf = 32
    channels = input_shape[-1]

    # Downsampling
    d1 = conv2d(unet_input, gf)
    d2 = conv2d(d1, gf * 2)
    d3 = conv2d(d2, gf * 4)
    d4 = conv2d(d3, gf * 8)

    # Upsampling
    u1 = deconv2d(d4, d3, gf * 4)
    u2 = deconv2d(u1, d2, gf * 2)
    u3 = deconv2d(u2, d1, gf)

    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(channels, kernel_size=4, strides=1, padding='same',
                        kernel_initializer=init, activation='tanh')(u4)

    return Model(unet_input, output_img)


def generator_unet_cyclegan_addinit_conv3(input_shape):
    def conv2d(layer_input, filters, f_size=3):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same',
                   kernel_initializer=init)(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=3, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same',
                   kernel_initializer=init)(u)
        u = LeakyReLU(alpha=0.2)(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    unet_input = Input(shape=input_shape)
    gf = 32
    channels = input_shape[-1]

    # Downsampling
    d1 = conv2d(unet_input, gf)
    d2 = conv2d(d1, gf * 2)
    d3 = conv2d(d2, gf * 4)
    d4 = conv2d(d3, gf * 8)

    # Upsampling
    u1 = deconv2d(d4, d3, gf * 4)
    u2 = deconv2d(u1, d2, gf * 2)
    u3 = deconv2d(u2, d1, gf)

    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(channels, kernel_size=3, strides=1, padding='same',
                        kernel_initializer=init, activation='tanh')(u4)

    return Model(unet_input, output_img)


def generator_unet_cyclegan_addinit_conv3_extraconv(input_shape):
    def conv2d(layer_input, filters, f_size=3):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=1, padding='same',
                   kernel_initializer=init)(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)

        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same',
                   kernel_initializer=init)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=3, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same',
                   kernel_initializer=init)(u)
        u = LeakyReLU(alpha=0.2)(u)

        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)

        u = Concatenate()([u, skip_input])

        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same',
                   kernel_initializer=init)(u)
        u = LeakyReLU(alpha=0.2)(u)
        u = InstanceNormalization()(u)

        return u

    unet_input = Input(shape=input_shape)
    gf = 32
    channels = input_shape[-1]

    # Downsampling
    d1 = conv2d(unet_input, gf)
    d2 = conv2d(d1, gf * 2)
    d3 = conv2d(d2, gf * 4)
    d4 = conv2d(d3, gf * 8)

    # Upsampling
    u1 = deconv2d(d4, d3, gf * 4)
    u2 = deconv2d(u1, d2, gf * 2)
    u3 = deconv2d(u2, d1, gf)

    u4 = UpSampling2D(size=2)(u3)
    u5 = Conv2D(channels, kernel_size=3, strides=1, padding='same',
                kernel_initializer=init)(u4)
    u5 = LeakyReLU(alpha=0.2)(u5)
    u5 = InstanceNormalization()(u5)
    output_img = Conv2D(channels, kernel_size=3, strides=1, padding='same',
                        kernel_initializer=init, activation='tanh')(u5)

    return Model(unet_input, output_img)


def generator_unet_real(input_shape):
    def conv2d(layer_input, filters, f_size=3):
        """Layers used during downsampling"""
        d = Conv3D(filters, kernel_size=f_size, strides=1, padding='same',
                   kernel_initializer=init)(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)

        d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same',
                   kernel_initializer=init)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=3, dropout_rate=0):
        """Layers used during upsampling"""
        u = Conv3DTranspose(filters, kernel_size=f_size, strides=2, padding='same',
                            kernel_initializer=init)(layer_input)
        u = LeakyReLU(alpha=0.2)(u)

        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)

        u = Concatenate()([u, skip_input])

        u = Conv3DTranspose(filters, kernel_size=f_size, strides=1, padding='same',
                            kernel_initializer=init)(u)
        u = LeakyReLU(alpha=0.2)(u)
        u = InstanceNormalization()(u)

        return u

    unet_input = Input(shape=input_shape)
    gf = 32
    channels = input_shape[-1]

    # Downsampling
    d1 = conv2d(unet_input, gf)
    d2 = conv2d(d1, gf * 2)
    d3 = conv2d(d2, gf * 4)
    d4 = conv2d(d3, gf * 8)

    # Upsampling
    u1 = deconv2d(d4, d3, gf * 4)
    u2 = deconv2d(u1, d2, gf * 2)
    u3 = deconv2d(u2, d1, gf)

    # u4 = UpSampling2D(size=2)(u3)
    u5 = Conv3DTranspose(channels, kernel_size=3, strides=2, padding='same',
                         kernel_initializer=init)(u3)
    u5 = LeakyReLU(alpha=0.2)(u5)
    u5 = InstanceNormalization()(u5)
    output_img = Conv3D(channels, kernel_size=3, strides=1, padding='same',
                        kernel_initializer=init, activation='tanh')(u5)

    return Model(unet_input, output_img)


def generator_unet_real_residualskip(input_shape):
    def conv2d(layer_input, filters, f_size=3):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=1, padding='same',
                   kernel_initializer=init)(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)

        d = Conv2D(filters, kernel_size=f_size, strides=1, padding='same',
                   kernel_initializer=init)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)

        # Residual Skip
        d_identity = Conv2D(filters, kernel_size=1, strides=1, padding='same',
                            kernel_initializer=init)(layer_input)
        d = Add()([d_identity, d])

        # Relu and normalization on added
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)

        # Downsampling
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same',
                   kernel_initializer=init)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)

        return d

    def deconv2d(layer_input, skip_input, filters, f_size=3, dropout_rate=0):
        """Layers used during upsampling"""
        u1 = Conv2DTranspose(filters, kernel_size=f_size, strides=2, padding='same',
                             kernel_initializer=init)(layer_input)
        u = LeakyReLU(alpha=0.2)(u1)

        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)

        u = Concatenate()([u, skip_input])

        u = Conv2DTranspose(filters, kernel_size=f_size, strides=1, padding='same',
                            kernel_initializer=init)(u)
        u = LeakyReLU(alpha=0.2)(u)
        u = InstanceNormalization()(u)

        u = Conv2DTranspose(filters, kernel_size=f_size, strides=1, padding='same',
                            kernel_initializer=init)(u)

        u_identity = Conv2D(filters, kernel_size=1, strides=1, padding='same',
                            kernel_initializer=init)(u1)
        u = Add()([u_identity, u])

        u = LeakyReLU(alpha=0.2)(u)
        u = InstanceNormalization()(u)

        return u

    unet_input = Input(shape=input_shape)
    gf = 32
    channels = input_shape[-1]

    # Downsampling
    d1 = conv2d(unet_input, gf)
    d2 = conv2d(d1, gf * 2)
    d3 = conv2d(d2, gf * 4)
    d4 = conv2d(d3, gf * 8)

    # Upsampling
    u1 = deconv2d(d4, d3, gf * 4)
    u2 = deconv2d(u1, d2, gf * 2)
    u3 = deconv2d(u2, d1, gf)

    # u4 = UpSampling2D(size=2)(u3)
    u5 = Conv2DTranspose(channels, kernel_size=3, strides=2, padding='same',
                         kernel_initializer=init)(u3)
    u5 = LeakyReLU(alpha=0.2)(u5)
    u5 = InstanceNormalization()(u5)
    output_img = Conv2D(channels, kernel_size=3, strides=1, padding='same',
                        kernel_initializer=init, activation='tanh')(u5)

    return Model(unet_input, output_img)


def generator_unet_cyclegan(input_shape):
    def conv2d(layer_input, filters, f_size=4):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    unet_input = Input(shape=input_shape)
    gf = 32
    channels = input_shape[-1]

    # Downsampling
    d1 = conv2d(unet_input, gf)
    d2 = conv2d(d1, gf * 2)
    d3 = conv2d(d2, gf * 4)
    d4 = conv2d(d3, gf * 8)

    # Upsampling
    u1 = deconv2d(d4, d3, gf * 4)
    u2 = deconv2d(u1, d2, gf * 2)
    u3 = deconv2d(u2, d1, gf)

    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

    return Model(unet_input, output_img)


def discriminator(input_shape):
    def d_layer(layer_input, filters, f_size=4, normalization=True):
        """Discriminator layer"""
        d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same', kernel_initializer=init)(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = InstanceNormalization()(d)
        return d

    img = Input(shape=input_shape)
    df = 64

    d1 = d_layer(img, df, normalization=False)
    d2 = d_layer(d1, df * 2)
    d3 = d_layer(d2, df * 4)
    d4 = d_layer(d3, df * 8)

    d5 = Conv3D(1, kernel_size=4, strides=1, padding='same', kernel_initializer=init)(d4)
    d6 = Flatten()(d5)
    d7 = Dense(1024, kernel_initializer=init)(d6)
    d8 = LeakyReLU()(d7)
    d8 = Dropout(0.5)(d8)
    validity = Dense(1, activation='sigmoid')(d8)
    return Model(img, validity)


def gan(input_shape, generator, discriminator):
    input_img = Input(shape=input_shape)
    model = generator(input_img)
    validity = discriminator(model)

    return Model(input_img, [model, validity])


# Residual block
def res_block_gen(model, kernel_size, filters, strides):
    gen = model

    model = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    # Using Parametric ReLU
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
    model = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)

    model = add([gen, model])

    return model


def generator_srgan(image_shape):
    gen_input = Input(shape=image_shape)

    model = Conv3D(filters=64, kernel_size=9, strides=1, padding="same")(gen_input)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(
        model)

    gen_model = model

    # Using 16 Residual Blocks
    for index in range(16):
        model = res_block_gen(model, 3, 64, 1)

    model = Conv3D(filters=64, kernel_size=3, strides=1, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = add([gen_model, model])

    # Using 2 UpSampling Blocks
    # for index in range(2):
    #     model = up_sampling_block(model, 3, 256, 1)

    model = Conv3D(filters=1, kernel_size=9, strides=1, padding="same")(model)
    model = Activation('tanh')(model)

    generator_model = Model(inputs=gen_input, outputs=model)

    return generator_model
