import os
import cv2
import tqdm
import numpy as np
# from D2_GAN.Models import discriminator, generator_unet
from Processing.Models_test import generator_unet_cyclegan, generator_unet_cyclegan_addinit, \
    discriminator, gan, generator_unet_cyclegan_addinit_conv3, generator_unet_real, generator_unet_cyclegan_addinit_conv3_extraconv
# from D2_GAN.Models import discriminator, generator_unet
from keras.optimizers import Adam
from D2_GAN.output_network import save_weights, intermediate_images
from D2_GAN.Losses import jaccard_distance_loss
from D2_GAN.utils import make_summary_fion
'''
FUNCTION TO LOAD IMAGES
'''
def load_images(path):
    image_array = []
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file), 0)
        img_reshape = (img - 127.5) / 127.5
        image_array.append(img_reshape)

    image_array = np.asarray(image_array)
    image_array = np.expand_dims(image_array, axis=3)

    return image_array

def train_GAN(input_path, output_master_path, HR_array, LR_array, input_shape, batch_size, lr_d, lr_g, beta1_d, beta1_g,
              n_pretrain_epochs, n_train_discrim, loss_weights_1, n_epochs, d_loss_fion, g_loss_fion):
    # Create output folders
    experiment_name = 'g_%s_lr%0.5f_b1%0.1f_d_%s_lr%0.5f_b1%0.1f_losswt%d' % (g_loss_fion, lr_g, beta1_g, d_loss_fion,
                                                                              lr_d, beta1_d, loss_weights_1)
    output_path = os.path.join(output_master_path, experiment_name)
    output_weights = os.path.join(output_path, 'weights')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(output_weights):
        os.mkdir(output_weights)

    n_channels = input_shape[-1]
    '''
    PREPROCESSING
    '''
    # Make a simulation summary
    final_activation = 'tanh'
    make_summary_fion(output_path, n_channels, input_shape[0], batch_size, n_epochs, n_pretrain_epochs, n_train_discrim, beta1_d,
                      beta1_g, d_loss_fion, g_loss_fion, [loss_weights_1, 1], final_activation, lr_g, lr_d)


    #Initialize optimizers - learning rate and beta 1 are chosen to be similar to DC-GAN
    opt_d = Adam(lr=lr_d, beta_1=beta1_d)
    opt_gan = Adam(lr=lr_g, beta_1=beta1_g)

    #Initialize and compile discriminator
    discriminator_model = discriminator(input_shape)  # ((128,128,1))
    discriminator_model.compile(optimizer=opt_d, loss=d_loss_fion, metrics=['accuracy'])

    #Initialize generator
    generator = generator_unet_real(input_shape)  # generator_unet_cyclegan((128,128,1))

    #Put discriminator trainable on False. Keep in mind that because of compilation of discriminator
    # before, the weights of the discriminator remain trainable, although the discriminator that will
    # compiled in the gan will not have trainable weights
    discriminator_model.trainable = False

    #Create and compile GAN
    gan_model = gan(input_shape, generator, discriminator_model)
    gan_model.compile(optimizer=opt_gan, loss=[g_loss_fion, d_loss_fion], loss_weights=[loss_weights_1, 1])

    #Real and fake labels for simulation
    labels_real = np.ones((batch_size, 1))
    labels_fake = np.zeros((batch_size, 1))

    '''
    PRE-TRAIN DISCRIMINATOR
    '''
    for _ in tqdm.tqdm(range(n_pretrain_epochs)):
        permutated_indexes = np.random.permutation(HR_array.shape[0])

        d_all_losses = []

        for index in range(int(HR_array.shape[0] / batch_size)):
            batch_indices = permutated_indexes[index * batch_size:(index + 1) * batch_size]
            real_HR = HR_array[batch_indices]
            LR_batch_input = LR_array[batch_indices]

            fake_HR = generator.predict(LR_batch_input)

            d_loss_HR = discriminator_model.train_on_batch(real_HR, labels_real)
            d_loss_LR = discriminator_model.train_on_batch(fake_HR, labels_fake)
            d_loss = 0.5 * np.add(d_loss_HR, d_loss_LR)
            d_all_losses.append(d_loss)
        d_all_losses_mn = np.mean(d_all_losses, axis=0)
        print(d_all_losses_mn)



    for epoch in tqdm.tqdm(range(n_epochs)):
        permutated_indexes = np.random.permutation(HR_array.shape[0])

        d_all_losses = []
        g_all_losses = []
        d_all_fake = []
        d_all_real = []
        for index in range(int(HR_array.shape[0] / batch_size)):
            batch_indices = permutated_indexes[index * batch_size:(index + 1) * batch_size]
            real_HR = HR_array[batch_indices]
            LR_batch_input = LR_array[batch_indices]

            labels_real = np.ones((batch_size, 1)) #- (np.random.rand(batch_size, 1) / 50)
            labels_fake = np.zeros((batch_size, 1)) #+ (np.random.rand(batch_size, 1) / 50)

            fake_HR = generator.predict(LR_batch_input)

            for _ in range(n_train_discrim):
                d_loss_HR = discriminator_model.train_on_batch(real_HR, labels_real)
                d_loss_LR = discriminator_model.train_on_batch(fake_HR, labels_fake)
                d_loss = 0.5 * np.add(d_loss_HR, d_loss_LR)
                d_all_losses.append(d_loss)
                d_all_fake.append(d_loss_LR)
                d_all_real.append(d_loss_HR)

            g_loss = gan_model.train_on_batch(LR_batch_input, [real_HR, labels_real])
            g_all_losses.append(g_loss)
        d_all_losses_mn = np.mean(d_all_losses, axis=0)
        g_all_losses_mn = np.mean(g_all_losses)
        d_all_real_mn = np.mean(d_all_real)
        d_all_fake_mn = np.mean(d_all_fake)
        print(d_all_losses_mn, g_all_losses_mn)

        with open(os.path.join(output_path, 'log.txt'), 'a+') as f:
            f.write('{} - {} - {} - {} - {} - {}\n'.format(epoch, d_all_losses_mn[0], d_all_losses_mn[1], g_all_losses_mn,
                                                      d_all_real_mn, d_all_fake_mn))

        if epoch % 5 == 0:
            save_weights(generator, discriminator_model, epoch, int(np.mean(g_all_losses_mn)), output_weights)
            intermediate_images(output_path, input_path, generator, epoch, n_channels, 3)