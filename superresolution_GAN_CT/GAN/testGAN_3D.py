import os
import cv2
import tqdm
import numpy as np
# from D2_GAN.Models import discriminator, generator_unet
from Processing.Models_test_3D import generator_unet_cyclegan, generator_unet_cyclegan_addinit, \
    discriminator, gan, generator_unet_real, generator_unet_real_residualskip, generator_unet_cyclegan_addinit_conv3, generator_unet_cyclegan_addinit_conv3_extraconv
# from D2_GAN.Models import discriminator
from Processing.Models_test_3D import generator_srgan
from keras.optimizers import Adam
from D2_GAN.output_network import save_weights, intermediate_images, intermediate_images_volumes
from D2_GAN.Losses import jaccard_distance_loss, dice_coef_loss
import keras.backend as K

'''
INPUT
'''
#Input path (should contain folders with HR and LR data - in Keras structure)
input_path = r'F:\UNet\AM23\THREEDIMENSIONAL\TRAINING'
HR_path = os.path.join(input_path, 'HR_test', 'HR_test')
LR_path = os.path.join(input_path, 'LR_test', 'LR_test')

#Image input details
dim = 64
n_channels = 1
input_shape = (dim, dim, dim, n_channels)

#Hyperparameters
batch_size = 8
n_pretrain_epochs = 3
n_epochs = 5
n_train_discrim = 1

#Create output folders
output_path = os.path.join(input_path, 'output_gan_3D_pretrain2_dtrain2')
output_weights = os.path.join(output_path, 'weights')
if not os.path.isdir(output_path):
    os.mkdir(output_path)
if not os.path.isdir(output_weights):
    os.mkdir(output_weights)

'''
FUNCTION TO LOAD VOLUMES
'''
def load_volumes(path):
    image_array = []
    for file in os.listdir(path):
        img = np.load(os.path.join(path, file))
        #img = cv2.imread(os.path.join(path, file), 0)
        # img_reshape = (img - 127.5) / 127.5
        # img_reshape = img / 255
        image_array.append(img)

    image_array = np.asarray(image_array)
    image_array = np.expand_dims(image_array, axis=4)

    return image_array



'''
PREPROCESSING
'''
#Load images into an array
HR_array = load_volumes(HR_path)
LR_array = load_volumes(LR_path)

#Initialize optimizers - learning rate and beta 1 are chosen to be similar to DC-GAN
opt_d = Adam(lr=2E-4, beta_1=0.5)
opt_gan = Adam(lr=1E-4, beta_1=0.5)

#Initialize and compile discriminator
discriminator = discriminator(input_shape)  # ((128,128,1))
discriminator.compile(optimizer=opt_d, loss='mse', metrics=['accuracy'])

#Initialize generator
generator = generator_unet_real(input_shape)  # generator_unet_cyclegan((128,128,1))

#Put discriminator trainable on False. Keep in mind that because of compilation of discriminator
# before, the weights of the discriminator remain trainable, although the discriminator that will
# compiled in the gan will not have trainable weights
discriminator.trainable = False

#Create and compile GAN
gan = gan(input_shape, generator, discriminator)
gan.compile(optimizer=opt_gan, loss=['mae', 'mse'], loss_weights=[100, 1], metrics=['accuracy'])

#Real and fake labels for simulation
labels_real = np.ones((batch_size,))
labels_fake = np.zeros((batch_size,))

discriminator.trainable = True
'''
PRE-TRAIN DISCRIMINATOR
'''
for epoch in tqdm.tqdm(range(n_pretrain_epochs)):
    permutated_indexes = np.random.permutation(HR_array.shape[0])

    d_all_losses = []

    for index in range(int(HR_array.shape[0] / batch_size)):
        batch_indices = permutated_indexes[index * batch_size:(index + 1) * batch_size]
        real_HR = np.float32(HR_array[batch_indices])
        LR_batch_input = LR_array[batch_indices]

        fake_HR = generator.predict(LR_batch_input)
        labels_real = labels_real - np.random.random_sample(batch_size)*0.1

        d_loss_HR = discriminator.train_on_batch(real_HR, labels_real)
        d_loss_LR = discriminator.train_on_batch(fake_HR, labels_fake)
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
        real_HR = np.float32(HR_array[batch_indices])
        LR_batch_input = LR_array[batch_indices]

        labels_real = np.ones((batch_size,)) - np.random.random_sample(batch_size)*0.1#- (np.random.rand(batch_size, 1) / 50)
        labels_fake = np.zeros((batch_size,)) #+ (np.random.rand(batch_size, 1) / 50)

        fake_HR = generator.predict(LR_batch_input)

        for _ in range(n_train_discrim):
            d_loss_HR = discriminator.train_on_batch(real_HR, labels_real)
            d_loss_LR = discriminator.train_on_batch(fake_HR, labels_fake)
            d_loss = 0.5 * np.add(d_loss_HR, d_loss_LR)
            d_all_losses.append(d_loss)
            d_all_fake.append(d_loss_LR)
            d_all_real.append(d_loss_HR)

        discriminator.trainable = False

        g_loss = gan.train_on_batch(LR_batch_input, [real_HR, labels_real])

        discriminator.trainable = True

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
            save_weights(generator, discriminator, epoch, int(np.mean(g_all_losses_mn)), output_weights)
            intermediate_images_volumes(output_path, LR_path, HR_path, generator, epoch, 3)