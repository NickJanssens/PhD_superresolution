import os
from Processing.testGAN_function import load_images, train_GAN

'''
INPUT
'''
# Input path (should contain folders with HR and LR data - in Keras structure)
input_path = r'F:\UNet\AM23'
HR_path = os.path.join(input_path, 'HR_test', 'HR_test')
LR_path = os.path.join(input_path, 'LR_test', 'LR_test')
output_master_path = os.path.join(input_path, 'output_gridsearch')
if not os.path.isdir(output_master_path):
    os.mkdir(output_master_path)

# Load images into an array
HR_array = load_images(HR_path)
LR_array = load_images(LR_path)

# Image input details
dim = 64
n_channels = 1
input_shape = (dim, dim, n_channels)

# Hyperparameters - fixed
n_pretrain_epochs = 2
n_epochs = 401
n_train_discrim = 1

# Hyperparameters - variable
batch_size = 32
lr_discriminator = [1E-4, 2E-4]
lr_generator = [1E-4]
beta1_discriminator = [0.5, 0.9]
beta1_generator = [0.5, 0.9]
loss_weights_1 = [10, 100, 1]
discriminator_loss = ['mse', 'mae']
generator_loss = ['mae', 'mse', 'binary_crossentropy']

for d_loss_fion in discriminator_loss:
    for g_loss_fion in generator_loss:
        for lr_disc in lr_discriminator:
            for lr_gen in lr_generator:
                for beta1_d in beta1_discriminator:
                    for beta1_g in beta1_generator:
                        for loss_wts in loss_weights_1:
                            train_GAN(input_path, output_master_path, HR_array, LR_array, input_shape, batch_size, lr_disc,
                                      lr_gen, beta1_d, beta1_g, n_pretrain_epochs, n_train_discrim, loss_wts, n_epochs,
                                      d_loss_fion, g_loss_fion)
