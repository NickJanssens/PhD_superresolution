import os
import cv2
import tqdm
import numpy as np
#from D2_GAN.Models import discriminator, generator_unet
from Processing.Models_test import generator_unet_cyclegan, generator_unet_cyclegan_addinit #, discriminator
from D2_GAN.Models import discriminator, generator_unet
from keras.optimizers import Adam


input_path = r'C:\Users\u0105452\PhD\NeuralNetworks\UNet\XRE'
HR_path = os.path.join(input_path, 'HR_test', 'HR_test')
LR_path = os.path.join(input_path, 'LR_test', 'LR_test')
dim = 128
batch_size = 4
n_epochs = 2


def load_reshape_images(path, dim):
    image_array = []
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file))
        img_reshape = (cv2.resize(img, (dim, dim)) - 127.5) / 127.5
        image_array.append(img_reshape)
    image_array = np.asarray(image_array)

    return image_array

def load_images(path):
    image_array = []
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file), 0)
        img_reshape = (img - 127.5) / 127.5
        image_array.append(img_reshape)

    image_array = np.asarray(image_array)
    image_array = np.expand_dims(image_array, axis=3)

    return image_array

def make_random_images(dim,batch_size):
    random_image = np.random.rand(batch_size,dim,dim)
    scaled_random_image = (random_image - 0.5) / 0.5

    scaled_random_image = np.expand_dims(scaled_random_image, axis=3)
    return scaled_random_image

def make_simulated_images(LR_images, generator):
    simulated_image = generator.predict(LR_images)
    scaled_simulated_image = (simulated_image - 0.5) / 0.5

    return simulated_image, generator

HR_array = load_images(HR_path)
LR_array = load_images(LR_path)

opt_d = Adam(lr=2E-4, beta_1=0.5)

discriminator = discriminator() #((128,128,1))
discriminator.compile(optimizer=opt_d, loss='binary_crossentropy', metrics=['accuracy'])
generator = generator_unet(1) #generator_unet_cyclegan((128,128,1))
generator.compile(optimizer=opt_d, loss='binary_crossentropy', metrics=['accuracy'])

labels_HR = np.ones((batch_size, 1))
labels_LR = np.zeros((batch_size, 1))

for epoch in tqdm.tqdm(range(n_epochs)):
    permutated_indexes = np.random.permutation(HR_array.shape[0])

    d_all_losses = []
    for index in range(int(HR_array.shape[0] / batch_size)):
        batch_indices = permutated_indexes[index * batch_size:(index + 1) * batch_size]
        HR_batch = HR_array[batch_indices]
        LR_batch_input = LR_array[batch_indices]
        LR_batch, generator = make_simulated_images(LR_batch_input, generator)

        d_loss_HR = discriminator.train_on_batch(HR_batch, labels_HR)
        d_loss_LR = discriminator.train_on_batch(LR_batch, labels_LR)
        # d_loss = discriminator.train_on_batch(discrim_input, discrim_output)

        d_loss = 0.5 * np.add(d_loss_HR, d_loss_LR)
        d_all_losses.append(d_loss)

    d_all_losses = np.mean(d_all_losses, axis=0)
    print(d_all_losses)
