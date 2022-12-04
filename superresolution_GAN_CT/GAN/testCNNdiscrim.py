import os
import cv2
import tqdm
import numpy as np
from D2_GAN.Models import discriminator, generator_unet
from keras.optimizers import Adam
from D2_GAN.utils import

input_path = r'C:\Users\u0105452\Downloads\Convolutional_Neural_Networks\dataset\training_set'
dogs_path = os.path.join(input_path, 'dogs')
cats_path = os.path.join(input_path, 'cats')
dim = 128
batch_size = 4
n_epochs = 100


def load_reshape_images(path, dim):
    image_array = []
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file))
        img_reshape = (cv2.resize(img, (dim, dim)) - 127.5) / 127.5
        image_array.append(img_reshape)
    image_array = np.asarray(image_array)

    return image_array


dogs_array = load_reshape_images(dogs_path, dim)
cats_array = load_reshape_images(cats_path, dim)

opt_d = Adam(lr=2E-4, beta_1=0.5)

discriminator = discriminator()
discriminator.compile(optimizer=opt_d, loss='binary_crossentropy', metrics=['accuracy'])
generator = generator_unet(1)

labels_dogs = np.zeros((batch_size, 1))
labels_cats = np.ones((batch_size, 1))

for epoch in tqdm.tqdm(range(n_epochs)):
    permutated_indexes = np.random.permutation(cats_array.shape[0])

    d_all_losses = []
    for index in range(int(cats_array.shape[0] / batch_size)):
        batch_indices = permutated_indexes[index * batch_size:(index + 1) * batch_size]
        dogs_batch = dogs_array[batch_indices]
        cats_batch = cats_array[batch_indices]

        d_loss_cats = discriminator.train_on_batch(cats_batch, labels_cats)
        d_loss_dogs = discriminator.train_on_batch(dogs_batch, labels_dogs)
        # d_loss = discriminator.train_on_batch(discrim_input, discrim_output)

        d_loss = 0.5 * np.add(d_loss_cats, d_loss_dogs)
        d_all_losses.append(d_loss)

    d_all_losses = np.mean(d_all_losses, axis=0)
    print(d_all_losses)
