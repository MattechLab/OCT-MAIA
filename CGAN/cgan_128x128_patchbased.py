from __future__ import print_function, division

import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import sys
from cgan_70x70_patchbased import load_patches

from Minibatch import MinibatchDiscrimination


class CGAN():
    def __init__(self, filepath, replay=True, replay_start_record=5000, replay_start_substitution=6000, replay_interval=100,
                 replay_examples=1, replay_file=True, replay_proportion=0.3, latent_dim=100, num_classes=10):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.filepath = filepath

        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)

        self.replay = replay
        if self.replay is True:
            self.replay_start_record = replay_start_record
            self.replay_start_substitution = replay_start_substitution
            self.replay_interval = replay_interval
            self.replay_examples = replay_examples
            self.replay_file = replay_file
            self.replay_proportion = replay_proportion
            self.replay_count = 0

            if self.replay_file is True:
                self.replay_file = h5py.File(self.filepath + "/replay.hdf5", 'a')
            else:
                self.gen_history = []
                self.label_history = []

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_deep_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        label_tensor = Input(shape=(int(self.img_rows / 2), int(self.img_rows / 2),
                                    self.num_classes), dtype='float32')
        img = self.generator([noise, label, label_tensor])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label, label_tensor], valid)
        self.combined.compile(loss=losses,
            optimizer=optimizer)

    def build_deep_generator(self):
        d1 = int(self.img_rows / 8)

        model = Sequential()
        # first block (Dense)
        model.add(Dense(256 * d1 * d1, input_dim=self.latent_dim, kernel_initializer='he_normal'))
        model.add(BatchNormalization(momentum=0.8))
        #        model.add(Dropout(rate = 0.4) )
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((d1, d1, 256)))  # size:(d1,d1,256)
        # second block (Convolutional)
        model.add(UpSampling2D(interpolation='bilinear'))  # size: (2*d1,2*d1,256)
        model.add(Conv2D(256, kernel_size=3, padding="same", kernel_initializer='he_normal'))
        model.add(BatchNormalization(momentum=0.8))
        #        model.add(Dropout(rate = 0.4) )
        model.add(LeakyReLU(alpha=0.2))
        # third block (Convolutional and concatenation)
        model.add(UpSampling2D(interpolation='bilinear'))  # size: (4*d1,4*d1,256)

        model2 = Sequential()  # size: (4*d1,4*d1,256+num_classes)
        model2.add(Conv2D(128, kernel_size=3, padding="same",  # size: (4*d1,4*d1,128)
                          input_shape=(4 * d1, 4 * d1, 256 + self.num_classes), kernel_initializer='he_normal'))
        model2.add(BatchNormalization(momentum=0.8))
        model2.add(LeakyReLU(alpha=0.2))
        # fourth block (Convolutional)
        model2.add(UpSampling2D(interpolation='bilinear'))  # size: (img_size,img_size,128)
        model2.add(Conv2D(64, kernel_size=3, padding="same",
                          kernel_initializer='he_normal'))  # size: (img_size,img_size,64)
        model2.add(BatchNormalization(momentum=0.8))
        model2.add(LeakyReLU(alpha=0.2))
        # Convolutional layer + activation
        model2.add(Conv2D(self.channels, kernel_size=3, padding='same'))  # size: (img_size,img_size,3)
        model2.add(Activation("tanh"))

        model2.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_tensor = Input(shape=(4 * d1, 4 * d1, self.num_classes), dtype='float32')

        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        model_input = multiply([noise, label_embedding])
        r = model(model_input)
        merged = concatenate([r, label_tensor])

        img = model2(merged)

        return Model([noise, label, label_tensor], img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=1, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D())
        model.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D())
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D())
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(MinibatchDiscrimination(25, 15))
        model.add(Dense(1, activation="sigmoid"))

        model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Embedding(self.num_classes, np.prod(self.img_shape))(label)
        label_embedding = Reshape(target_shape=self.img_shape)(label_embedding)

        model_input = multiply([img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)

    def train(self, epochs, batch_size=128, sample_interval=50, model_interval=1000, stride=2, thresh=0.8, smooth_type="onesided"):

        total_replay_examples = (epochs - self.replay_start_record) * self.replay_examples

        img_shape_list = list(self.img_shape)
        img_shape_list.insert(0, total_replay_examples)
        replay_imgs_shape = tuple(img_shape_list)
        replay_labels_shape = (total_replay_examples, 1)

        self.replay_imgs_dataset = self.replay_file.create_dataset(name="replay_imgs", shape=replay_imgs_shape)
        self.replay_labels_dataset = self.replay_file.create_dataset(name="replay_labels", shape=replay_labels_shape)

        # Load the dataset
        (X_train, y_train) = self.load_data(stride, thresh)

        print(X_train.shape)
        print(y_train.shape)

        # Adversarial ground truths
        valid_o = np.ones((batch_size, 1))
        fake_o = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # Label smoothing:
            if smooth_type is not None:
                valid = self.label_smoothing(valid_o, smooth_type=smooth_type)
                fake = self.label_smoothing(fake_o, smooth_type=smooth_type)
            else:
                valid = valid_o
                fake = fake_o

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            imgs = (imgs.astype(np.float32) - 0.5) / 0.5 # for the images to be in [-1,1]

            # Image labels. 0-9
            img_labels = y_train[idx]

            img_labels = img_labels.astype(np.float32)

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # The labels of the digits that the generator tries to create an
            # image representation of
            sampled_labels = np.random.randint(0, self.num_classes, (batch_size, 1))
            label_tensor = self.get_label_tensor(sampled_labels, batch_size)

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, sampled_labels, label_tensor])

            if self.replay:
                if self.replay_file:
                    if epoch > self.replay_start_record and epoch % self.replay_interval:
                        for i in range(self.replay_examples):
                            self.replay_imgs_dataset[self.replay_count] = gen_imgs[i]
                            self.replay_labels_dataset[self.replay_count] = sampled_labels[i]
                            self.replay_count += 1
                    if epoch > self.replay_start_substitution:
                        gen_imgs, sampled_labels = self.add_replays(gen_imgs, sampled_labels)
                        label_tensor = self.get_label_tensor(sampled_labels, batch_size)
                else:
                    if epoch > self.replay_start_record and epoch % self.replay_interval:
                        for i in range(self.replay_examples):
                            self.gen_history.append(gen_imgs[i])
                            self.label_history.append(sampled_labels[i])
                            self.replay_count += 1
                    if epoch > self.replay_start_substitution:
                        gen_imgs, sampled_labels = self.add_replays(gen_imgs, sampled_labels)
                        label_tensor = self.get_label_tensor(sampled_labels, batch_size)

            sampled_labels = sampled_labels.astype(np.float32)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, img_labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, sampled_labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels, label_tensor], valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

            if epoch % model_interval == 0:
                self.save_model(epoch)

        self.replay_file.close()

    def sample_images(self, epoch):
        r, c = 10, self.num_classes
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])

        label_tensor = self.get_label_tensor(sampled_labels, r * c)
        gen_imgs = self.generator.predict([noise, sampled_labels, label_tensor])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(np.transpose(gen_imgs[cnt,:,:,0]), cmap='gray', vmin=0, vmax=1)
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(self.filepath + "/%d.png" % epoch)
        plt.close()

    def save_model(self, epoch):
        self.generator.save(self.filepath + "/generator_%s.hdf5" % str(epoch))

    def load_data(self, stride, thresh):
        """
        Loads your image data (x_train) and corresponding class labels (y train) here.
        INPUT:
        - stride: stride between the patches in the original OCT image,
        - thresh: treshold.
        OUTPUT:
        - x_train: patches,
        - y_train: labels.
        """
        x_train, y_train = load_patches("/data/line/OCT-patchbased-CGAN/generated_patches/128x128/training", stride, thresh)
        return x_train, y_train

    def label_smoothing(self, vector, max_dev=0.2, smooth_type="onesided"):
        d = max_dev * np.random.rand(vector.shape[0], vector.shape[1])
        if vector[0][0] == 0:
            if smooth_type == "onesided":
                return vector
            else:
                return vector + d
        else:
            return vector - d

    def add_replays(self, gen_imgs, sampled_labels):
        """
        Substitute randomly a portion of the newly generated images with some
        older (generated) ones
        """
        if self.replay_file is False:
            n = int(gen_imgs.shape[0] * self.replay_proportion)
            n = min(n, len(self.label_history))
            idx_gen = np.random.randint(0, gen_imgs.shape[0], n)
            idx_hist = np.random.randint(0, len(self.gen_history), n)
            for i_g, i_h in zip(idx_gen, idx_hist):
                gen_imgs[i_g] = self.gen_history[i_h]
                sampled_labels[i_g] = self.label_history[i_h]
        else:
            n = int(gen_imgs.shape[0] * self.replay_proportion)
            n = min(n, self.replay_count)
            idx_gen = np.random.randint(0, gen_imgs.shape[0], n)
            idx_hist = np.random.randint(0, self.replay_count, n)
            for i_g, i_h in zip(idx_gen, idx_hist):
                gen_imgs[i_g] = self.replay_imgs_dataset[i_h]
                sampled_labels[i_g] = self.replay_labels_dataset[i_h]

        return gen_imgs, sampled_labels

    def get_label_tensor(self, label, batch_size):
        shape = (batch_size, int(self.img_rows / 2), int(self.img_rows / 2), self.num_classes)
        t = np.zeros(shape=shape, dtype='float32')
        for i in range(batch_size):
            idx = int(label[i])
            t[i, :, :, idx] = 1

        return t


if __name__ == '__main__':
    """ default:
    cgan = CGAN(filepath="cgan_128x128_patchbased", replay=True, replay_start_record=5000, replay_start_substitution=6000, replay_interval=100,
                replay_examples=1, replay_file=True, latent_dim=100, num_classes=10)
    cgan.train(epochs=1000000, batch_size=64, sample_interval=100, model_interval=1000, smooth_type="onesided")
    """
        # default variables
    stride = 4
    thresh = 0.8

    # command
    args = sys.argv[1:]
    for i in range(len(args)):
        if args[i] == '-s':
            stride = args[i+1]
        if args[i] == '-t':
            thresh = np.double(args[i+1])/100
    
    print("Stride=", stride)
    print("Threshold=", thresh)
    
    print(tf.config.list_physical_devices('GPU'))
    cgan = CGAN(filepath="/data/line/OCT-patchbased-CGAN/cgan_128x128_patchbased"+'/thresh='+ str(int(thresh*100))+"/s="+str(stride), replay=True, replay_start_record=5000, replay_start_substitution=6000, replay_interval=100,
                replay_examples=1, replay_file=True, latent_dim=100, num_classes=2)
    cgan.train(epochs=100000, batch_size=64, sample_interval=100, model_interval=1000, stride=stride, thresh=thresh, smooth_type="onesided")