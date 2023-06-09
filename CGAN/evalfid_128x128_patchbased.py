from __future__ import print_function, division
from keras.models import load_model
import numpy as np
from GAN import fid
from cgan_70x70_patchbased import load_patches

class CGAN():
    def __init__(self, load_path, num_classes=10, latent_dim=100, gen_save_step=1000):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.gen_save_step = gen_save_step

        self.load_path = load_path

        self.x_train, self.y_train = self.load_data()

        self.x_train = np.transpose(self.x_train, axes=(0, 3, 1, 2))
        self.x_train = np.repeat(self.x_train, repeats=3, axis=1)

    def get_label_tensor(self, label, batch_size):
        shape = (batch_size, int(self.img_rows / 2), int(self.img_rows / 2), self.num_classes)
        t = np.zeros(shape=shape, dtype='float32')
        for i in range(batch_size):
            idx = int(label[i])
            t[i, :, :, idx] = 1

        return t

    def load_data(self):
        # load your image data (x_train) and corresponding class labels (y train) here
        return #x_train, y_train

    def calculate_inception(self, batch_class_images, num_batches):
        import os
        items = os.listdir(self.load_path)

        new_items = []

        for item in items:
            if item.endswith(".hdf5") and item.startswith("generator_"):
                new_item = int(item.split(".")[0].split("_")[1])
                if new_item % self.gen_save_step == 0:
                    new_items.append(new_item)

        new_items.sort()

        batch_size = batch_class_images * self.num_classes

        generator = load_model(self.load_path + "/generator_" + str(new_items[0]) + ".hdf5")

        for generator_file in new_items:
            generator.load_weights(self.load_path + "/generator_" + str(generator_file) + ".hdf5")
            all_patches = np.zeros(
                shape=(batch_class_images * num_batches * self.num_classes, self.img_cols, self.img_rows, 1),
                dtype='uint8')

            for k in range(num_batches):
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_class_images * self.num_classes, self.latent_dim))

                # The labels of the digits that the generator tries to create an
                # image representation of
                sampled_labels = np.array([num for _ in range(batch_class_images) for num in range(self.num_classes)])
                label_tensor = self.get_label_tensor(sampled_labels, batch_class_images * self.num_classes)

                gen_imgs = generator.predict([noise, sampled_labels, label_tensor])
                gen_imgs = (0.5 * gen_imgs + 0.5) * 255 #keep this 

                all_patches[k * batch_size:(k + 1) * batch_size] = gen_imgs

            all_patches = np.transpose(all_patches, axes=(0, 3, 1, 2))
            all_patches = np.repeat(all_patches, repeats=3, axis=1)

            fid_score = fid.get_fid(all_patches, self.x_train)

            with open(self.load_path.split("/")[0] + '/fid.log', 'a') as f:
                f.writelines('%d, %s\n' % (generator_file, fid_score))


if __name__ == '__main__':
    cgan = CGAN(load_path="cgan_32x32_patchbased", num_classes=10, latent_dim=100)
    cgan.calculate_inception(batch_class_images=1, num_batches=300)
