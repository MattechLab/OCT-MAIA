from __future__ import print_function, division
from keras.models import load_model
import numpy as np


class CGAN():
    def __init__(self, load_path, latent_dim=100, num_classes=10, num_gens=100, gen_save_step=1000, gen_range=(0, 750)):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.num_gens = num_gens                 # number of generators to sample from
        self.gen_save_step = gen_save_step       # step size to use to increment through generator epoch numbers (equivalent to the step size for saving the generators)
        self.gen_range_start = gen_range[0]      # epoch at start of range of generators to consider
        self.gen_range_end = gen_range[1]        # epoch at end of range of generators to consider

        self.load_path = load_path

        fids = []

        with open(self.load_path + "/fid.log", 'r') as fp:
            line = fp.readline()
            while line:
                fid = float(line.split(', ')[1].split("\n")[0])
                fids.append(fid)
                line = fp.readline()

        fids = np.asarray(fids)
        fids = fids[self.gen_range_start:self.gen_range_end]
        fid_inds = np.argsort(fids)
        best_fids = fids[fid_inds][0:self.num_gens]
        best_fid_inds = fid_inds[0:self.num_gens] * self.gen_save_step

        self.generator_list = []

        for i in range(best_fid_inds.shape[0]):
            self.generator_list.append(self.load_path + "/generator_" + str(best_fid_inds[i]) + ".hdf5")

    def get_label_tensor(self, label, batch_size):
        shape = (batch_size, int(self.img_rows / 2), int(self.img_rows / 2), self.num_classes)
        t = np.zeros(shape=shape, dtype='float32')
        for i in range(batch_size):
            idx = int(label[i])
            t[i, :, :, idx] = 1

        return t

    def generate(self, batch_class_images, num_batches):
        batches_per_gen = int(np.floor(num_batches / self.num_gens))
        samples_per_gen = batches_per_gen * batch_class_images * self.num_classes

        all_patches = np.zeros(shape=(batch_class_images * batches_per_gen * len(self.generator_list) * self.num_classes, self.img_cols, self.img_rows, 1), dtype='uint8')
        all_labels = np.zeros(shape=(batch_class_images * batches_per_gen * len(self.generator_list) * self.num_classes,), dtype='uint8')

        generator = load_model(self.generator_list[0])

        for gen in range(len(self.generator_list)):
            generator.load_weights(self.generator_list[gen])
            for k in range(batches_per_gen):
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_class_images * self.num_classes, self.latent_dim))

                # The labels of the digits that the generator tries to create an
                # image representation of
                sampled_labels = np.array([num for _ in range(batch_class_images) for num in range(self.num_classes)])
                label_tensor = self.get_label_tensor(sampled_labels, batch_class_images * self.num_classes)

                gen_imgs = generator.predict([noise, sampled_labels, label_tensor])
                gen_imgs = (0.5 * gen_imgs + 0.5) # between [0,1]    * 255

                batch_size = batch_class_images * self.num_classes

                all_patches[gen * samples_per_gen + k * batch_size:gen * samples_per_gen + (k + 1) * batch_size] = gen_imgs
                all_labels[gen * samples_per_gen + k * batch_size:gen * samples_per_gen + (k + 1) * batch_size] = sampled_labels

        all_patches = np.squeeze(all_patches)

        ### SAVE GENERATED PATCHES (all_patches) AND CORRESPONDING LABELS (all_labels) HERE


if __name__ == '__main__':
    acgan = CGAN(load_path="cgan_32x32_patchbased", latent_dim=100, num_classes=10, num_gens=100, gen_save_step=1000, gen_range=(0, 750))
    acgan.generate(batch_class_images=16, num_batches=18015)
