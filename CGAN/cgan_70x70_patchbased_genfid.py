# inspired from https://github.com/jakugel/oct-patchbased-cgan 

from __future__ import print_function, division
from keras.models import load_model
import numpy as np
import os


class CGAN():
    def __init__(self, load_path, latent_dim=100, num_classes=2, num_gens=100, gen_save_step=1000, gen_range=(0, 750)):
        """
        INPUT:
        - load_path: str, path where the generators are saved,
        - latent_dim: int, latent dimension of the generator,
        - num_classes: int, number of classes,
        - num_gens: int, number of generators to sample from for the FID score,
        - gen_save_step:  int, step size to use to increment through generator epoch numbers (equivalent to the step size for saving the generators)
        - gen_range: tuple, (epoch at start of range of generators to consider, epoch at end of range of generators to consider).
        """
        self.img_rows = 70
        self.img_cols = 70
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.num_gens = num_gens                 
        self.gen_save_step = gen_save_step       
        self.gen_range_start = gen_range[0]       
        self.gen_range_end = gen_range[1]        

        self.load_path = load_path

        # create the list of generators:
        fids = [] # list of paths of generators to consider

        # load the fid score of each generator 
        with open(self.load_path + "/fid.log", 'r') as fp:
            line = fp.readline()
            while line:
                fid = float(line.split(', ')[1].split("\n")[0])
                fids.append(fid)
                line = fp.readline()

        # take the lowest fids scores (i.e. best generator WRT fid score)
        fids = np.asarray(fids)
        fids = fids[self.gen_range_start:self.gen_range_end]
        fid_inds = np.argsort(fids)
        best_fids = fids[fid_inds][0:self.num_gens]
        best_fid_inds = fid_inds[0:self.num_gens] * self.gen_save_step

        self.generator_list = []

        # load the generators witht the lowest fid score
        for i in range(best_fid_inds.shape[0]):
            self.generator_list.append(self.load_path + "/generator_" + str(best_fid_inds[i]) + ".hdf5")

    def get_label_tensor(self, label, batch_size):
        """
        Creates the label tensor.
        INPUT:
        - label: 
        - batch_size: batch size.
        OUTPUT:
        - t: array of dimension (batch_size, img_rows/2, img_rows/2, num_classes). Is used as an input to generate patches.
        """
        shape = (batch_size, int(self.img_rows / 2), int(self.img_rows / 2), self.num_classes)
        t = np.zeros(shape=shape, dtype='float32')
        for i in range(batch_size):
            idx = int(label[i])
            t[i, :, :, idx] = 1

        return t
    
    def save_data(final_labels, final_patches, save_folder, s):
        """
        Saves patches and labels in save_folder.
        INPUT:
        - final_labels: numpy.arrays of shape (n,), labels to save,
        - final_patches: numpy.arrays of shape (n, cols, rows, 1), patches to save,
        - save_folder: str, folder path where to save the labels and patches,
        - s: int, stride considered.
        """
        if type(save_folder) == str:
            file_name_labels = save_folder + '/s=' + str(s) + '/labels.NPY'
            file_name_patches = save_folder + '/s=' + str(s) + '/patches.NPY'
            os.makedirs(os.path.dirname(file_name_labels), exist_ok=True)
            os.makedirs(os.path.dirname(file_name_patches), exist_ok=True)
            with open(file_name_labels, 'wb') as f:
                np.save(f , final_labels)
            with open(file_name_patches, 'wb') as f:
                np.save(f, final_patches)
                print("Patches saved in folder ", f)

    def generate(self, batch_class_images, num_batches, save_path, stride):
        """
        Generates and saves synthetic patches and corresponding labels, from the best generators (according to FID score).
        INPUT:
        - batch_class_images:
        - num_batches: 
        """
        batches_per_gen = int(np.floor(num_batches / self.num_gens)) # number of batches per generator
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

        # SAVE GENERATED PATCHES (all_patches) AND CORRESPONDING LABELS (all_labels) HERE
        self.save_data(final_labels=all_labels, final_patches=all_patches, save_folder=save_path, s=stride)


if __name__ == '__main__':
    stride = 2
    thresh = 0.8
    acgan = CGAN(load_path="/data/line/OCT-patchbased-CGAN/cgan_70x70_patchbased"+'/thresh='+ str(int(thresh*100))+"/s="+str(stride), latent_dim=100, num_classes=2, num_gens=100, gen_save_step=1000, gen_range=(0, 750))
    acgan.generate(batch_class_images=16, num_batches=18015, save_path="/data/line/OCT-patchbased-CGAN/GAN-generated_patches/70x70", stride=stride)
