# inspired from https://github.com/jakugel/oct-patchbased-cgan 

from __future__ import print_function, division
from keras.models import load_model
import numpy as np
import fid
from cgan_70x70_patchbased import load_patches

class CGAN():
    def __init__(self, load_path, num_classes=2, latent_dim=100, gen_save_step=1000, stride=2, thresh=0.8):
        """
        INPUT:
        - loadpath: path where the models and images were saved,
        - num_classes: number of classes,
        - latent_dim: latent dimension of the generator,
        - gen_save_step: interval for which the model is picked,
        - stride: stride in between the patches from the original OCT images.
        """
        # Input shape
        self.img_rows = 70
        self.img_cols = 70
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.gen_save_step = gen_save_step

        self.load_path = load_path

        self.x_train, self.y_train = self.load_data(stride, thresh)

        self.x_train = np.transpose(self.x_train, axes=(0, 3, 1, 2))
        #np.repeat: repeats: number of repetitions for each element. axis=1: the axis along which to repeat values. Here axis=1: repeats on the axis of channels.
        self.x_train = np.repeat(self.x_train, repeats=3, axis=1) 

    def get_label_tensor(self, label, batch_size):
        shape = (batch_size, int(self.img_rows / 2), int(self.img_rows / 2), self.num_classes)
        t = np.zeros(shape=shape, dtype='float32')
        for i in range(batch_size):
            idx = int(label[i])
            t[i, :, :, idx] = 1

        return t

    def load_data(self, stride, thresh):
        """
        Loads your image data (x_train) and corresponding class labels (y train) here.
        INPUT:
        - stride: stride between the patches in the original OCT image.
        - thresh: threshold
        OUTPUT:
        - x_train: patches,
        - y_train: labels.
        """
        # Here we take the validation set
        x_train, y_train = load_patches("/data/line/OCT-patchbased-CGAN/generated_patches/validation", stride, thresh)
        x_train = x_train*255 # for fid, between [0, 255]
        return x_train, y_train

    def calculate_inception(self, batch_class_images, num_batches):
        """
        INPUT:
        - batch_class_images: size of the images batch (per batch, per generator)
        - num_batches: number of batches (per generator)
        """
        import os
        items = os.listdir(self.load_path)

        new_items = []

        # load the generators, at each gen_save_step, from the load_path folder:
        for item in items:
            if item.endswith(".hdf5") and item.startswith("generator_"):
                new_item = int(item.split(".")[0].split("_")[1])
                if new_item % self.gen_save_step == 0:
                    new_items.append(new_item)

        new_items.sort()

        batch_size = batch_class_images * self.num_classes

        generator = load_model(self.load_path + "/generator_" + str(new_items[0]) + ".hdf5")

        # loop on the generators:
        for generator_file in new_items:
            #load the generator weights:
            generator.load_weights(self.load_path + "/generator_" + str(generator_file) + ".hdf5")
            # store all the patches from the generator here
            all_patches = np.zeros(
                shape=(batch_class_images * num_batches * self.num_classes, self.img_cols, self.img_rows, 1),
                dtype='uint8')

            for k in range(num_batches):
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_class_images * self.num_classes, self.latent_dim))

                # The labels of the digits that the generator tries to create an image representation of
                sampled_labels = np.array([num for _ in range(batch_class_images) for num in range(self.num_classes)]) #bunch of 0s and 1s here (labels)
                label_tensor = self.get_label_tensor(sampled_labels, batch_class_images * self.num_classes)

                # generator predict from the noise, sampled labels and label_tensor
                gen_imgs = generator.predict([noise, sampled_labels, label_tensor])
                # resize the images between 0 and 255
                gen_imgs = (0.5 * gen_imgs + 0.5)* 255

                all_patches[k * batch_size:(k + 1) * batch_size] = gen_imgs

            all_patches = np.transpose(all_patches, axes=(0, 3, 1, 2))
            all_patches = np.repeat(all_patches, repeats=3, axis=1)

            # compute the FID score
            fid_score = fid.get_fid(all_patches, self.x_train)

            with open(self.load_path.split("/")[0] + '/fid.log', 'a') as f:
                f.writelines('%d, %s\n' % (generator_file, fid_score))


if __name__ == '__main__':
    stride = 2
    thresh = 0.8
    cgan = CGAN(load_path="/data/line/OCT-patchbased-CGAN/cgan_70x70_patchbased/" + "thresh=" + str(int(0.8*100)) + "/s="+str(stride), num_classes=2, latent_dim=100, stride=stride, thresh=thresh)
    cgan.calculate_inception(batch_class_images=1, num_batches=300)
