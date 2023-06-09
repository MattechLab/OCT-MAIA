# inspired from https://github.com/jakugel/oct-choroid-seg/blob/master/train_script_patchbased_general.py 

import training
import training_parameters as tparams
import keras.optimizers
import image_database as imdb
import patch_based_network_models as patch_models
import dataset_construction
from keras.layers.cudnn_recurrent import CuDNNLSTM, CuDNNGRU
import h5py
from helper_patches import load_patches

keras.backend.set_image_data_format('channels_last')

INPUT_CHANNELS = 1
DATASET_NAME = "first_try"     # can choose a name if desired
DATASET_FILE = h5py.File("example_data.hdf5", 'r')
PATCH_SIZE = (70, 70)       # modify depending on desired patch size
NUM_CLASSES = 2 # number of classes

# images numpy array should be of the shape: (number of images, image width, image height, 1)
# segs numpy array should be of the shape: (number of images, number of classes, image width)



# if you would like to generate patches differently, comment out these lines and write a replacement function
stride = 2
train_patches, train_patch_labels = load_patches("generated_patches/training", stride)
val_patches, val_patch_labels = load_patches("generated_patches/validation", stride)


train_patch_labels = keras.utils.to_categorical(train_patch_labels, num_classes=NUM_CLASSES)
val_patch_labels = keras.utils.to_categorical(val_patch_labels, num_classes=NUM_CLASSES)

train_patch_imdb = imdb.ImageDatabase(images=train_patches, labels=train_patch_labels, name=DATASET_NAME, filename=DATASET_NAME, mode_type='patch', num_classes=NUM_CLASSES)
val_patch_imdb = imdb.ImageDatabase(images=val_patches, labels=val_patch_labels, name=DATASET_NAME, filename=DATASET_NAME, mode_type='patch', num_classes=NUM_CLASSES)

# patch-based models from the "Automatic choroidal segmentation in OCT images using supervised deep learning methods" paper
model_cifar = patch_models.cifar_cnn(NUM_CLASSES, train_patches.shape[1], train_patches.shape[2])
model_complex = patch_models.complex_cnn(NUM_CLASSES, train_patches.shape[1], train_patches.shape[2])
model_rnn = patch_models.rnn_stack(4, ('ver', 'hor', 'ver', 'hor'), (True, True, True, True),
                               (CuDNNGRU, CuDNNGRU, CuDNNGRU, CuDNNGRU), (0.25, 0.25, 0.25, 0.25), (1, 1, 2, 2), (1, 1, 2, 2),
                               (16, 16, 16, 16), False, 0, INPUT_CHANNELS, train_patches.shape[1], train_patches.shape[2],
                               NUM_CLASSES)

opt_con = keras.optimizers.Adam
opt_params = {}     # default params
loss = keras.losses.categorical_crossentropy
metric = keras.metrics.categorical_accuracy
epochs = 100
batch_size = 1024

train_params = tparams.TrainingParams(model_complex, opt_con, opt_params, loss, metric, epochs, batch_size, model_save_best=True)

training.train_network(train_patch_imdb, val_patch_imdb, train_params)