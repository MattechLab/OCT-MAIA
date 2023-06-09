"""helper functions for generating patches (randomly, serially) and loading patches."""

import numpy as np
import os
import warnings
from sklearn.feature_extraction.image import extract_patches_2d


def create_random_patches(X, y, n_patches, d, thresh=0.8):
    """ 
    Extract n_patches random patches from OCT X with associated labels y. 
    Outputs patches of dimension d x d, with associated label. 
    Label of one patch is set with a threshold thresh. If none label has more than thresh*100 % of the labels, then the patch is discarded.
    Input: 
    - X: OCT images, dim (N, 70);
    - y: labels associated to OCT images, dim (N,);
    - n_patches: number of output patches;
    - d: dimension of the patches (d x d);
    - thresh: label i is chosen for a patch if it has more than thresh*100 %.
    Output: 
    - patches: images of dim (n_patches, dxd)
    - labels: labels of the corresponding patches, of dim (n_patches,)
    """
    
    # first channel is the OCT, second channel is the label.
    images = np.zeros((X.shape + (2,))) # shape of (X.shape, 2)
    images[:, :, 0] = X
    images[:, :, 1] = y.reshape((-1, 1))
    patches_labels = extract_patches_2d(image=images, patch_size=(d, d), max_patches=n_patches, random_state=42)
    
    # for each patch, extract one and only one label (either 0: not in the BCEA, or 1: in the BCEA) with respect to the threshold thresh.
    labels = patches_labels[:, :, 0, 1] # shape: n_patches x d
    patches = patches_labels[:, :, :, 0] # shape: n_patches x d x d

    n_in_bcea = np.sum(labels, axis=1)
    ratio_in_bcea = n_in_bcea/d
    final_labels = np.zeros(n_patches)
    final_labels[ratio_in_bcea>thresh] = 1

    # only select patches and labels with a ratio over thresh*100 percent
    mask = (ratio_in_bcea<1-thresh) | (ratio_in_bcea > thresh)
    final_labels = final_labels[mask]
    final_patches = patches[mask]
    
    print(f"{n_patches-final_labels.shape[0]} patches were dropped because of their ratio.")
    print(f"There are {int(np.sum(final_labels))} labels equal to 1 (in the BCEA) and {int(np.sum(1-final_labels))} labels equal to 0.")
    return final_patches, final_labels


def create_serial_patches(X, y, s, d=70, thresh=0.8, save_folder = None):
    """ 
    Extract serial patches with stride s from OCT X of dimension (N_OCT, 768, 70) with associated labels y of dimension (N_OCT, 768). 
    The patches are next to each other, separated with a stride s.
    Outputs patches of dimension d x d, with associated label. 
    Label of one patch is set with a threshold thresh. If none label has more than thresh*100 % of the labels, then the patch is discarded.
    Input: 
    - X: OCT images, dim (N_OCT, 768, 70), with N_OCT the number of OCT images;
    - y: labels associated to OCT images, dim (N_OCT, 768);
    - s: stride in between the patches;
    - d: dimension of the patches (d x d);
    - thresh: label i is chosen for a patch if it has more than thresh*100 %.
    - save_folder: if the argument is a string, save the labels and patches in the folder save_folder
    Output: 
    - final_patches: images of dim (n_tot, dxd)
    - final_labels: labels of the corresponding patches, of dim (n_tot,)
    """

    if(d > X.shape[2]):
        warnings.warn("d cannot be bigger than the last dimension of X. Now d is set to X.shape[2].")
        d = X.shape[2]
    
    if((thresh>1) | (thresh<0)):
        warnings.warn("threshold must be between 0 and 1. thresh is set to 0.8.")
        thresh = 0.8

    if(s>d):
        warnings.warn(f"The stride s={s} is bigger than the dimension of the patch d={d}.")
        
    print("patches of size:", d, "x", d)
    print("stride of ", s)
    print("threshold:", thresh)
    # number of OCT images:
    N_OCT = X.shape[0]
    # number of patches per OCT:
    n_patches_OCT = int((X.shape[1]-d)/s)
    # total number of patches: 
    n_tot = N_OCT*n_patches_OCT
    print("number of patches per OCT:", n_patches_OCT)
    print("total number of patches: ", n_tot)

    patches = np.zeros((n_tot, d, d))
    labels = np.zeros((n_tot, d))

    # create the patches and the corresponding labels:
    for k_oct in range(N_OCT):
        for k_patch in range(n_patches_OCT):
            patches[k_oct*n_patches_OCT + k_patch, :, :] = X[k_oct, s*k_patch:d+s*k_patch, :d]
            labels[k_oct*n_patches_OCT + k_patch, :] = y[k_oct, s*k_patch:d+s*k_patch]

    
    # for each patch, extract one and only one label (either 0: not in the BCEA, or 1: in the BCEA) with respect to the threshold thresh.
    n_in_bcea = np.sum(labels, axis=1)
    ratio_in_bcea = n_in_bcea/d
    final_labels = np.zeros(n_tot)
    final_labels[ratio_in_bcea>thresh] = 1

    # only select patches and labels with a ratio over thresh*100 percent
    mask = (ratio_in_bcea < 1-thresh) | (ratio_in_bcea > thresh)
    final_labels = final_labels[mask]
    final_patches = patches[mask]
    
    print(f"{n_tot-final_labels.shape[0]} patches were dropped because of their ratio.")
    print(f"There are {int(np.sum(final_labels))} labels equal to 1 (in the BCEA) and {int(np.sum(1-final_labels))} labels equal to 0.")

    if type(save_folder) == str:
        file_name_labels = save_folder + '/thresh=' + str(int(thresh*100)) + '/s=' + str(s) + '/labels.NPY'
        file_name_patches = save_folder + '/thresh=' + str(int(thresh*100)) +'/s=' + str(s) + '/patches.NPY'
        os.makedirs(os.path.dirname(file_name_labels), exist_ok=True)
        os.makedirs(os.path.dirname(file_name_patches), exist_ok=True)
        with open(file_name_labels, 'wb') as f:
            np.save(f , final_labels)
        with open(file_name_patches, 'wb') as f:
            np.save(f, final_patches)
            print("labels and patches saved in folder ", f)
    
    return final_patches, final_labels


def load_patches(save_folder, s, thresh=0.8):
    """
    loads the saved patches in folder save_folder, with stride s.
    Input:
    - save_folder: str, folder where the patches are saved,
    - s: stride,
    - thresh: treshold.
    Output:
    - loaded_patches: patches of dim (n_patches, dxd),
    - loaded_labels: labels of dim (n_patches,).
    """

    with open(save_folder + '/thresh=' + str(int(thresh*100)) + '/s=' + str(s) + '/labels.NPY', 'rb') as f:
        loaded_labels = np.load(f)
    with open(save_folder + '/thresh=' + str(int(thresh*100)) + '/s=' + str(s) + '/patches.NPY', 'rb') as f:
        loaded_patches = np.expand_dims(np.load(f), axis=3)
        
    return loaded_patches, loaded_labels