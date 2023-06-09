# Patch-based CGAN on OCT images 
The CGAN model was based on Kugelman et al. paper "Data augmentation for patch-based oct chorio-retinal segmentation using generative adversarial networks". Our code was based on their implementation that you can find on their [github repository](https://github.com/jakugel/oct-patchbased-cgan). 
# Dependencies
* Python 
* tensorflow 
* matplotlib

The file [tensorflow-env.yml](tensorflow-env.yml) is a working environment for training the CGAN. 

# Structure 
- `cgan_NxN_patchbased.py` trains a conditional CGAN on NxN patches.
- `evalfid_NxN_patchbased.py` evaluate trained generators of a GAN, using the Frechet Inception Distance. 
- `cgan_NxN_patchbased_genfid.py` constructs synthetic patches and save the images as required. 
- [patches.ipynb](patches.ipynb) is the notebook generating original patches of a given shape. 
- [script_evalfid.sh](script_evalfid.sh) and [script.sh](script.sh) are shells to launch the jobs on the clusters.
- [fid.py](fid.py), [Minibatch.py](Minibatch.py) and [training_patchbased.py](training_patchbased.py) are helpers for the CGAN training and the synthetic patches generation. They were given by Kugelman's et al. implementations.

## Example: instructions for 70x70 patches
Follow the following steps to train and generate synthetic patches:
1. Train a conditional GAN on 70x70 patches using *cgan_70x70_patchbased.py*. Load data using the *load_data* function.
2. Evaluate trained generators of a GAN, using the Frechet Inception Distance (FID), using *evalfid_70x70_patchbased.py* and by specifying the folder containing the generators as *load_path*
3. Construct synthetic patches using *cgan_70x70_patchbased_genfid.py*. Save the generated images as required.
