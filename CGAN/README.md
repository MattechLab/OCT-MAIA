# Patch-based CGAN on OCT images 

# Dependencies
* Python 
* tensorflow 
* matplotlib

# Instructions
1. Train a conditional GAN on 70x70 patches using *cgan_70x70_patchbased.py*. Load data using the *load_data* function.
2. Evaluate trained generators of a GAN, using the Frechet Inception Distance (FID), using *evalfid_70x70_patchbased.py* and by specifying the folder containing the generators as *load_path*
3. Construct synthetic patches using *cgan_70x70_patchbased_genfid.py*. Save the generated images as required.
