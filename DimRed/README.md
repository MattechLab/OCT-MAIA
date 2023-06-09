# Dimensionality Reduction
The folder is structured as follows:

- [helpers.py](helpers.py): python file containing all the helper functions. 
- [tests.ipynb](tests.ipynb): jupyter notebook containing some tests to check the normalization. 
- [exploration.ipynb](exploration.ipynb): jupyter notebook exploring the thickness data. 
- [Stacked_Data.ipynb](Stacked_Data.ipynb): jupyter notebook exploring the stacked data, plotting the distribution of the thickness layers, and applying one model as test for the first time. 
- [Dim_Red_ModelComparisons.ipynb](Dim_Red_ModelComparisons.ipynb): jupyter notebook running a serie of dimensionality reduction models on the thickness layers in order to compare the results. 

  4 Model Comparisons on Thickness Layers
  - 4.1 One model study
    - 4.1.1 NMF model
    - 4.1.2 t-SNE
    - 4.1.3 UMAP
  - 4.2 compare supervised dim. reduction models
  
- [Dim_Red_GradientImage_ModelComparisons.ipynb](Dim_Red_GradientImage_ModelComparisons.ipynb): jupyter notebook running a serie of dimensionality reduction models on the gradient of the OCT images in order to compare the results. 
  
  5 Gradient Image: Model Comparisons
  - 5.1 Preprocess the OCTs
  - 5.2 One model study
      - 5.2.1 t-SNE
      - 5.2.2 UMAP
  - 5.3 compare supervised dim. reduction models
  
- [HorData.ipynb](HorData.ipynb): jupyter notebook considering the vectors as horizontal. Performs dimensionality reduction on the horizontal data, i.e. data with shape $(14\cdot 18 \cdot 5, 768)$.
  Table of contents: 

  6 Horizontal Data
  - 6.1: TSNE
      - 6.1.1: TSNE with different parameters
  - 6.2: UMAP
      - 6.2.1: UMAP with different parameters
  - 6.3: Layer-labeled horizontal data
      - 6.3.1: TSNE with layer labels
      - 6.3.2: UMAP with layer labels
  - 6.4 Dim. Red. on single layers
      - 6.4.1: TSNE on single layers
      - 6.4.2: UMAP on single layers

To summarize, the most important files are [Dim_Red_ModelComparisons.ipynb](Dim_Red_ModelComparisons.ipynb), [Dim_Red_GradientImage_ModelComparisons.ipynb](Dim_Red_GradientImage_ModelComparisons.ipynb) and [HorData.ipynb](HorData.ipynb), which compare models on thickness layers, gradient filtered scans and horizontal data, thanks to the helper functions in [helpers.py](helpers.py). The other files are for exploration of the data and testing.
