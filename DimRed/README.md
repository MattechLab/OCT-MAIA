# Dimensionality Reduction
The folder is structured as follows:

- `helpers.py`: python file containing all the helper functions. 
- `tests.ipynb`: jupyter notebook containing some tests to check the normalization. 
- `exploration.ipynb`: jupyter notebook exploring the thickness data. 
- `Stacked_Data.ipynb`: jupyter notebook exploring the stacked data, plotting the distribution of the thickness layers, and applying one model as test for the first time. 
- `Dim_Red_ModelComparisons.ipynb`: jupyter notebook running a serie of dimensionality reduction models on the thickness layers in order to compare the results. 
- `Dim_Red_GradientImage_ModelComparisons.ipynb`: jupyter notebook running a serie of dimensionality reduction models on the gradient of the OCT images in order to compare the results. 
- `HorData.ipynb`: jupyter notebook considering the vectors as horizontal. Performs dimensionality reduction on the horizontal data, i.e. data with shape $(14\cdot 18 \cdot 5, 768)$.
  Table of contents: 

  6. Horizontal Data
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

  
