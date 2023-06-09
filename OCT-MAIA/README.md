# OCT MAIA
Exploration of the data, preprocessing, classification and data augmentation. 

# Structure
Notebooks:
- [AMD_vs_control.ipynb](AMD_vs_control.ipynb): sanity check. Shows that there exists a model which can predict AMD vs control patient from the OCT scans. 
- [DataAugmentation.ipynb](DataAugmentation.ipynb): application of data augmentation on the OCT scans. Study the performance of the Naive Bayes classifier with and without augmented data. 
- [exploration.ipynb](exploration.ipynb): exploration of the data. Training of a classifier on edge-enhanced scans. 
- [ModelComparaisons.ipynb](ModelComparaisons.ipynb): comparisons of the results of simple classification models on the OCT scans. 

Helper files: 
- [buildDataset.py](buildDataset.py): helper functions to build the dataset. 