import os
import sys
sys.path.insert(1,"../../2_Preprocessing_Code/amd/prl")
import dataset
from buildDataset import upsample
from skimage import exposure
import pandas as pd 
import numpy as np 
import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,ConfusionMatrixDisplay,f1_score
from sklearn.decomposition import PCA, KernelPCA, NMF

def getStackedXYData(paths, normmode = 'Z-score', normmode_thick = 'EQ-hist'):
    """ 
    returns OCT and thickness in X, and corresponding labels in y. 
    """
    def normalize(img,normmode = 'Z-score'):
        if normmode == 'Z-score':
            img = dataset.z_score(img)
        elif normmode == 'EQ-hist': 
            img = exposure.equalize_hist(img)
        elif normmode == 'CLAHE':
            #TO Implement normalization by zone
            # Adaptive Equalization
            img = exposure.equalize_adapthist(img/max((img.max(),-img.min())), clip_limit=0.03)            
        return img
    
    def getxmin():
        themin = 100000
        for path in paths:
            PPocts = pd.read_pickle(path)
            for PPoct in PPocts:
                xnew = PPoct.image()
                if(xnew.shape[0] < themin):
                    themin = xnew.shape[0]
        return themin
    
    xmin_thick = 5 # number of layers -1
    xmin = getxmin()
    
    X = np.empty((0,xmin))
    X_thickness = np.empty((0,xmin_thick))
    y = np.empty((0,))
    xnew_thick = np.zeros((768, xmin_thick))

    for path in paths:
        PPocts = pd.read_pickle(path)
        for PPoct in PPocts:
            # since PPoct.thicknesses only gives the thickness of the 5 first layers (INFL, ONFL, IPL, OPL, ICL)
            # stack and normalize the thickness layers
            xnew_thick = np.transpose(PPoct.thicknesses())
            xnew = np.transpose(normalize(PPoct.image(),normmode = normmode))
            xnew = cv2.resize(xnew, dsize=(xmin,768 ))
            ynew = PPoct.label()
            #stack vertically the x and y
            X = np.vstack((X,xnew))
            X_thickness = np.vstack((X_thickness, xnew_thick))
            y = np.hstack((y,ynew))

    sum = np.sum(X_thickness, axis = 1)  
    xmax = sum.max()
    last_layer = xmax - sum
    X_thickness = np.hstack((last_layer.reshape(-1, 1), X_thickness))   
    # normalize the thicknesses layers   
    X_thickness = normalize(X_thickness, normmode=normmode_thick)
    X = np.hstack((X, X_thickness))
    return X,y

def getBalancedStackedXYData(normmode = 'EQ-hist', normmode_thick='EQ-hist'):
    rootpath = r'C:\Users\line\Desktop\Mauro\3_DataSet\OCT_balanced'
    paths = glob.glob(os.path.join(rootpath,'controlP' + '\*.pickle'))
    paths2 = glob.glob(os.path.join(rootpath,'amdP' + '\*.pickle'))
    merged = [[paths[2*i],paths[2*i+1],paths2[i]]  for i in range(len(paths2))]
    paths = [item for sublist in merged for item in sublist]
    return getStackedXYData(paths, normmode = normmode, normmode_thick = normmode_thick)

def NormalizeThickness(X, normmode=None, std_train=None, mean_train=None):
    """
    Normalize the vector of the thickness X with norm mode normmode. 
    if std_train is given, the normalisation will use that std. Same for mean_train. 
    """
    # if the arguments mean or std are given:
    if mean_train:
        mean = mean_train
    else:
        mean = X.mean()

    if std_train:
        std = std_train
    else:
        std = X.std()
    
    # go through the different normalization modes:
    if normmode == 'EUCLID':
        norms = np.linalg.norm(X, axis=1)
        indices_zero = np.isclose(norms, np.zeros(norms.shape[0]))
        norms = norms.reshape((norms.shape[0], 1))
        X[~indices_zero] = (1/norms[~indices_zero])*X[~indices_zero]

    elif normmode == 'Mean-std':
        X = (X-mean)/std

    elif normmode=='EUCLID-std':
        norms = np.linalg.norm(X, axis=1)
        indices_zero = np.isclose(norms, np.zeros(norms.shape[0]))
        norms = norms.reshape((norms.shape[0], 1))
        X[~indices_zero] = (1/norms[~indices_zero])*X[~indices_zero]

        if std_train:
            std = std_train
        else:
            std = X.std()

        X = X/std
    elif normmode == 'std':
        X = X/std

    return X

def getLabelledMatrices(X, y, normmode=None, std_train=None, mean_train=None):
    """ 
    Outputs matrices M0 and M1, concatenation of the vectors of the thickness data with label 0, resp. 1.
    The columns of the matrix X are normalized according to the argument normmode.

    INPUT:
    - X: thickness data
    - y: corresponding labels
    - normmode: norm to normalize the vectors
    - std_train: std of the training data, if X is a validation data.
    - mean_train: mean of the training data, if X is a validation data.
    OUTPUT:
    - M0: concatenation of the normalized vectors of the thickness data, with label 0
    - M1: same but with label 1.
    """
    norm_X = NormalizeThickness(X, normmode, std_train, mean_train)
    mask0 = np.array(1-y).astype(bool)
    mask1 = np.array(y).astype(bool)
    M0 = norm_X[mask0]
    M1 = norm_X[mask1]
    return M0, M1

def train_dim_red(X_train, y_train, normmode, dimred_model, params=None, plot=True, alpha=0.1, savefile=None):
    if savefile:
        print("START TRAINING:", savefile)
    print("training set:", X_train.shape[0])
    if normmode:
        print("Norm mode: " + normmode)
    else:
        print("No norm mode.")
    M0, M1 = getLabelledMatrices(X_train, y_train, normmode=normmode)
    print("M0.shape:", M0.shape)
    print("M1.shape:", M1.shape)
    print("M0.shape + M1.shape:", M0.shape[0] + M1.shape[0])

    
    
    if dimred_model == KernelPCA:
        # reduce the size, since the capacity computer cannot handle all the data. 
        size = 5000
        print("Kernel PCA. Reduce the training data from ", M0.shape[0], " to size:", size)
        random_indices = np.random.choice(M0.shape[0], size, replace=False)
        M0 = M0[random_indices]
        random_indices = np.random.choice(M1.shape[0], size, replace=False)
        M1 = M1[random_indices]

    # label 0
    print("LABEL 0:")
    if params:
        dimred_model0 = dimred_model(**params)
    else: 
        dimred_model0 = dimred_model()
    # train the first model on M0
    dimred_model0.fit(M0)
    if hasattr(dimred_model0, 'components_'):
        print("components:\n", dimred_model0.components_)
    if hasattr(dimred_model0, 'singular_values_'):
        print("singular values:", dimred_model0.singular_values_)
        if plot and savefile:
            fig = plt.figure(figsize=(5,5))
            ax = fig.gca()
            ax.plot(dimred_model0.singular_values_)
            fig.savefig("figures/train/sing_val/"+savefile)
    if hasattr(dimred_model0, 'n_features_'):
        print("n_features:", dimred_model0.n_features_)
 
    # label 1
    print("\nLABEL 1:")
    if params:
        dimred_model1 = dimred_model(**params)
    else: 
        dimred_model1 = dimred_model()
    # train the second model on M1
    dimred_model1.fit(M1)
    if hasattr(dimred_model1, 'components_'):
        print("components:\n", dimred_model1.components_)
    if hasattr(dimred_model1, 'singular_values_'):
        print("singular values:", dimred_model1.singular_values_)
    elif hasattr(dimred_model0, 'n_features_'):
        print("n_features:", dimred_model1.n_features_)

    if plot:
        print("plotting...")
        M0, M1 = getLabelledMatrices(X_train, y_train, normmode=normmode)
        # Project the 2 classes in the reduced spaces 0 and 1.
        M0_t0 = dimred_model0.transform(M0)
        M0_t1 = dimred_model1.transform(M0)
        M1_t1 = dimred_model1.transform(M1)
        M1_t0 = dimred_model0.transform(M1)

        # plot
        nb_pts_to_plot = 2000
        if (nb_pts_to_plot>M0.shape[0]) or (nb_pts_to_plot>M1.shape[0]):
            print(nb_pts_to_plot)
            print("M0.shape[0]=", M0.shape[0])
            print("M1.shape[0]", M1.shape[0])
            nb_pts_to_plot = np.min([M0.shape[0], M1.shape[0]])
            print(nb_pts_to_plot)

        random_indices0 = np.random.choice(M0.shape[0], nb_pts_to_plot, replace=False)
        random_indices1 = np.random.choice(M1.shape[0], nb_pts_to_plot, replace=False)

        fig, axes = plt.subplots(2, sharex=True, sharey=True)
        axes[0].scatter(M0_t0[random_indices0 ,0], M0_t0[random_indices0 ,1], c='g', marker='d', alpha=alpha)
        axes[0].scatter(M1_t0[random_indices1 ,0], M1_t0[random_indices1 ,1], c='r', marker='o', alpha=alpha)
        axes[0].legend(['proj. of M0 on M0', 'proj. of M1 on M0'])
        axes[0].set_title('M1 & M0, projected on M0')
        axes[1].scatter(M0_t1[random_indices0 ,0], M0_t1[random_indices0 ,1], c='y', marker='d', alpha=alpha)
        axes[1].scatter(M1_t1[random_indices1 ,0], M1_t1[random_indices1 ,1], c='b', marker='o', alpha=alpha)
        axes[1].legend(['proj. of M0 on M1', 'proj. of M1 on M1'])
        axes[1].set_title('M1 & M0, projected on M1')
        
        if savefile:
            fig.suptitle(savefile)
            fig.savefig("figures/train/" + savefile)


    return dimred_model0, dimred_model1

def plot_validation(X_valid, y_valid, normmode, dimred_model0, dimred_model1, std_train, mean_train, savefile=None,):
    M0_valid, M1_valid = getLabelledMatrices(X_valid, y_valid, normmode=normmode, std_train=std_train, mean_train=mean_train)

    M0_valid_t0 = dimred_model0.transform(M0_valid)
    M0_valid_t1 = dimred_model1.transform(M0_valid)

    M1_valid_t0 = dimred_model0.transform(M1_valid)
    M1_valid_t1 = dimred_model1.transform(M1_valid)

    nb_pts_to_plot = 2000
    if (nb_pts_to_plot>M0_valid.shape[0]) or (nb_pts_to_plot>M1_valid.shape[0]):
            print(nb_pts_to_plot)
            print("M0.shape[0]=", M0_valid.shape[0])
            print("M1.shape[0]", M1_valid.shape[0])
            nb_pts_to_plot = np.min([M0_valid.shape[0], M1_valid.shape[0]])
            print(nb_pts_to_plot)
    alpha = 0.1
    random_indices = np.random.choice(M1_valid.shape[0], nb_pts_to_plot, replace=False)

    fig, axes = plt.subplots(2, sharex=True, sharey=True)
    axes[0].scatter(M0_valid_t0[random_indices ,0], M0_valid_t0[random_indices ,1], c='g', marker='d', alpha=alpha)
    axes[0].scatter(M1_valid_t0[random_indices ,0], M1_valid_t0[random_indices ,1], c='r', marker='o', alpha=alpha)
    axes[0].legend(['proj. of M0 on M0', 'proj. of M1 on M0'])
    axes[1].scatter(M0_valid_t1[random_indices ,0], M0_valid_t1[random_indices ,1], c='y', marker='d', alpha=alpha)
    axes[1].scatter(M1_valid_t1[random_indices ,0], M1_valid_t1[random_indices ,1], c='b', marker='o', alpha=alpha)
    axes[1].legend(['proj. of M0 on M1', 'proj. of M1 on M1'])
    axes[0].set_title('M1 & M0, projected on M0')
    axes[1].set_title('M1 & M0, projected on M1')

    if savefile:
        fig.suptitle('Validation data, '+savefile)
        fig.savefig("figures/validation/" + savefile)


def plot_3D(X_valid, y_valid, normmode, dimred_model0, dimred_model1, std_train, mean_train, savefile=None, alpha=0.1,):
    
    M0_valid, M1_valid = getLabelledMatrices(X_valid, y_valid, normmode, std_train=std_train, mean_train=mean_train)

    M0_valid_t0 = dimred_model0.transform(M0_valid)
    M0_valid_t1 = dimred_model1.transform(M0_valid)

    M1_valid_t1 = dimred_model1.transform(M1_valid)
    M1_valid_t0 = dimred_model0.transform(M1_valid)
    
    if M0_valid_t0.shape[1] >=3:
        fig=plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(M0_valid_t1[:,0], M0_valid_t1[:,1], M0_valid_t1[:,2], c='y', alpha=alpha)
        ax.scatter(M1_valid_t1[:,0], M1_valid_t1[:,1], M1_valid_t1[:,2], c='b', alpha=alpha)
        ax.set_title("Projection of M0 and M1 on M1")
        ax.legend(['proj. of M0 on M1', 'proj. of M1 on M1'])
        fig.savefig("figures/3D_proj_on_M1")


        fig=plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(M0_valid_t0[:,0], M0_valid_t0[:,1], M0_valid_t0[:,2], c='g', alpha=alpha)
        ax.scatter(M1_valid_t0[:,0], M1_valid_t0[:,1], M1_valid_t0[:,2], c='r', alpha=alpha)
        ax.legend(['proj. of M0 on M0', 'proj. of M1 on M0'])
        ax.set_title("Projection of M0 and M1 on M0")
        if savefile:
            fig.savefig("figures/validation/3D/3D_proj_on_M0-" + savefile)

def plot_cv(dimred_model, X, y, dimred_params=None, dimred_normmode=None, cv=None, savefile=None, plot3D=False):
    """
    plot train and validation projections for the model dimred_model, data X and label y. 
    """
    
    for fold, (train_fold_index, val_fold_index) in enumerate(cv.split(X, y)):
        # Get the training data
        X_train_fold, y_train_fold = X[train_fold_index], y[train_fold_index]
        # Get the validation data
        X_val_fold, y_val_fold = X[val_fold_index], y[val_fold_index]

        # Upsample only the data in the training section
        X_train_fold_upsample, y_train_fold_upsample = X_train_fold,y_train_fold #before: upsample() 
        # apply dimension reduction model
        dimred0, dimred1 = train_dim_red(X_train_fold_upsample, y_train_fold_upsample, dimred_normmode, dimred_model, params=dimred_params, alpha=0.1, savefile=savefile+ "_fold_"+str(fold))
        std_train = X_train_fold.std()
        mean_train = X_train_fold.mean()
        plot_validation(X_val_fold, y_val_fold, dimred_normmode, dimred0, dimred1, std_train=std_train, mean_train=mean_train, savefile=savefile+"_fold_"+str(fold))
        if plot3D:
            plot_3D(X_val_fold, y_val_fold, dimred_normmode, dimred0, dimred1, std_train=std_train, mean_train=mean_train, alpha=0.1, savefile=savefile+ "_fold_"+str(fold))


def score_dimred_model_PR(model, dimred_model, X, y, params=None, dimred_params=None, dimred_normmode=None, cv=None, plotMatrix = None, plotROC = None, setThreshold = None,
                           ax2 = None, modelname = '', label = None, savefile=None):
    """
    TO COMPLETE LATER   
    Creates folds manually, and upsamples within each fold.
    Returns an array of validation (recall) scores
    """
    #smoter = SMOTE(random_state=42)
    
    scores = []
    y_real = []
    y_proba = []
    
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for fold, (train_fold_index, val_fold_index) in enumerate(cv.split(X, y)):
        # Get the training data
        X_train_fold, y_train_fold = X[train_fold_index], y[train_fold_index]
        # Get the validation data
        X_val_fold, y_val_fold = X[val_fold_index], y[val_fold_index]

        # Upsample only the data in the training section
        X_train_fold_upsample, y_train_fold_upsample = upsample(X_train_fold,y_train_fold) 

        # apply dimension reduction model
        dimred0, dimred1 = train_dim_red(X_train_fold_upsample, y_train_fold_upsample, dimred_normmode, dimred_model, params=dimred_params, alpha=0.1)
        X_train_fold_upsample_t0 = dimred0.transform(X_train_fold_upsample)
        X_val_fold_t0 = dimred0.transform(X_val_fold)

        # Fit the model on the upsampled training data
        if params:
            model_obj = model(**params).fit(X_train_fold_upsample_t0, y_train_fold_upsample)
        else:
            model_obj = model.fit(X_train_fold_upsample_t0, y_train_fold_upsample)
        pred_proba = model_obj.predict_proba(X_val_fold_t0)
        y_real.append(y_val_fold)
        preds = pred_proba[:,1]
        y_proba.append(preds)
        if(plotMatrix):
            # Display confusion matrix
            ConfusionMatrixDisplay.from_estimator(model_obj, X_val_fold_t0, y_val_fold)
            plt.show()
        if(plotROC):
            viz = PrecisionRecallDisplay.from_predictions(
                y_val_fold,
                preds,
                name=f"PR fold {fold}",
                alpha=0.3,
                lw=1,
                ax=ax,
            )

        if(setThreshold != None):
            predictions = (model_obj.predict_proba(X_val_fold_t0)[:,1] >= setThreshold).astype(bool)
        # Score the model on the (non-upsampled) validation data
        else:
            predictions = model_obj.predict(X_val_fold_t0)
        scores.append(recall_score(y_val_fold, predictions))
        scores.append(precision_score(y_val_fold, predictions))
        scores.append(accuracy_score(y_val_fold, predictions))
        scores.append(f1_score(y_val_fold, predictions))
        
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    if(plotROC):
        PrecisionRecallDisplay.from_predictions(
                    y_real,
                    y_proba,
                    name=f"PR AVG",
                    color="b",
                    lw=2,
                    alpha=0.8,
                    ax=ax,
                )
    if ax2:
        if label != None:
            PrecisionRecallDisplay.from_predictions(
                y_real,
                y_proba,
                name=f"{label}",
                lw=2,
                alpha=0.8,
                ax=ax2,
            )
        else:
            PrecisionRecallDisplay.from_predictions(
                y_real,
                y_proba,
                name=f"{str(list(params.items()))} {modelname}",
                lw=2,
                alpha=0.8,
                ax=ax2,
            )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="Precision",
        ylabel="Recall",
        title=f"Mean PR curve",
    )
    if savefile:
        fig.savefig("figures/PRcurve/"+savefile)
    plt.show()

    return np.array(scores)

# section 6: horizontal data (see HorData.ipynb)
def plot_proj_notraining(X, y, model, params, dimred_normmode, title, savefig):
    """
    Plot the dimensionality reduction results for model with parameters params on data X and y, for different norm modes dimred_normmode. 
    Models must have no training needed, like TSNE or UMAP. 
    """
    savefile_normmode = dimred_normmode.copy()
    mask0 = np.array(1-y).astype(bool)
    mask1 = np.array(y).astype(bool)
    dimred_model = model(**params)

    for j, normmode in enumerate(dimred_normmode):
        norm_X = NormalizeThickness(X, normmode)
        X_t = dimred_model.fit_transform(norm_X)
        # plot
        alpha = 0.1
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.scatter(X_t[mask0,0], X_t[mask0,1], c='g', marker='d', alpha=alpha, label='label 0')
        ax.scatter(X_t[mask1,0], X_t[mask1,1],  c='r', marker='o', alpha=alpha, label='label 1')
        ax.legend()
        ax.set_title(title + ' ' + normmode)
        fig.savefig(savefig + savefile_normmode[j], bbox_inches='tight')


def plot_per_layer(X, y, mask, model, params, dimred_normmode, title, savefig):
    """ 
    Per layer, plot the dim. red. model results for data X and labels y, for model with parameters params. Models don't have training and testing data, like TSNE or UMAP.
    y is of shape (N, 2), with y[:,0] containing labels for AMD or control, and y[:, 1] containing labels indicating the layer number.
    The mask is of shape (nbr of layers, y_shuffle.shape[0]), where mask[i,:] is a mask indicating if the data belongs to layer i or not.
    """
    alpha = 0.1
    dimred_model = model(**params)
    number_layers = mask.shape[0]
    savefile_normmode = dimred_normmode.copy()

    for j, normmode in enumerate(dimred_normmode):
        # number of subplots = number of layers. 
        fig, ax = plt.subplots(number_layers, figsize=(6,8))

        for i in range(number_layers):
            norm_X = NormalizeThickness(X[mask[i]], normmode)
            X_t = dimred_model.fit_transform(norm_X)
            # mask for AMD or control
            mask0 = np.array(1-y[mask[i], 0]).astype(bool)
            mask1 = np.array(y[mask[i], 0]).astype(bool)

            ax[i].scatter(X_t[mask0,0], X_t[mask0,1], c='g', marker='d', alpha=alpha, label='label 0')
            ax[i].scatter(X_t[mask1,0], X_t[mask1,1],  c='r', marker='d', alpha=alpha, label='label 1')
            ax[i].legend()
            ax[i].set_title('layer '+str(i+1))

        fig.suptitle(title + savefile_normmode[j])
        fig.tight_layout()
        fig.savefig(savefig + savefile_normmode[j], bbox_inches='tight')